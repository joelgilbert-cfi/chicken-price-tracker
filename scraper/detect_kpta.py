from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from playwright.sync_api import Error, Page, TimeoutError

from scraper.browser import assert_no_security_challenge, dismiss_overlays, screenshot_page
from scraper.exceptions import ConfigurationError, KPTANotFoundError, SecurityChallengeError, SiteUnavailableError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionResult:
    page_number: int
    confidence: float
    x: int
    y: int
    width: int
    height: int
    page_screenshot: str
    zoom_screenshot: str


def scan_for_kpta(
    page: Page,
    edition_url: str,
    artifacts_dir: Path,
    *,
    target_date: date | None = None,
    max_pages: int = 16,
    threshold: float = 0.70,
) -> DetectionResult:
    del edition_url  # Keep the public call shape, but avoid direct page URL jumps.

    templates = load_templates(Path(__file__).parent / "templates")
    pages_dir = artifacts_dir / "pages"
    detection_dir = artifacts_dir / "detection"
    detection_dir.mkdir(parents=True, exist_ok=True)

    best_matches: list[dict[str, object]] = []

    for page_number in range(1, max_pages + 1):
        LOGGER.info("Scanning newspaper page %s", page_number)
        if page_number > 1:
            go_to_next_viewer_page(page, page_number)

        dismiss_overlays(page)
        assert_no_security_challenge(page)
        settle_viewer(page)

        screenshot_path = pages_dir / f"page_{page_number}.png"
        screenshot_page(page, screenshot_path)

        match = best_template_match(screenshot_path, templates)
        best_matches.append(
            {
                "page": page_number,
                "confidence": match["confidence"],
                "x": match["x"],
                "y": match["y"],
                "width": match["width"],
                "height": match["height"],
                "template": match["template"],
            }
        )
        _write_best_matches(best_matches, detection_dir)
        LOGGER.info("Best KPTA confidence on page %s: %.3f", page_number, match["confidence"])

        if match["confidence"] < threshold:
            continue

        selected = DetectionResult(
            page_number=page_number,
            confidence=float(match["confidence"]),
            x=int(match["x"]),
            y=int(match["y"]),
            width=int(match["width"]),
            height=int(match["height"]),
            page_screenshot=str(screenshot_path),
            zoom_screenshot=str(artifacts_dir / "ocr" / "zoom.png"),
        )
        _write_detection_overlay(selected, artifacts_dir / "detection" / "selected_match.png")
        capture_kpta_detail_view(page, selected, artifacts_dir / "ocr" / "zoom.png", target_date=target_date)
        LOGGER.info("Selected page %s with confidence %.3f", selected.page_number, selected.confidence)
        return selected

    raise KPTANotFoundError(f"No KPTA match met confidence threshold {threshold}")


def go_to_next_viewer_page(page: Page, target_page_number: int) -> None:
    LOGGER.info("Navigating to page %s using viewer controls", target_page_number)
    assert_no_security_challenge(page)
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(700)

    if click_page_number_control(page, target_page_number):
        page.wait_for_timeout(2_000)
        assert_no_security_challenge(page)
        return

    if click_next_control(page):
        page.wait_for_timeout(2_000)
        assert_no_security_challenge(page)
        return

    raise KPTANotFoundError(f"Could not navigate to page {target_page_number} using viewer controls")


def click_page_number_control(page: Page, page_number: int) -> bool:
    candidates = page.get_by_text(str(page_number), exact=True)
    try:
        count = min(candidates.count(), 20)
    except Error:
        return False

    for index in range(count):
        candidate = candidates.nth(index)
        try:
            box = candidate.bounding_box(timeout=1_000)
            if not box or not candidate.is_visible(timeout=1_000):
                continue
            if not _looks_like_viewer_nav_box(box):
                continue
            candidate.click(timeout=3_000)
            return True
        except Error:
            continue
    return False


def click_next_control(page: Page) -> bool:
    selectors = [
        "[aria-label*='Next']",
        "[title*='Next']",
        ".next",
        ".page-next",
        ".fa-chevron-right",
        ".fa-angle-right",
        "a:has-text('›')",
        "button:has-text('›')",
        "a:has-text('>')",
        "button:has-text('>')",
    ]
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if locator.count() and locator.is_visible(timeout=1_000):
                locator.click(timeout=3_000)
                return True
        except Error:
            continue

    viewport = page.viewport_size or {"width": 1920, "height": 1080}
    try:
        page.mouse.click(viewport["width"] * 0.62, viewport["height"] * 0.58)
        return True
    except Error:
        return False


def settle_viewer(page: Page) -> None:
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(700)
    page.evaluate("window.scrollTo(0, Math.min(document.body.scrollHeight / 3, 700))")
    page.wait_for_timeout(1_000)
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(700)


def capture_kpta_detail_view(
    page: Page,
    result: DetectionResult,
    output_path: Path,
    *,
    target_date: date | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if target_date is not None:
        try:
            from scraper.article_api import find_kpta_article_image_on_page

            find_kpta_article_image_on_page(target_date, result.page_number, output_path.parent.parent)
            LOGGER.info("Saved API KPTA page article image to %s", output_path)
            return
        except Exception as exc:
            LOGGER.warning("KPTA page API image selection failed; trying click response: %s", exc)

    LOGGER.info("Clicking detected KPTA block to resolve exact API article image")

    center_x = result.x + result.width / 2
    center_y = result.y + result.height / 2
    viewport = page.viewport_size or {"width": 1920, "height": 1080}
    scroll_y = max(0, int(center_y - viewport["height"] / 2))

    page.evaluate("(y) => window.scrollTo(0, y)", scroll_y)
    page.wait_for_timeout(500)
    click_y = center_y - scroll_y

    try:
        from scraper.article_api import save_article_detail_image

        with page.expect_response(
            lambda response: "operation=getArticleByArticleId" in response.url,
            timeout=6_000,
        ) as response_info:
            page.mouse.click(center_x, click_y)
        response = response_info.value
        image_url = save_article_detail_image(response.json(), output_path)
        LOGGER.info("Saved API KPTA article image to %s from %s", output_path, image_url)
        return
    except Error as exc:
        LOGGER.warning("KPTA API article response was not captured; falling back to page crop: %s", exc)
        crop_kpta_region_from_page(result, output_path)
        return
    except Exception as exc:
        LOGGER.warning("KPTA API image download failed; falling back to page crop: %s", exc)
        crop_kpta_region_from_page(result, output_path)
        return


def crop_kpta_region_from_page(result: DetectionResult, output_path: Path) -> None:
    image = Image.open(result.page_screenshot)
    left = max(0, int(result.x - result.width * 0.15))
    top = max(0, int(result.y - result.height * 0.30))
    right = min(image.width, int(result.x + result.width * 1.35))
    bottom = min(image.height, int(result.y + result.height * 6.4))
    crop = image.crop((left, top, right, bottom))
    crop = crop.resize((crop.width * 5, crop.height * 5), Image.Resampling.LANCZOS)
    crop.save(output_path)
    LOGGER.info("Saved fallback KPTA page crop to %s", output_path)


def load_templates(templates_dir: Path) -> list[tuple[str, np.ndarray]]:
    templates: list[tuple[str, np.ndarray]] = []
    template_paths = [
        path
        for pattern in ("*.png", "*.jpg", "*.jpeg")
        for path in templates_dir.glob(pattern)
    ]
    for path in sorted(template_paths):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None and image.size:
            templates.append((path.name, image))
    if not templates:
        raise ConfigurationError(
            f"No template image files found in {templates_dir}. Add at least 3 KPTA crops."
        )
    return templates


def best_template_match(
    screenshot_path: Path,
    templates: list[tuple[str, np.ndarray]],
) -> dict[str, object]:
    image = cv2.imread(str(screenshot_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise KPTANotFoundError(f"Could not read page screenshot: {screenshot_path}")

    best: dict[str, object] = {
        "confidence": -1.0,
        "x": 0,
        "y": 0,
        "width": 0,
        "height": 0,
        "template": None,
    }
    for template_name, template in templates:
        if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
            continue
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        _, max_value, _, max_location = cv2.minMaxLoc(result)
        if max_value > float(best["confidence"]):
            best = {
                "confidence": float(max_value),
                "x": int(max_location[0]),
                "y": int(max_location[1]),
                "width": int(template.shape[1]),
                "height": int(template.shape[0]),
                "template": template_name,
            }
    return best


def _write_best_matches(best_matches: list[dict[str, object]], detection_dir: Path) -> None:
    (detection_dir / "best_matches.json").write_text(
        json.dumps(best_matches, indent=2),
        encoding="utf-8",
    )


def _write_detection_overlay(result: DetectionResult, output_path: Path) -> None:
    image = Image.open(result.page_screenshot).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [
            result.x,
            result.y,
            result.x + result.width,
            result.y + result.height,
        ],
        outline="red",
        width=5,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    (output_path.parent / "selected_match.json").write_text(
        json.dumps(asdict(result), indent=2),
        encoding="utf-8",
    )


def _looks_like_viewer_nav_box(box: dict[str, float]) -> bool:
    x = box.get("x", 0)
    y = box.get("y", 0)
    width = box.get("width", 0)
    height = box.get("height", 0)
    return 450 <= x <= 1400 and (0 <= y <= 320 or 900 <= y <= 1250) and width <= 80 and height <= 80
