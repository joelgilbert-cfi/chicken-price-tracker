from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from playwright.sync_api import Error, Page

from scraper.browser import dismiss_overlays, screenshot_page
from scraper.exceptions import ConfigurationError, KPTANotFoundError

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
    max_pages: int = 16,
    threshold: float = 0.70,
) -> DetectionResult:
    templates = load_templates(Path(__file__).parent / "templates")
    pages_dir = artifacts_dir / "pages"
    detection_dir = artifacts_dir / "detection"
    detection_dir.mkdir(parents=True, exist_ok=True)

    best_matches: list[dict[str, object]] = []
    selected: DetectionResult | None = None

    for page_number in range(1, max_pages + 1):
        LOGGER.info("Scanning newspaper page %s", page_number)
        open_page_number(page, edition_url, page_number)
        dismiss_overlays(page)
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
        LOGGER.info("Best KPTA confidence on page %s: %.3f", page_number, match["confidence"])

        if match["confidence"] >= threshold and (
            selected is None or match["confidence"] > selected.confidence
        ):
            zoom_path = artifacts_dir / "ocr" / "zoom.png"
            selected = crop_zoom_region(
                screenshot_path,
                zoom_path,
                int(match["x"]),
                int(match["y"]),
                int(match["width"]),
                int(match["height"]),
                page_number,
                float(match["confidence"]),
            )

    (detection_dir / "best_matches.json").write_text(
        json.dumps(best_matches, indent=2),
        encoding="utf-8",
    )

    if selected is None:
        raise KPTANotFoundError(f"No KPTA match met confidence threshold {threshold}")

    _write_detection_overlay(selected, artifacts_dir / "detection" / "selected_match.png")
    LOGGER.info("Selected page %s with confidence %.3f", selected.page_number, selected.confidence)
    return selected


def load_templates(templates_dir: Path) -> list[tuple[str, np.ndarray]]:
    templates: list[tuple[str, np.ndarray]] = []
    for path in sorted(templates_dir.glob("*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None and image.size:
            templates.append((path.name, image))
    if not templates:
        raise ConfigurationError(
            f"No template PNG files found in {templates_dir}. Add at least 3 KPTA crops."
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


def open_page_number(page: Page, edition_url: str, page_number: int) -> None:
    target_url = _page_url(edition_url, page_number)
    try:
        page.goto(target_url, wait_until="domcontentloaded", timeout=45_000)
        page.wait_for_timeout(1_500)
    except Error as exc:
        raise KPTANotFoundError(f"Could not open page {page_number}: {exc}") from exc


def crop_zoom_region(
    page_screenshot: Path,
    zoom_path: Path,
    x: int,
    y: int,
    width: int,
    height: int,
    page_number: int,
    confidence: float,
) -> DetectionResult:
    zoom_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(page_screenshot)
    pad_x = max(40, width // 2)
    pad_y = max(40, height // 2)
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(image.width, x + width + pad_x)
    bottom = min(image.height, y + height + pad_y)
    crop = image.crop((left, top, right, bottom))
    crop.save(zoom_path)
    return DetectionResult(
        page_number=page_number,
        confidence=confidence,
        x=x,
        y=y,
        width=width,
        height=height,
        page_screenshot=str(page_screenshot),
        zoom_screenshot=str(zoom_path),
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


def _page_url(edition_url: str, page_number: int) -> str:
    if re.search(r"/page/\d+", edition_url):
        return re.sub(r"/page/\d+", f"/page/{page_number}", edition_url)
    return edition_url.rstrip("/#") + f"/page/{page_number}#"
