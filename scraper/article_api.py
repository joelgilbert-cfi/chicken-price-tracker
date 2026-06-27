from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

from scraper.detect_kpta import load_templates
from scraper.exceptions import KPTANotFoundError, SiteUnavailableError

LOGGER = logging.getLogger(__name__)

API_BASE_URL = "https://enewspapr.com/epaper-api/epaper-api.php"
ISSUE_PREFIX = "VVAANINEW_BEN"
ARTICLE_MATCH_THRESHOLD = 0.70
REQUEST_TIMEOUT_SECONDS = 20
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class ArticleCandidate:
    issue_id: str
    page_number: int
    article_id: str
    image_url: str
    image_path: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    template: str | None


@dataclass(frozen=True)
class PageArticleCandidate:
    issue_id: str
    page_number: int
    article_id: str
    image_url: str
    image_path: str
    green_ratio: float
    red_ratio: float
    score: float
    x1: int
    y1: int
    x2: int
    y2: int


KPTA_GREEN_RATIO_THRESHOLD = 0.05
KPTA_RED_RATIO_THRESHOLD = 0.04


def find_kpta_article_image(
    target_date: date,
    artifacts_dir: Path,
    *,
    threshold: float = ARTICLE_MATCH_THRESHOLD,
) -> Path:
    issue_id = build_issue_id(target_date)
    articles_dir = artifacts_dir / "articles"
    articles_dir.mkdir(parents=True, exist_ok=True)
    ocr_dir = artifacts_dir / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Trying article API for issue %s", issue_id)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json,text/plain,*/*"})

    issue_pages = fetch_issue_pages(session, issue_id)
    article_refs = flatten_article_refs(issue_id, issue_pages)
    if not article_refs:
        raise KPTANotFoundError(f"No article images returned for issue {issue_id}")

    templates = load_templates(Path(__file__).parent / "templates")
    candidates: list[ArticleCandidate] = []
    best: ArticleCandidate | None = None

    for article in article_refs:
        image_url = article["image_url"]
        article_id = article["article_id"]
        image_path = articles_dir / f"{sanitize_filename(article_id)}.jpg"
        try:
            download_image(session, image_url, image_path)
            match = best_template_match_multiscale(image_path, templates)
        except Exception as exc:
            LOGGER.warning("Skipping article %s: %s", article_id, exc)
            continue

        candidate = ArticleCandidate(
            issue_id=issue_id,
            page_number=int(article["page_number"]),
            article_id=article_id,
            image_url=image_url,
            image_path=str(image_path),
            confidence=float(match["confidence"]),
            x=int(match["x"]),
            y=int(match["y"]),
            width=int(match["width"]),
            height=int(match["height"]),
            template=match["template"],
        )
        candidates.append(candidate)
        if best is None or candidate.confidence > best.confidence:
            best = candidate

    write_article_candidates(articles_dir, candidates)
    if best is None or best.confidence < threshold:
        best_confidence = best.confidence if best else None
        raise KPTANotFoundError(
            f"No API article image matched KPTA above {threshold}; best={best_confidence}"
        )

    selected_path = Path(best.image_path)
    ocr_image_path = ocr_dir / "zoom.png"
    Image.open(selected_path).convert("RGB").save(ocr_image_path)
    (articles_dir / "selected_article.json").write_text(
        json.dumps(asdict(best), indent=2),
        encoding="utf-8",
    )
    LOGGER.info(
        "Selected API article %s on page %s with confidence %.3f",
        best.article_id,
        best.page_number,
        best.confidence,
    )
    return ocr_image_path


def build_issue_id(target_date: date) -> str:
    return f"{ISSUE_PREFIX}_{target_date.strftime('%Y%m%d')}"


def fetch_issue_pages(session: requests.Session, issue_id: str) -> list[dict[str, object]]:
    try:
        response = session.get(
            API_BASE_URL,
            params={
                "issueID": issue_id,
                "operation": "getAllArticleByArticleId",
                "cache": "random",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code >= 500:
            raise SiteUnavailableError(f"Article API returned HTTP {response.status_code}")
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise SiteUnavailableError(f"Article API request failed: {exc}") from exc
    except ValueError as exc:
        raise SiteUnavailableError(f"Article API returned invalid JSON: {exc}") from exc

    if not isinstance(payload, list):
        raise SiteUnavailableError("Article API returned an unexpected response shape")
    return payload


def flatten_article_refs(issue_id: str, pages: list[dict[str, object]]) -> list[dict[str, object]]:
    refs: list[dict[str, object]] = []
    for page in pages:
        try:
            page_number = int(str(page.get("pageno") or "0"))
        except ValueError:
            continue
        for item in page.get("Articles") or []:
            article = item.get("Article") if isinstance(item, dict) else None
            if not isinstance(article, dict):
                continue
            article_id = str(article.get("article_image_id") or "")
            image_url = str(article.get("r2imagename") or article.get("imagename") or "")
            if not article_id.startswith(issue_id) or not image_url:
                continue
            refs.append(
                {
                    "page_number": page_number,
                    "article_id": article_id,
                    "image_url": image_url,
                    "x1": _safe_int(article.get("x1")),
                    "y1": _safe_int(article.get("y1")),
                    "x2": _safe_int(article.get("x2")),
                    "y2": _safe_int(article.get("y2")),
                }
            )
    return refs


def find_kpta_article_image_on_page(
    target_date: date,
    page_number: int,
    artifacts_dir: Path,
) -> Path:
    issue_id = build_issue_id(target_date)
    articles_dir = artifacts_dir / "articles"
    articles_dir.mkdir(parents=True, exist_ok=True)
    ocr_dir = artifacts_dir / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Finding KPTA article image from API page %s for issue %s", page_number, issue_id)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json,text/plain,*/*"})
    article_refs = [
        article
        for article in flatten_article_refs(issue_id, fetch_issue_pages(session, issue_id))
        if article["page_number"] == page_number
    ]
    if not article_refs:
        raise KPTANotFoundError(f"No article images returned for issue {issue_id} page {page_number}")

    candidates: list[PageArticleCandidate] = []
    for article in article_refs:
        article_id = str(article["article_id"])
        image_url = str(article["image_url"])
        image_path = articles_dir / f"{sanitize_filename(article_id)}.jpg"
        try:
            download_image(session, image_url, image_path)
            green_ratio, red_ratio = measure_kpta_color_ratios(image_path)
        except Exception as exc:
            LOGGER.warning("Skipping API article %s during color validation: %s", article_id, exc)
            continue

        score = green_ratio * red_ratio
        candidates.append(
            PageArticleCandidate(
                issue_id=issue_id,
                page_number=page_number,
                article_id=article_id,
                image_url=image_url,
                image_path=str(image_path),
                green_ratio=green_ratio,
                red_ratio=red_ratio,
                score=score,
                x1=int(article.get("x1") or 0),
                y1=int(article.get("y1") or 0),
                x2=int(article.get("x2") or 0),
                y2=int(article.get("y2") or 0),
            )
        )

    payload = [asdict(candidate) for candidate in candidates]
    payload.sort(key=lambda item: item["score"], reverse=True)
    (articles_dir / "page_article_candidates.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    if not candidates:
        raise KPTANotFoundError(f"No API article images could be evaluated for page {page_number}")

    selected = select_kpta_page_article_candidate(candidates)
    if selected is None:
        best = max(candidates, key=lambda candidate: candidate.score)
        raise KPTANotFoundError(
            "No page API article had enough KPTA green-header and red-price color signal; "
            f"best green={best.green_ratio:.4f}, red={best.red_ratio:.4f}"
        )

    output_path = ocr_dir / "zoom.png"
    Image.open(selected.image_path).convert("RGB").save(output_path)
    (articles_dir / "selected_page_article.json").write_text(
        json.dumps(asdict(selected), indent=2),
        encoding="utf-8",
    )
    LOGGER.info(
        "Selected API page article %s with green_ratio=%.4f red_ratio=%.4f",
        selected.article_id,
        selected.green_ratio,
        selected.red_ratio,
    )
    return output_path


def select_kpta_page_article_candidate(
    candidates: list[PageArticleCandidate],
) -> PageArticleCandidate | None:
    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate.green_ratio >= KPTA_GREEN_RATIO_THRESHOLD
        and candidate.red_ratio >= KPTA_RED_RATIO_THRESHOLD
    ]
    if not valid_candidates:
        return None
    return max(valid_candidates, key=lambda candidate: candidate.score)


def measure_kpta_color_ratios(image_path: Path) -> tuple[float, float]:
    image = np.array(Image.open(image_path).convert("RGB"))
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([95, 255, 255]))
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 55, 40]), np.array([12, 255, 255])),
        cv2.inRange(hsv, np.array([168, 55, 40]), np.array([180, 255, 255])),
    )
    total_pixels = max(1, image.shape[0] * image.shape[1])
    return float(np.count_nonzero(green_mask) / total_pixels), float(np.count_nonzero(red_mask) / total_pixels)


def download_image(session: requests.Session, image_url: str, output_path: Path) -> None:
    response = session.get(image_url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def save_article_detail_image(payload: object, output_path: Path) -> str:
    image_url = extract_article_detail_image_url(payload)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    download_image(session, image_url, output_path)
    return image_url


def extract_article_detail_image_url(payload: object) -> str:
    if isinstance(payload, list) and payload:
        first = payload[0]
    elif isinstance(payload, dict):
        first = payload
    else:
        raise KPTANotFoundError("Article detail API returned no article records")

    if not isinstance(first, dict):
        raise KPTANotFoundError("Article detail API returned an unexpected record shape")

    for key in ("r2imagepath", "fallbackimagepath", "r2imagename", "imagename"):
        value = first.get(key)
        if isinstance(value, str) and value.startswith("http"):
            return value

    raise KPTANotFoundError("Article detail API response did not include an image URL")


def best_template_match_multiscale(
    image_path: Path,
    templates: list[tuple[str, np.ndarray]],
) -> dict[str, object]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise KPTANotFoundError(f"Could not read article image: {image_path}")

    best: dict[str, object] = {
        "confidence": -1.0,
        "x": 0,
        "y": 0,
        "width": 0,
        "height": 0,
        "template": None,
    }
    scales = (0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
    for template_name, template in templates:
        for scale in scales:
            resized = resize_template(template, scale)
            if resized.shape[0] > image.shape[0] or resized.shape[1] > image.shape[1]:
                continue
            result = cv2.matchTemplate(image, resized, cv2.TM_CCOEFF_NORMED)
            _, max_value, _, max_location = cv2.minMaxLoc(result)
            if max_value > float(best["confidence"]):
                best = {
                    "confidence": float(max_value),
                    "x": int(max_location[0]),
                    "y": int(max_location[1]),
                    "width": int(resized.shape[1]),
                    "height": int(resized.shape[0]),
                    "template": template_name,
                }
    return best


def resize_template(template: np.ndarray, scale: float) -> np.ndarray:
    width = max(1, int(template.shape[1] * scale))
    height = max(1, int(template.shape[0] * scale))
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(template, (width, height), interpolation=interpolation)


def write_article_candidates(articles_dir: Path, candidates: list[ArticleCandidate]) -> None:
    payload = [asdict(candidate) for candidate in candidates]
    payload.sort(key=lambda item: item["confidence"], reverse=True)
    (articles_dir / "article_candidates.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _safe_int(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0
