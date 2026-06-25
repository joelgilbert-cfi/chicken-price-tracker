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
                }
            )
    return refs


def download_image(session: requests.Session, image_url: str, output_path: Path) -> None:
    response = session.get(image_url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    output_path.write_bytes(response.content)


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
