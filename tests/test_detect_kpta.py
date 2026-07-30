from __future__ import annotations

from datetime import date
from pathlib import Path

from PIL import Image

from scraper.detect_kpta import DetectionResult, capture_kpta_detail_view
from scraper.exceptions import KPTANotFoundError


def test_capture_uses_detected_page_crop_when_api_image_is_not_verified(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page_screenshot = tmp_path / "page.png"
    Image.new("RGB", (500, 500), "white").save(page_screenshot)
    output_path = tmp_path / "ocr" / "zoom.png"
    result = DetectionResult(
        page_number=12,
        confidence=0.91,
        x=100,
        y=100,
        width=100,
        height=50,
        page_screenshot=str(page_screenshot),
        zoom_screenshot=str(output_path),
    )

    def reject_api_image(*_args, **_kwargs) -> None:
        raise KPTANotFoundError("No API article passed KPTA identity verification")

    monkeypatch.setattr(
        "scraper.article_api.find_kpta_article_image_on_page",
        reject_api_image,
    )

    capture_kpta_detail_view(None, result, output_path, target_date=date(2026, 7, 28))

    cropped = Image.open(output_path)
    assert cropped.size == (750, 1_675)
