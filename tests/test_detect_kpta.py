from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from PIL import Image

from scraper.detect_kpta import (
    DetectionResult,
    best_kpta_card_structure,
    capture_kpta_detail_view,
)
from scraper.exceptions import KPTANotFoundError


def test_capture_saves_detected_page_crop_for_review_when_api_image_is_not_verified(
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

    with pytest.raises(KPTANotFoundError, match="No verified high-resolution KPTA image"):
        capture_kpta_detail_view(None, result, output_path, target_date=date(2026, 7, 28))

    cropped = Image.open(output_path.with_name("page_crop_review.png"))
    assert cropped.size == (750, 1_675)


def test_structure_detector_prefers_kpta_green_header_over_teal_banner(tmp_path: Path) -> None:
    page = Image.new("RGB", (1_000, 1_000), "white")
    pixels = page.load()
    for y in range(30, 75):
        for x in range(10, 990):
            pixels[x, y] = (0, 170, 170)  # Blue-green newspaper banner.
    for y in range(700, 760):
        for x in range(80, 480):
            pixels[x, y] = (0, 220, 0)  # KPTA's vivid green header.
    for y in range(780, 950):
        for x in range(300, 450):
            pixels[x, y] = (220, 0, 0)  # Red price-column signal.

    path = tmp_path / "page.png"
    page.save(path)

    match = best_kpta_card_structure(path)

    assert match["method"] == "structure"
    assert match["confidence"] >= 0.70
    assert match["x"] == 80
    assert match["y"] == 700
