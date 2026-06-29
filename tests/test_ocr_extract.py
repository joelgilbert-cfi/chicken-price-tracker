from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from scraper.exceptions import AmbiguousPriceError, PriceNotFoundError
from scraper.ocr_extract import extract_numbers, extract_price, normalize_ocr_image_size


def test_extract_numbers_filters_two_and_three_digit_values() -> None:
    assert extract_numbers("abc 135 rs 500 7 0400") == [135, 500]


def test_extract_numbers_handles_noisy_concatenated_ocr_runs() -> None:
    assert extract_numbers("2 100142 6044 140438 2400 27540296") == [142, 140, 275, 296]


def test_extract_numbers_corrects_missing_hundreds_digit() -> None:
    assert extract_numbers("053 702136 700145 200 290 31 0 100 715") == [
        153,
        136,
        145,
        200,
        290,
        31,
        100,
        715,
    ]


def test_extract_price_accepts_single_plausible_candidate(monkeypatch, tmp_path: Path) -> None:
    image_path = _blank_png(tmp_path)

    def fake_image_to_data(*_args, **_kwargs):
        return {
            "text": ["", "135"],
            "conf": ["-1", "91"],
            "left": [0, 10],
            "top": [0, 12],
            "width": [0, 30],
            "height": [0, 20],
        }

    monkeypatch.setattr("pytesseract.image_to_data", fake_image_to_data)
    result = extract_price(image_path, tmp_path / "artifacts")
    assert result.price == 135
    assert result.confidence == 0.91
    assert (tmp_path / "artifacts" / "ocr" / "raw_text.txt").exists()
    assert (tmp_path / "artifacts" / "ocr" / "candidates.json").exists()


def test_extract_price_rejects_no_plausible_candidate(monkeypatch, tmp_path: Path) -> None:
    image_path = _blank_png(tmp_path)

    def fake_image_to_data(*_args, **_kwargs):
        return {
            "text": ["42", "500"],
            "conf": ["80", "80"],
            "left": [0, 10],
            "top": [0, 12],
            "width": [0, 30],
            "height": [0, 20],
        }

    monkeypatch.setattr("pytesseract.image_to_data", fake_image_to_data)
    with pytest.raises(PriceNotFoundError):
        extract_price(image_path, tmp_path / "artifacts")


def test_extract_price_rejects_ambiguous_candidates(monkeypatch, tmp_path: Path) -> None:
    image_path = _blank_png(tmp_path)

    def fake_image_to_data(*_args, **_kwargs):
        return {
            "text": ["135", "142"],
            "conf": ["88", "87"],
            "left": [0, 40],
            "top": [0, 0],
            "width": [30, 30],
            "height": [20, 20],
        }

    monkeypatch.setattr("pytesseract.image_to_data", fake_image_to_data)
    with pytest.raises(AmbiguousPriceError):
        extract_price(image_path, tmp_path / "artifacts")


def test_extract_price_selects_corrected_top_price(monkeypatch, tmp_path: Path) -> None:
    image_path = _blank_png(tmp_path)

    def fake_image_to_data(*_args, **_kwargs):
        return {
            "text": ["053", "702136", "700145", "200"],
            "conf": ["52", "48", "49", "51"],
            "left": [100, 100, 100, 100],
            "top": [10, 60, 110, 160],
            "width": [50, 90, 90, 50],
            "height": [20, 20, 20, 20],
        }

    monkeypatch.setattr("pytesseract.image_to_data", fake_image_to_data)
    result = extract_price(image_path, tmp_path / "artifacts")
    assert result.price == 153


def test_normalize_ocr_image_size_upscales_small_images() -> None:
    image = Image.new("RGB", (300, 200), "white")
    normalized = normalize_ocr_image_size(image)
    assert normalized.width >= 600
    assert normalized.height > image.height


def _blank_png(tmp_path: Path) -> Path:
    path = tmp_path / "zoom.png"
    Image.new("RGB", (120, 80), "white").save(path)
    return path
