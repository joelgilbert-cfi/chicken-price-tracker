from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from scraper.exceptions import AmbiguousPriceError, PriceNotFoundError
from scraper.ocr_extract import extract_numbers, extract_price


def test_extract_numbers_filters_two_and_three_digit_values() -> None:
    assert extract_numbers("abc 135 rs 500 7 0400") == [135, 500]


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


def _blank_png(tmp_path: Path) -> Path:
    path = tmp_path / "zoom.png"
    Image.new("RGB", (120, 80), "white").save(path)
    return path
