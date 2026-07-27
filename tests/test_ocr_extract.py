from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from scraper.exceptions import AmbiguousPriceError, PriceNotFoundError
from scraper.ocr_extract import (
    crop_top_price_region,
    extract_numbers,
    extract_price,
    extract_top_price_numbers,
    normalize_ocr_image_size,
)


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


def test_extract_numbers_corrects_noisy_ocr_170_run() -> None:
    assert extract_numbers("730470 700131 70165 4 20 340 1004 715") == [
        170,
        131,
        165,
        20,
        340,
        715,
    ]


def test_extract_top_price_numbers_accepts_leading_zero_ocr() -> None:
    assert extract_top_price_numbers("0170") == [170]


def test_extract_numbers_prefers_trailing_price_from_noisy_rupee_prefix() -> None:
    assert extract_numbers("30158 00136 700145 00210 803050325715") == [158, 136, 145, 210]


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


def test_extract_price_selects_today_top_broiler_price(monkeypatch, tmp_path: Path) -> None:
    image_path = _blank_png(tmp_path)

    def fake_image_to_data(*_args, **_kwargs):
        return {
            "text": ["30158", "00136", "700145", "00210", "803050325715"],
            "conf": ["63", "70", "68", "72", "55"],
            "left": [200, 200, 200, 200, 0],
            "top": [20, 80, 140, 200, 260],
            "width": [80, 80, 80, 80, 250],
            "height": [30, 30, 30, 30, 30],
        }

    monkeypatch.setattr("pytesseract.image_to_data", fake_image_to_data)
    result = extract_price(image_path, tmp_path / "artifacts")
    assert result.price == 158


def test_extract_price_selects_noisy_corrected_170_top_price(monkeypatch, tmp_path: Path) -> None:
    image_path = _blank_png(tmp_path)

    def fake_image_to_data(*_args, **_kwargs):
        return {
            "text": ["730470", "700131", "70165", "220", "340"],
            "conf": ["61", "72", "69", "70", "65"],
            "left": [200, 200, 200, 200, 80],
            "top": [20, 80, 140, 200, 260],
            "width": [90, 90, 90, 70, 60],
            "height": [30, 30, 30, 30, 30],
        }

    monkeypatch.setattr("pytesseract.image_to_data", fake_image_to_data)
    result = extract_price(image_path, tmp_path / "artifacts")
    assert result.price == 170


def test_extract_price_uses_full_card_to_resolve_merged_top_price_token(monkeypatch, tmp_path: Path) -> None:
    image_path = _blank_png(tmp_path)

    def fake_image_to_data(*_args, **kwargs):
        if "--psm 8" in kwargs["config"]:
            return {
                "text": ["180146"],
                "conf": ["70"],
                "left": [10],
                "top": [10],
                "width": [100],
                "height": [30],
            }
        return {
            "text": ["7300146", "700136", "700185"],
            "conf": ["75", "72", "71"],
            "left": [200, 200, 200],
            "top": [20, 80, 140],
            "width": [90, 90, 90],
            "height": [30, 30, 30],
        }

    monkeypatch.setattr("pytesseract.image_to_data", fake_image_to_data)

    result = extract_price(image_path, tmp_path / "artifacts")

    assert result.price == 146
    assert result.candidates == [146, 136, 185]


def test_normalize_ocr_image_size_upscales_small_images() -> None:
    image = Image.new("RGB", (300, 200), "white")
    normalized = normalize_ocr_image_size(image)
    assert normalized.width >= 600
    assert normalized.height > image.height


def test_crop_top_price_region_uses_green_kpta_header_not_whole_image_position() -> None:
    image = Image.new("RGB", (1_000, 1_000), "white")
    pixels = image.load()
    for y in range(420, 540):
        for x in range(80, 920):
            pixels[x, y] = (0, 220, 0)

    crop, left, top = crop_top_price_region(image)

    assert (left, top) == (550, 554)
    assert crop.size == (370, 72)


def test_crop_top_price_region_falls_back_when_green_header_is_absent() -> None:
    image = Image.new("RGB", (1_000, 1_000), "white")

    crop, left, top = crop_top_price_region(image)

    assert (left, top) == (660, 230)
    assert crop.size == (340, 110)


def _blank_png(tmp_path: Path) -> Path:
    path = tmp_path / "zoom.png"
    Image.new("RGB", (120, 80), "white").save(path)
    return path
