from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

from scraper.exceptions import AmbiguousPriceError, PriceNotFoundError

LOGGER = logging.getLogger(__name__)

MIN_PRICE = 80
MAX_PRICE = 400


@dataclass(frozen=True)
class OcrResult:
    price: int
    raw_text: str
    confidence: float | None
    candidates: list[int]


@dataclass(frozen=True)
class OcrCandidate:
    value: int
    text: str
    confidence: float | None
    left: int | None = None
    top: int | None = None
    width: int | None = None
    height: int | None = None


def extract_price(image_path: Path, artifacts_dir: Path) -> OcrResult:
    ocr_dir = artifacts_dir / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)

    image = normalize_ocr_image_size(Image.open(image_path).convert("RGB"))
    image.save(ocr_dir / "zoom.png")
    red_mask = build_red_mask(image)
    Image.fromarray(red_mask).save(ocr_dir / "red_mask.png")

    ocr_image = Image.fromarray(red_mask)
    raw_text, candidates = run_tesseract(ocr_image)
    LOGGER.info("Raw OCR output: %r", raw_text)
    LOGGER.info("OCR candidate numbers: %s", [candidate.value for candidate in candidates])

    (ocr_dir / "raw_text.txt").write_text(raw_text, encoding="utf-8")
    (ocr_dir / "candidates.json").write_text(
        json.dumps([asdict(candidate) for candidate in candidates], indent=2),
        encoding="utf-8",
    )

    plausible = [candidate for candidate in candidates if MIN_PRICE <= candidate.value <= MAX_PRICE]
    selected = select_kpta_chicken_candidate(plausible)
    if selected is None:
        raise PriceNotFoundError("OCR did not return a plausible broiler wholesale price")

    selected_confidences = [
        candidate.confidence
        for candidate in plausible
        if candidate.value == selected.value and candidate.confidence is not None
    ]
    confidence = (
        round(sum(selected_confidences) / len(selected_confidences) / 100.0, 4)
        if selected_confidences
        else None
    )
    LOGGER.info("Accepted price: %s", selected.value)
    return OcrResult(
        price=selected.value,
        raw_text=raw_text,
        confidence=confidence,
        candidates=[candidate.value for candidate in plausible],
    )


def select_kpta_chicken_candidate(candidates: list[OcrCandidate]) -> OcrCandidate | None:
    if not candidates:
        return None

    distinct_values = sorted({candidate.value for candidate in candidates})
    if len(distinct_values) == 1:
        return candidates[0]

    positioned = [
        candidate
        for candidate in candidates
        if candidate.top is not None and candidate.left is not None and candidate.height is not None
    ]
    if len(positioned) < len(candidates):
        raise AmbiguousPriceError(f"OCR returned multiple plausible prices: {distinct_values}")

    ordered = sorted(positioned, key=lambda candidate: (candidate.top or 0, -(candidate.left or 0)))
    first = ordered[0]
    second = ordered[1]
    first_bottom = (first.top or 0) + max(first.height or 0, 1)
    if (second.top or 0) <= first_bottom:
        raise AmbiguousPriceError(f"OCR returned multiple plausible prices: {distinct_values}")

    return first


def build_red_mask(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lower_red_1 = np.array([0, 55, 40])
    upper_red_1 = np.array([12, 255, 255])
    lower_red_2 = np.array([168, 55, 40])
    upper_red_2 = np.array([180, 255, 255])
    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask_1, mask_2)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def normalize_ocr_image_size(image: Image.Image) -> Image.Image:
    if image.width >= 600:
        return image
    scale = max(2, round(600 / max(image.width, 1)))
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)


def run_tesseract(image: Image.Image) -> tuple[str, list[OcrCandidate]]:
    config = "--psm 6 -c tessedit_char_whitelist=0123456789"
    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config=config,
    )
    raw_parts: list[str] = []
    candidates: list[OcrCandidate] = []

    for index, text in enumerate(data.get("text", [])):
        value_text = str(text).strip()
        if not value_text:
            continue
        raw_parts.append(value_text)
        for number in extract_numbers(value_text):
            candidates.append(
                OcrCandidate(
                    value=number,
                    text=value_text,
                    confidence=_parse_confidence(_item_at(data.get("conf", []), index)),
                    left=_safe_int(_item_at(data.get("left", []), index)),
                    top=_safe_int(_item_at(data.get("top", []), index)),
                    width=_safe_int(_item_at(data.get("width", []), index)),
                    height=_safe_int(_item_at(data.get("height", []), index)),
                )
            )

    raw_text = " ".join(raw_parts)
    if not candidates:
        for number in extract_numbers(raw_text):
            candidates.append(OcrCandidate(value=number, text=str(number), confidence=None))
    return raw_text, candidates


def extract_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    for match in re.finditer(r"\d+", text):
        run = match.group(0)
        if 2 <= len(run) <= 3:
            numbers.append(int(run))
            continue
        numbers.extend(_extract_numbers_from_long_run(run))
    return numbers


def _extract_numbers_from_long_run(run: str) -> list[int]:
    if len(run) == 4:
        return []

    candidates: list[int] = []

    # Tesseract often reads Kannada "ರೂ." / punctuation before the price as
    # leading digits, e.g. "100142" for the visible price "142".
    if len(run) in (5, 6) and run.startswith(("10", "100", "00", "0")):
        value = int(run[-3:])
        if MIN_PRICE <= value <= MAX_PRICE:
            return [value]

    first_three = int(run[:3])
    if MIN_PRICE <= first_three <= MAX_PRICE:
        candidates.append(first_three)

    last_three = int(run[-3:])
    if MIN_PRICE <= last_three <= MAX_PRICE and last_three not in candidates:
        candidates.append(last_three)

    return candidates


def _parse_confidence(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _item_at(values: list[object], index: int) -> object | None:
    if index >= len(values):
        return None
    return values[index]


def _safe_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
