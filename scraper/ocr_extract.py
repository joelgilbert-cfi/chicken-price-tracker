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

    top_crop, crop_left, crop_top = crop_top_price_region(image)
    top_crop.save(ocr_dir / "top_price_crop.png")
    top_red_mask = build_red_mask(top_crop)
    Image.fromarray(top_red_mask).save(ocr_dir / "top_price_red_mask.png")
    top_raw_text, top_candidates = run_tesseract(
        Image.fromarray(top_red_mask),
        config="--psm 8 -c tessedit_char_whitelist=0123456789",
        number_extractor=extract_top_price_numbers,
        offset_left=crop_left,
        offset_top=crop_top,
    )

    red_mask = build_red_mask(image)
    Image.fromarray(red_mask).save(ocr_dir / "red_mask.png")

    ocr_image = Image.fromarray(red_mask)
    raw_text, candidates = run_tesseract(ocr_image)
    LOGGER.info("Top price OCR output: %r", top_raw_text)
    LOGGER.info("Raw OCR output: %r", raw_text)
    if top_candidates:
        LOGGER.info("Top price OCR candidate numbers: %s", [candidate.value for candidate in top_candidates])
    LOGGER.info("OCR candidate numbers: %s", [candidate.value for candidate in candidates])

    (ocr_dir / "raw_text.txt").write_text(
        f"top_price={top_raw_text}\nfull={raw_text}\n",
        encoding="utf-8",
    )
    (ocr_dir / "candidates.json").write_text(
        json.dumps(
            {
                "top_price": [asdict(candidate) for candidate in top_candidates],
                "full": [asdict(candidate) for candidate in candidates],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    top_plausible = [
        candidate for candidate in top_candidates if MIN_PRICE <= candidate.value <= MAX_PRICE
    ]
    selected = select_kpta_chicken_candidate(top_plausible)
    plausible_source = top_plausible

    plausible = [candidate for candidate in candidates if MIN_PRICE <= candidate.value <= MAX_PRICE]
    if selected is None:
        selected = select_kpta_chicken_candidate(plausible)
        plausible_source = plausible
    if selected is None:
        raise PriceNotFoundError("OCR did not return a plausible broiler wholesale price")

    selected_confidences = [
        candidate.confidence
        for candidate in plausible_source
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
        raw_text=raw_text if not top_raw_text else f"{top_raw_text} {raw_text}".strip(),
        confidence=confidence,
        candidates=[candidate.value for candidate in plausible_source],
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


def crop_top_price_region(image: Image.Image) -> tuple[Image.Image, int, int]:
    header_bounds = find_kpta_header_bounds(image)
    if header_bounds is not None:
        header_left, header_top, header_right, header_bottom = header_bounds
        header_width = header_right - header_left
        header_height = header_bottom - header_top

        # The first KPTA price is just below the green header, on the right.
        # This remains true when another price card shares the downloaded image.
        left = int(header_left + header_width * 0.56)
        top = int(header_bottom + header_height * 0.12)
        right = header_right
        bottom = int(header_bottom + header_height * 0.72)
        LOGGER.info(
            "Cropping first KPTA price relative to green header: "
            "header=(%s,%s,%s,%s), crop=(%s,%s,%s,%s)",
            header_left,
            header_top,
            header_right,
            header_bottom,
            left,
            top,
            right,
            bottom,
        )
        return image.crop((left, top, right, bottom)), left, top

    width, height = image.size
    left = int(width * 0.66)
    top = int(height * 0.23)
    right = width
    bottom = int(height * 0.34)
    LOGGER.warning("KPTA green header was not detected; using legacy fixed top-price crop")
    return image.crop((left, top, right, bottom)), left, top


def find_kpta_header_bounds(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Find the bright green horizontal KPTA header in an article image."""
    rgb = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(
        hsv,
        np.array([35, 80, 70]),
        np.array([95, 255, 255]),
    )

    height, width = green_mask.shape
    min_coverage = max(1, int(width * 0.35))
    matching_rows = np.count_nonzero(green_mask, axis=1) >= min_coverage
    runs = _contiguous_true_runs(matching_rows)
    if not runs:
        return None

    # The header is a substantial horizontal green band. Select the strongest
    # one so isolated green text or images cannot become the crop anchor.
    candidates: list[tuple[int, int, int]] = []
    for start, end in runs:
        run_height = end - start
        if run_height < max(5, int(height * 0.01)):
            continue
        coverage = int(np.count_nonzero(green_mask[start:end]))
        candidates.append((coverage, start, end))
    if not candidates:
        return None

    _, start, end = max(candidates)
    header_pixels = green_mask[start:end] > 0
    columns = np.where(np.any(header_pixels, axis=0))[0]
    if len(columns) == 0:
        return None
    return int(columns[0]), start, int(columns[-1]) + 1, end


def _contiguous_true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


def normalize_ocr_image_size(image: Image.Image) -> Image.Image:
    if image.width >= 600:
        return image
    scale = max(2, round(600 / max(image.width, 1)))
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)


def run_tesseract(
    image: Image.Image,
    *,
    config: str = "--psm 6 -c tessedit_char_whitelist=0123456789",
    number_extractor=None,
    offset_left: int = 0,
    offset_top: int = 0,
) -> tuple[str, list[OcrCandidate]]:
    if number_extractor is None:
        number_extractor = extract_numbers

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
        for number in number_extractor(value_text):
            candidates.append(
                OcrCandidate(
                    value=number,
                    text=value_text,
                    confidence=_parse_confidence(_item_at(data.get("conf", []), index)),
                    left=_offset_int(_safe_int(_item_at(data.get("left", []), index)), offset_left),
                    top=_offset_int(_safe_int(_item_at(data.get("top", []), index)), offset_top),
                    width=_safe_int(_item_at(data.get("width", []), index)),
                    height=_safe_int(_item_at(data.get("height", []), index)),
                )
            )

    raw_text = " ".join(raw_parts)
    if not candidates:
        for number in number_extractor(raw_text):
            candidates.append(OcrCandidate(value=number, text=str(number), confidence=None))
    return raw_text, candidates


def extract_top_price_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    for match in re.finditer(r"\d+", text):
        run = match.group(0)
        if len(run) == 4 and run.startswith("0"):
            value = int(run[-3:])
            if MIN_PRICE <= value <= MAX_PRICE:
                numbers.append(value)
                continue
        numbers.extend(extract_numbers(run))
    return numbers


def extract_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    for match in re.finditer(r"\d+", text):
        run = match.group(0)
        if 2 <= len(run) <= 3:
            numbers.extend(_extract_numbers_from_short_run(run))
            continue
        numbers.extend(_extract_numbers_from_long_run(run))
    return numbers


def _extract_numbers_from_short_run(run: str) -> list[int]:
    value = int(run)
    if len(run) == 3 and run.startswith("0") and value < MIN_PRICE:
        corrected = value + 100
        if MIN_PRICE <= corrected <= MAX_PRICE:
            return [corrected]
    return [value]


def _extract_numbers_from_long_run(run: str) -> list[int]:
    if len(run) == 4:
        return []

    candidates: list[int] = []

    # Tesseract often reads Kannada "ರೂ." / punctuation before the price as
    # leading digits, e.g. "100142" for "142" and "30158" for "158".
    if len(run) in (5, 6) and run.startswith(("10", "100", "00", "0", "20", "30")):
        value = int(run[-3:])
        if MIN_PRICE <= value <= MAX_PRICE:
            return [value]

    # A red rupee marker can be read as leading junk, and a leading "1" in the
    # price can be read as "4", e.g. "730470" for the visible price "170".
    last_three = int(run[-3:])
    if len(run) == 6 and run.startswith("730") and 400 <= last_three <= 499:
        corrected = last_three - 300
        if MIN_PRICE <= corrected <= MAX_PRICE:
            candidates.append(corrected)

    first_three = int(run[:3])
    if MIN_PRICE <= first_three <= MAX_PRICE:
        candidates.append(first_three)

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


def _offset_int(value: int | None, offset: int) -> int | None:
    if value is None:
        return None
    return value + offset
