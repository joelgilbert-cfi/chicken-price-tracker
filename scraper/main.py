from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from playwright.sync_api import Error as PlaywrightError, sync_playwright

from scraper.article_api import find_kpta_article_image
from scraper.browser import launch_browser
from scraper.detect_kpta import DetectionResult, scan_for_kpta
from scraper.edition import open_bengaluru_edition
from scraper.exceptions import (
    AmbiguousPriceError,
    ConfigurationError,
    EditionNotFoundError,
    KPTANotFoundError,
    PriceNotFoundError,
    SecurityChallengeError,
    SiteUnavailableError,
)
from scraper.ocr_extract import OcrResult, extract_price
from scraper.sheets_writer import PriceRow, upsert_price, validate_sheet_env

IST = ZoneInfo("Asia/Kolkata")
ARTIFACTS_DIR = Path("artifacts")

LOGGER = logging.getLogger(__name__)


def main() -> int:
    configure_logging()
    artifacts_dir = ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(IST)
    LOGGER.info("Starting KPTA scraper for IST date %s", now.strftime("%d-%m-%Y"))
    write_json(artifacts_dir / "run_info.json", {"started_at_ist": format_datetime(now)})

    try:
        validate_sheet_env()
    except Exception:
        LOGGER.exception("Required environment validation failed")
        write_final_status(artifacts_dir, "TECHNICAL_ERROR", 1, "Missing required environment variables")
        return 1

    try:
        ocr = run_api_pipeline(now, artifacts_dir)
        source = "API"
    except (KPTANotFoundError, PriceNotFoundError, AmbiguousPriceError, SiteUnavailableError) as exc:
        LOGGER.warning("API pipeline did not produce a price; falling back to Playwright: %s", exc)
        try:
            ocr = run_playwright_pipeline(now, artifacts_dir)
            source = "Playwright"
        except KPTANotFoundError as fallback_exc:
            LOGGER.exception("KPTA block not found")
            row = build_sheet_row(now, price="N/A")
            return write_expected_failure(row, artifacts_dir, "CONTENT_MISS", str(fallback_exc))
        except (PriceNotFoundError, AmbiguousPriceError) as fallback_exc:
            LOGGER.exception("OCR failed")
            row = build_sheet_row(now, price="N/A")
            return write_expected_failure(row, artifacts_dir, "OCR_FAILED", str(fallback_exc))
        except (
            ConfigurationError,
            SecurityChallengeError,
            SiteUnavailableError,
            EditionNotFoundError,
            PlaywrightError,
        ) as fallback_exc:
            LOGGER.exception("Technical scraper failure")
            row = build_sheet_row(now, price="N/A")
            return write_technical_failure(row, artifacts_dir, str(fallback_exc))
    except ConfigurationError as exc:
        LOGGER.exception("Technical scraper failure")
        row = build_sheet_row(now, price="N/A")
        return write_technical_failure(row, artifacts_dir, str(exc))

    try:
        row = build_sheet_row(
            now,
            price=ocr.price if ocr else "N/A",
        )
        upsert_price(row)
        write_final_status(artifacts_dir, "OK", 0, f"Success via {source}")
        return 0
    except Exception as exc:
        LOGGER.exception("Unexpected scraper failure")
        (artifacts_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        row = build_sheet_row(now, price="N/A")
        return write_technical_failure(row, artifacts_dir, str(exc))


def run_api_pipeline(now: datetime, artifacts_dir: Path) -> OcrResult:
    image_path = find_kpta_article_image(now.date(), artifacts_dir)
    return extract_price(image_path, artifacts_dir)


def run_playwright_pipeline(now: datetime, artifacts_dir: Path) -> OcrResult:
    detection: DetectionResult | None = None
    with sync_playwright() as playwright:
        browser, _context, page = launch_browser(playwright)
        try:
            edition_url = open_bengaluru_edition(page, now.date(), artifacts_dir)
            detection = scan_for_kpta(page, edition_url, artifacts_dir)
            return extract_price(Path(detection.zoom_screenshot), artifacts_dir)
        finally:
            browser.close()


def build_sheet_row(
    now: datetime,
    *,
    price: int | str,
) -> PriceRow:
    return PriceRow(
        date=now.strftime("%d/%m/%Y"),
        price=price,
    )


def write_expected_failure(row: PriceRow, artifacts_dir: Path, status: str, notes: str) -> int:
    try:
        upsert_price(row)
    except Exception:
        LOGGER.exception("Failed writing expected failure row to Google Sheets")
        write_final_status(artifacts_dir, "TECHNICAL_ERROR", 1, "Google Sheets write failed")
        return 1
    write_final_status(artifacts_dir, status, 0, notes)
    return 0


def write_technical_failure(row: PriceRow, artifacts_dir: Path, notes: str) -> int:
    del row
    LOGGER.info("Skipping Google Sheet update for technical failure")
    write_final_status(artifacts_dir, "TECHNICAL_ERROR", 1, notes)
    return 1


def write_final_status(artifacts_dir: Path, status: str, exit_code: int, notes: str) -> None:
    write_json(
        artifacts_dir / "final_status.json",
        {"status": status, "exit_code": exit_code, "notes": notes},
    )
    LOGGER.info("Final status=%s exit_code=%s", status, exit_code)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def format_datetime(value: datetime) -> str:
    return value.strftime("%d-%m-%Y %H:%M:%S")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
