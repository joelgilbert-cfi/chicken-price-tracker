from __future__ import annotations

import json
from pathlib import Path

from scraper.main import write_expected_failure
from scraper.sheets_writer import PriceRow


def test_write_expected_failure_does_not_update_google_sheet(tmp_path: Path, monkeypatch) -> None:
    def fail_if_called(_row: PriceRow) -> None:
        raise AssertionError("Google Sheets must not be updated for an OCR failure")

    monkeypatch.setattr("scraper.main.upsert_price", fail_if_called)

    exit_code = write_expected_failure(
        PriceRow(date="27/07/2026", price="N/A"),
        tmp_path,
        "OCR_FAILED",
        "OCR candidates were ambiguous",
    )

    assert exit_code == 1
    assert json.loads((tmp_path / "final_status.json").read_text(encoding="utf-8")) == {
        "status": "OCR_FAILED",
        "exit_code": 1,
        "notes": "OCR candidates were ambiguous",
    }
