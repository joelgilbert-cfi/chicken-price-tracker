from __future__ import annotations

import pytest

from scraper import manual_override, reconcile
from scraper.sheets_writer import PriceRow


def test_manual_override_normalizes_date_and_upserts_confirmed_price(monkeypatch) -> None:
    written: list[PriceRow] = []
    monkeypatch.setattr("scraper.manual_override.upsert_price", written.append)

    assert manual_override.main(["--date", "2026-08-05", "--price", "138"]) == 0
    assert written == [PriceRow(date="05/08/2026", price=138)]


def test_reconcile_accepts_numeric_price(monkeypatch) -> None:
    monkeypatch.setattr("scraper.reconcile.get_price_for_date", lambda _date: "138")

    assert reconcile.main(["--date", "05/08/2026"]) == 0


def test_reconcile_rejects_missing_or_non_numeric_price(monkeypatch) -> None:
    monkeypatch.setattr("scraper.reconcile.get_price_for_date", lambda _date: None)
    with pytest.raises(SystemExit, match="No KPTA value"):
        reconcile.main(["--date", "05/08/2026"])

    monkeypatch.setattr("scraper.reconcile.get_price_for_date", lambda _date: "N/A")
    with pytest.raises(SystemExit, match="not numeric"):
        reconcile.main(["--date", "05/08/2026"])
