from scraper.sheets_writer import PriceRow, _find_date_row


def test_find_date_row_returns_existing_row_number() -> None:
    values = [
        "Date",
        "23/06/2026",
        "24/06/2026",
    ]
    assert _find_date_row(values, "24-06-2026") == 3


def test_find_date_row_returns_none_when_date_is_missing() -> None:
    values = [
        "Date",
        "23/06/2026",
    ]
    assert _find_date_row(values, "24-06-2026") is None


def test_price_row_only_sets_date_and_column_b() -> None:
    values = PriceRow(date="24/06/2026", price=135).values()
    assert len(values) == 2
    assert values[0] == "24/06/2026"
    assert values[1] == 135
