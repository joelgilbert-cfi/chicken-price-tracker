from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime

import gspread
from google.oauth2.service_account import Credentials

LOGGER = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
DATE_COLUMN = 1
KPTA_PRICE_COLUMN = 2
KPTA_PRICE_COLUMN_LETTER = "B"
DEFAULT_WORKSHEET_NAME = "Manual Input"


@dataclass(frozen=True)
class PriceRow:
    date: str
    price: int | str

    def values(self) -> list[object]:
        row = [""] * KPTA_PRICE_COLUMN
        row[DATE_COLUMN - 1] = self.date
        row[KPTA_PRICE_COLUMN - 1] = self.price
        return row


def validate_sheet_env() -> tuple[str, str]:
    credentials_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    sheet_id = os.environ.get("GOOGLE_SHEET_ID")
    missing = [
        name
        for name, value in {
            "GOOGLE_CREDENTIALS_JSON": credentials_json,
            "GOOGLE_SHEET_ID": sheet_id,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    return credentials_json or "", sheet_id or ""


def upsert_price(row: PriceRow) -> None:
    credentials_json, sheet_id = validate_sheet_env()
    credentials_info = json.loads(credentials_json)
    credentials = Credentials.from_service_account_info(credentials_info, scopes=SCOPES)
    client = gspread.authorize(credentials)
    spreadsheet = client.open_by_key(sheet_id)
    worksheet_name = os.environ.get("GOOGLE_WORKSHEET_NAME", DEFAULT_WORKSHEET_NAME)
    worksheet = spreadsheet.worksheet(worksheet_name)

    date_values = worksheet.col_values(DATE_COLUMN)
    target_row_number = _find_date_row(date_values, row.date)
    if target_row_number is None:
        worksheet.append_row(row.values(), value_input_option="USER_ENTERED")
        LOGGER.info("Appended KPTA price for %s", row.date)
    else:
        worksheet.update(
            f"{KPTA_PRICE_COLUMN_LETTER}{target_row_number}",
            [[row.price]],
            value_input_option="USER_ENTERED",
        )
        LOGGER.info("Updated KPTA price in row %s for %s", target_row_number, row.date)


def _find_date_row(date_values: list[str], date_value: str) -> int | None:
    target = _normalize_date(date_value)
    for index, existing_date in enumerate(date_values[1:], start=2):
        if _normalize_date(existing_date) == target:
            return index
    return None


def _normalize_date(value: object) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    text = str(value).strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return text
