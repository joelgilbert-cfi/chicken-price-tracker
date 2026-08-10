from __future__ import annotations

import argparse
from datetime import datetime

from scraper.sheets_writer import PriceRow, upsert_price


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set one confirmed KPTA price in Manual Input.")
    parser.add_argument("--date", required=True, help="Date in DD/MM/YYYY or YYYY-MM-DD format")
    parser.add_argument("--price", required=True, type=int, help="Confirmed KPTA price")
    args = parser.parse_args(argv)

    if not 80 <= args.price <= 400:
        parser.error("--price must be between 80 and 400")

    normalized_date = parse_date(args.date)
    upsert_price(PriceRow(date=normalized_date, price=args.price))
    return 0


def parse_date(value: str) -> str:
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue
    raise argparse.ArgumentTypeError("date must be DD/MM/YYYY or YYYY-MM-DD")


if __name__ == "__main__":
    raise SystemExit(main())
