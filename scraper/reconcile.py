from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

from scraper.manual_override import parse_date
from scraper.sheets_writer import get_price_for_date

IST = ZoneInfo("Asia/Kolkata")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify that Manual Input has today's confirmed KPTA price.")
    parser.add_argument("--date", help="Date in DD/MM/YYYY or YYYY-MM-DD format; defaults to today in IST")
    args = parser.parse_args(argv)
    date_value = parse_date(args.date) if args.date else datetime.now(IST).strftime("%d/%m/%Y")

    price = get_price_for_date(date_value)
    if price is None:
        raise SystemExit(f"No KPTA value exists in Manual Input for {date_value}")
    try:
        numeric_price = float(price.replace(",", ""))
    except ValueError as exc:
        raise SystemExit(f"KPTA value for {date_value} is not numeric: {price!r}") from exc
    if not 80 <= numeric_price <= 400:
        raise SystemExit(f"KPTA value for {date_value} is outside the expected range: {price!r}")

    print(f"Verified KPTA price for {date_value}: {price}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
