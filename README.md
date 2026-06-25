# KPTA Daily Price Scraper

Automated Python scraper for the Vijaya Vani Bengaluru e-paper KPTA poultry price block.

The scraper writes into the existing commodity history Google Sheet's `Manual Input` tab:

- Column A: `Date`
- Column B: `KPTA Rate (₹/kg)`

It does not create scraper-specific status, notes, OCR, or confidence columns in the sheet.

## Setup

1. Create a Google Cloud project and enable the Google Sheets API.
2. Create a service account, download its JSON key, and share the existing commodity price Google Sheet with the service account email as Editor.
3. Add these GitHub Actions secrets:
   - `GOOGLE_CREDENTIALS_JSON`
   - `GOOGLE_SHEET_ID`
4. Confirm the target worksheet tab is named `Manual Input`. If it is not, add a GitHub Actions secret or environment variable named `GOOGLE_WORKSHEET_NAME`.
5. Add at least three real KPTA crop templates to `scraper/templates/`.
6. Add real zoomed KPTA OCR fixtures under `tests/fixtures/` and extend `tests/test_ocr_extract.py` with expected prices.

## Local Run

```powershell
pip install -r requirements.txt
playwright install chromium
pytest tests/test_ocr_extract.py
$env:GOOGLE_CREDENTIALS_JSON = Get-Content -Raw C:\path\to\service-account.json
$env:GOOGLE_SHEET_ID = "your-sheet-id"
$env:GOOGLE_WORKSHEET_NAME = "Manual Input"
python -m scraper.main
```

Debug files are written under `artifacts/`.

## Scraper Flow

The scraper uses a hybrid flow:

1. Opens today's Bengaluru edition with Playwright.
2. Scans pages in the e-paper viewer using viewer controls.
3. Screenshots each page and runs template matching against KPTA header crops in `scraper/templates/`.
4. Once the KPTA block is found, uses the detected page number to query the e-paper article API for that specific page.
5. Downloads only the likely KPTA article image from that page, selected by strong green KPTA-header and red-price color signals.
6. Saves that clean article image to `artifacts/ocr/zoom.png`.
7. Upscales small API article images before OCR.
8. Extracts the KPTA chicken price and writes only the date and price to Google Sheets.

If the page-specific API image selection fails, the scraper falls back to a cropped page screenshot of the detected KPTA region.

If Cloudflare or a "verify you are human" page appears, the scraper stops and records a technical failure.

## Sheet Columns

The scraper expects the existing `Manual Input` worksheet layout from `Commodity Price History.xlsx`:

- Column A header: `Date`
- Column B header: `KPTA Rate (₹/kg)`

Rows are upserted by `Date`, so a manual rerun updates column B for today's row instead of appending a duplicate. If today's date does not exist, the scraper appends a new row with only column A and column B populated.

## Current Calibration Status

The code path and workflow are in place, but production accuracy depends on user-supplied real assets:

- KPTA template PNG crops from at least three different days.
- Zoomed KPTA OCR fixtures with known expected prices.
- A manual GitHub Actions run to confirm the live Vijaya Vani viewer selectors and page URL behavior.

The scraper rejects missing or ambiguous OCR rather than guessing.
