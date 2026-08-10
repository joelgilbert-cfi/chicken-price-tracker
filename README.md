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
4. Uses both template matching and KPTA-card structure detection: a vivid green header with a red price area beneath it.
5. Uses the detected page number to query the e-paper article API. An image is accepted only when OCR confirms `KPTA`, or confirms both the KPTA phone number and target date.
6. If the page API cannot verify an image, clicks the exact detected card and accepts the resulting detail image only after the same identity verification.
7. Saves a page crop for review when no verified high-resolution image is available, then fails safely. Browser screenshot crops are never OCR input for an automatic sheet update.
8. Runs compact and line OCR passes on the first red price row, then uses positional full-card OCR only when it can identify the top price row.
9. Writes only the confirmed date and price to Google Sheets.

If the page-specific API image selection fails, the scraper attempts the exact viewer-detail image. If that also cannot be verified as KPTA, it writes no sheet value and saves `artifacts/ocr/page_crop_review.png` for review.

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

## Reliability Operations

The daily workflow retries at the requested run time, then after 10, 20, and 40 minutes. It only writes a price after a verified KPTA image and a confirmed top-price result. A failed run leaves `Manual Input` unchanged and uploads debug artifacts.

To correct a confirmed historical value locally:

```powershell
python -m scraper.manual_override --date 05/08/2026 --price 138
```

The repository also includes a **KPTA Manual Price Override** GitHub workflow. Use it when the Google credentials exist only as GitHub secrets.

The **KPTA Daily Reconciliation** workflow verifies that today's `Manual Input` row contains a numeric price. Configure a second cron-job.org request at 11:00 AM Asia/Kolkata to dispatch this workflow. It does not change the sheet; a failed reconciliation is the review alert.
