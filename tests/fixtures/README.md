# OCR fixture screenshots

Place real zoomed KPTA block screenshots here and add expected-price tests before trusting production OCR.

Recommended filenames:

- `kpta_zoom_expected_135.png`
- `kpta_zoom_expected_142.png`
- `kpta_zoom_expected_<price>.png`

The current unit tests validate strict parsing and ambiguity handling with mocked Tesseract output. Real OCR calibration still requires real screenshots from the e-paper.
