from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

from playwright.sync_api import Error, Page

from scraper.browser import dismiss_overlays, safe_click, screenshot_page
from scraper.exceptions import EditionNotFoundError, SiteUnavailableError

LOGGER = logging.getLogger(__name__)

EPAPER_HOME = "https://epaper.vijayavani.net/"
DIRECT_URL_TEMPLATE = (
    "https://epaper.vijayavani.net/edition/Bengaluru/VVAANINEW_BEN/"
    "VVAANINEW_BEN_{yyyymmdd}/page/1#"
)


def open_bengaluru_edition(page: Page, target_date: date, artifacts_dir: Path) -> str:
    direct_url = DIRECT_URL_TEMPLATE.format(yyyymmdd=target_date.strftime("%Y%m%d"))
    LOGGER.info("Trying direct Bengaluru edition URL: %s", direct_url)
    try:
        response = page.goto(direct_url, wait_until="domcontentloaded", timeout=45_000)
        dismiss_overlays(page)
        if response is not None and response.status >= 500:
            raise SiteUnavailableError(f"Direct edition returned HTTP {response.status}")
        if _looks_like_bengaluru_edition(page):
            screenshot_page(page, artifacts_dir / "edition_opened.png")
            LOGGER.info("Direct Bengaluru edition loaded")
            return page.url
        LOGGER.warning("Direct URL loaded but could not verify Bengaluru edition")
    except SiteUnavailableError:
        raise
    except Error as exc:
        LOGGER.warning("Direct edition URL failed: %s", exc)

    LOGGER.info("Falling back to homepage discovery")
    try:
        response = page.goto(EPAPER_HOME, wait_until="domcontentloaded", timeout=45_000)
    except Error as exc:
        raise SiteUnavailableError(f"Could not open e-paper homepage: {exc}") from exc

    if response is not None and response.status >= 500:
        raise SiteUnavailableError(f"Homepage returned HTTP {response.status}")

    dismiss_overlays(page)
    link = _find_bengaluru_link(page)
    if link is None:
        screenshot_page(page, artifacts_dir / "homepage_bengaluru_not_found.png")
        raise EditionNotFoundError("Bengaluru edition link not found on homepage")

    LOGGER.info("Opening discovered Bengaluru edition: %s", link)
    try:
        page.goto(link, wait_until="domcontentloaded", timeout=45_000)
        dismiss_overlays(page)
    except Error as exc:
        raise EditionNotFoundError(f"Could not open discovered Bengaluru edition: {exc}") from exc

    if not _looks_like_bengaluru_edition(page):
        screenshot_page(page, artifacts_dir / "edition_verification_failed.png")
        raise EditionNotFoundError("Discovered page did not verify as Bengaluru edition")

    screenshot_page(page, artifacts_dir / "edition_opened.png")
    return page.url


def _looks_like_bengaluru_edition(page: Page) -> bool:
    url = page.url.lower()
    if "bengaluru" in url or "vvaaninew_ben" in url:
        return True
    try:
        text = page.locator("body").inner_text(timeout=5_000).lower()
    except Error:
        return False
    return "bengaluru" in text and "namaste bengaluru" not in text


def _find_bengaluru_link(page: Page) -> str | None:
    candidates: list[tuple[int, str, str]] = []
    links = page.locator("a")
    try:
        count = links.count()
    except Error:
        return None

    for index in range(count):
        link = links.nth(index)
        try:
            text = link.inner_text(timeout=1_000).strip()
            href = link.get_attribute("href", timeout=1_000)
        except Error:
            continue
        haystack = f"{text} {href or ''}".lower()
        if "bengaluru" not in haystack or "namaste" in haystack:
            continue
        score = 2 if "vvaaninew_ben" in haystack else 1
        if href:
            candidates.append((score, text, urljoin(page.url, href)))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    LOGGER.info("Homepage Bengaluru candidates: %s", [(score, text) for score, text, _ in candidates[:5]])
    return candidates[0][2]

