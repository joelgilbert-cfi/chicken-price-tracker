from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import Browser, BrowserContext, Error, Locator, Page, TimeoutError

LOGGER = logging.getLogger(__name__)

DESKTOP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

AD_DOMAINS = {
    "doubleclick.net",
    "googlesyndication.com",
    "adservice.google.com",
    "amazon-adsystem.com",
    "ade.clmbtech.com",
    "pagead2.googlesyndication.com",
}


def launch_browser(playwright, *, headless: bool = True) -> tuple[Browser, BrowserContext, Page]:
    browser = playwright.chromium.launch(headless=headless)
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent=DESKTOP_USER_AGENT,
        device_scale_factor=1,
    )
    block_ad_domains(context)
    page = context.new_page()
    page.set_default_timeout(20_000)
    page.set_default_navigation_timeout(45_000)
    return browser, context, page


def block_ad_domains(context: BrowserContext) -> None:
    def route_handler(route) -> None:
        host = urlparse(route.request.url).hostname or ""
        if any(host == domain or host.endswith(f".{domain}") for domain in AD_DOMAINS):
            LOGGER.debug("Blocking ad request: %s", route.request.url)
            route.abort()
            return
        route.continue_()

    context.route("**/*", route_handler)


def dismiss_overlays(page: Page) -> None:
    for key in ("Escape", "Escape"):
        try:
            page.keyboard.press(key)
            page.wait_for_timeout(500)
        except Error:
            LOGGER.debug("Overlay dismissal key press failed", exc_info=True)

    selectors = [
        "button:has-text('Close')",
        "button:has-text('×')",
        "[aria-label='Close']",
        ".close",
        ".modal-close",
        ".popup-close",
    ]
    for selector in selectors:
        try:
            candidate = page.locator(selector).first
            if candidate.count() and candidate.is_visible(timeout=1_000):
                candidate.click(timeout=2_000)
                page.wait_for_timeout(500)
        except Error:
            continue


def safe_click(target: Locator | Page, selector: str | None = None, *, timeout_ms: int = 10_000) -> None:
    try:
        locator = target.locator(selector) if selector is not None else target
        locator.click(timeout=timeout_ms)
    except TimeoutError:
        page = target if isinstance(target, Page) else target.page
        LOGGER.warning("Click timed out, pressing Escape and retrying once")
        dismiss_overlays(page)
        locator = target.locator(selector) if selector is not None else target
        locator.click(timeout=timeout_ms)


def screenshot_page(page: Page, path: Path, *, full_page: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(path), full_page=full_page)

