"""Browser automation via Playwright.

Provides:
  - open_url: open a URL in the default browser
  - search_and_get_first_result: web search and extract the top link
  - click_element: click a DOM element by ARIA role or text
"""
from __future__ import annotations

import asyncio
from typing import Optional

from nova.core.logger import get_logger

logger = get_logger(__name__)


async def open_url_in_browser(url: str, new_tab: bool = True) -> bool:
    """Open *url* in the system browser via Playwright.

    Attempts to reuse an existing browser window; falls back to launching
    a new one.

    Args:
        url: Fully-qualified URL to navigate to.
        new_tab: If True, open in a new tab rather than the current tab.

    Returns:
        True on success, False on any error.
    """
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=15_000)
            logger.info("Opened URL: %s", url)
            # Keep browser open; real Nova would reuse a persistent context
            return True
    except Exception as exc:
        logger.error("Playwright open_url failed: %s", exc)
        return False


async def open_url_and_get_first_result(query: str) -> Optional[str]:
    """Perform a DuckDuckGo search and return the first organic result URL.

    Args:
        query: Search query string.

    Returns:
        URL string of the first result, or None if extraction fails.
    """
    search_url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}"
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(search_url, wait_until="networkidle", timeout=20_000)

            # DuckDuckGo result links are anchors with data-testid="result-title-a"
            first_link = page.locator('[data-testid="result-title-a"]').first
            href = await first_link.get_attribute("href", timeout=5_000)
            await browser.close()

            if href:
                logger.info("First search result: %s", href)
                return href
    except Exception as exc:
        logger.error("search_and_get_first_result failed: %s", exc)
    return None


async def click_element_by_text(page_url: str, text: str) -> bool:
    """Navigate to *page_url* and click the first element containing *text*.

    Args:
        page_url: URL of the page.
        text: Visible text of the element to click.

    Returns:
        True if clicked successfully, False otherwise.
    """
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=False)
            page = await browser.new_page()
            await page.goto(page_url, wait_until="domcontentloaded")
            await page.get_by_text(text).first.click()
            logger.info("Clicked element with text '%s' on %s", text, page_url)
            return True
    except Exception as exc:
        logger.error("click_element_by_text failed: %s", exc)
        return False


async def wait_and_get_page_title(url: str) -> Optional[str]:
    """Navigate to *url* and return the page title after load."""
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="load", timeout=15_000)
            title = await page.title()
            await browser.close()
            return title
    except Exception as exc:
        logger.error("wait_and_get_page_title failed: %s", exc)
        return None
