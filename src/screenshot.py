"""Playwright screenshot engine for TJStats Streamlit apps on Hugging Face Spaces."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from playwright.async_api import async_playwright, Page, TimeoutError as PwTimeout

from .config import SCREENSHOTS_DIR

log = logging.getLogger(__name__)

# How long to wait for a HF Space to wake up (cold start)
HF_WAKE_TIMEOUT = 90_000  # 90 s
# How long to wait for Streamlit content to render after interaction
RENDER_TIMEOUT = 30_000  # 30 s


async def _wait_for_streamlit(page: Page) -> None:
    """Wait until Streamlit's main content is loaded (spinner gone)."""
    # Wait for the Streamlit app iframe or main block
    try:
        # HF Spaces wrap Streamlit in an iframe
        iframe_el = await page.wait_for_selector(
            "iframe", timeout=HF_WAKE_TIMEOUT
        )
        if iframe_el:
            frame = await iframe_el.content_frame()
            if frame:
                # Wait inside the iframe for Streamlit content
                await frame.wait_for_selector(
                    "[data-testid='stAppViewContainer']",
                    timeout=RENDER_TIMEOUT,
                )
                # Wait for any spinners to disappear
                await frame.wait_for_selector(
                    "[data-testid='stSpinner']",
                    state="detached",
                    timeout=RENDER_TIMEOUT,
                )
                return
    except PwTimeout:
        pass

    # Fallback: direct Streamlit (no iframe)
    try:
        await page.wait_for_selector(
            "[data-testid='stAppViewContainer']",
            timeout=RENDER_TIMEOUT,
        )
        await page.wait_for_selector(
            "[data-testid='stSpinner']",
            state="detached",
            timeout=RENDER_TIMEOUT,
        )
    except PwTimeout:
        log.warning("Streamlit spinner wait timed out — screenshotting anyway")


async def _get_streamlit_frame(page: Page) -> Page:
    """Return the Streamlit frame (iframe content or page itself)."""
    try:
        iframe_el = await page.wait_for_selector("iframe", timeout=15_000)
        if iframe_el:
            frame = await iframe_el.content_frame()
            if frame:
                return frame
    except PwTimeout:
        pass
    return page


async def select_player_in_dropdown(
    frame: Page, player_name: str, dropdown_index: int = 0
) -> bool:
    """Select a player from a Streamlit selectbox by typing into it.

    Returns True if selection was made, False otherwise.
    """
    # Streamlit selectboxes use data-testid="stSelectbox"
    selectboxes = await frame.query_selector_all(
        "[data-testid='stSelectbox']"
    )
    if not selectboxes or dropdown_index >= len(selectboxes):
        log.warning("No selectbox found at index %d", dropdown_index)
        return False

    box = selectboxes[dropdown_index]
    # Click to open the dropdown
    await box.click()
    await asyncio.sleep(0.5)

    # Type the player name to filter
    input_el = await box.query_selector("input")
    if input_el:
        await input_el.fill(player_name)
        await asyncio.sleep(1)

        # Select the first matching option
        option = await frame.wait_for_selector(
            "[data-testid='stSelectboxOption']",
            timeout=5_000,
        )
        if option:
            await option.click()
            await asyncio.sleep(2)
            return True

    return False


async def take_screenshot(
    url: str,
    player_name: str | None = None,
    output_name: str = "screenshot",
    full_page: bool = False,
    dropdown_index: int = 0,
    clip_selector: str | None = None,
) -> Path:
    """Navigate to a TJStats HF Space, optionally select a player, and screenshot.

    Args:
        url: Full URL to the HF Space.
        player_name: If provided, select this player from the first dropdown.
        output_name: Base filename (without extension) for the screenshot.
        full_page: Capture the full scrollable page.
        dropdown_index: Which selectbox to use (0-based).
        clip_selector: CSS selector to screenshot instead of full page.

    Returns:
        Path to the saved PNG screenshot.
    """
    output_path = SCREENSHOTS_DIR / f"{output_name}.png"

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            device_scale_factor=2,  # retina-quality
        )
        page = await context.new_page()

        log.info("Navigating to %s", url)
        await page.goto(url, wait_until="domcontentloaded", timeout=HF_WAKE_TIMEOUT)
        await _wait_for_streamlit(page)

        frame = await _get_streamlit_frame(page)

        # Select player if requested
        if player_name:
            ok = await select_player_in_dropdown(frame, player_name, dropdown_index)
            if ok:
                log.info("Selected player: %s", player_name)
                # Wait for chart/content to re-render
                await asyncio.sleep(3)
                try:
                    await frame.wait_for_selector(
                        "[data-testid='stSpinner']",
                        state="detached",
                        timeout=RENDER_TIMEOUT,
                    )
                except PwTimeout:
                    pass
            else:
                log.warning("Could not select player %s", player_name)

        # Take the screenshot
        if clip_selector:
            el = await frame.wait_for_selector(clip_selector, timeout=10_000)
            if el:
                await el.screenshot(path=str(output_path))
            else:
                await page.screenshot(path=str(output_path), full_page=full_page)
        else:
            await page.screenshot(path=str(output_path), full_page=full_page)

        await browser.close()

    log.info("Screenshot saved to %s", output_path)
    return output_path
