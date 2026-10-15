import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True)
            print("CHROMIUM_AVAILABLE")
            await browser.close()
        except Exception as e:
            print("CHROMIUM_ERROR:", e)

asyncio.run(main())
