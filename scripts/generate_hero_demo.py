"""Generate an elite, grounded animated WebP demo for OmniWatch without hallucinations."""

import asyncio
from pathlib import Path
from PIL import Image
from playwright.async_api import async_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
HTML_FILE = (REPO_ROOT / "web" / "index.html").as_uri()
OUT_DIR = REPO_ROOT / "docs" / "screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HERO_WEBP = OUT_DIR / "demo_surveillance_session.webp"
DEMO_WEBP_2 = OUT_DIR / "09_interactive_demo_session.webp"

async def generate_hero_demo():
    print(f"[INFO] Generating verified hero demo animation from: {HTML_FILE}")
    frames = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-web-security"]
        )
        context = await browser.new_context(
            viewport={"width": 1400, "height": 800},
            device_scale_factor=1.0
        )
        page = await context.new_page()
        await page.goto(HTML_FILE)
        await page.wait_for_timeout(1000)

        # Slide 1: Welcome & Search Console Ready
        print("[Slide 1/9] Ready console...")
        img_bytes = await page.screenshot()
        frames.append(Image.open(io_bytes(img_bytes)).convert("RGB"))

        # Slide 2: Grounded Bus Search on Cam1
        print("[Slide 2/9] Executing verified bus query on cam1...")
        await page.fill("#query-input", "When did a bus pass through North Gate (cam1)?")
        await page.click("#send-btn")
        await page.wait_for_timeout(900)
        img_bytes = await page.screenshot()
        frames.append(Image.open(io_bytes(img_bytes)).convert("RGB"))

        # Slide 3: Keyframe Reticle Evidence Modal
        print("[Slide 3/9] Keyframe evidence modal...")
        first_frame_btn = page.locator(".view-frame-btn").last
        if await first_frame_btn.is_visible():
            await first_frame_btn.click()
            await page.wait_for_timeout(600)
            img_bytes = await page.screenshot()
            frames.append(Image.open(io_bytes(img_bytes)).convert("RGB"))

        # Slide 4: Video Player Modal
        print("[Slide 4/9] Video player modal...")
        await page.click("#modal-tab-video")
        await page.wait_for_timeout(600)
        img_bytes = await page.screenshot()
        frames.append(Image.open(io_bytes(img_bytes)).convert("RGB"))
        await page.click("#modal-close-btn")
        await page.wait_for_timeout(400)

        # Slide 5: Cross-Camera Re-ID Path Timeline Modal
        print("[Slide 5/9] Re-ID path timeline modal...")
        first_reid_btn = page.locator(".reid-trace-btn").last
        if await first_reid_btn.is_visible():
            await first_reid_btn.click()
            await page.wait_for_timeout(600)
            img_bytes = await page.screenshot()
            frames.append(Image.open(io_bytes(img_bytes)).convert("RGB"))
            await page.click("#reid-close-btn")
            await page.wait_for_timeout(400)

        # Slide 6: Clarify-Once Spatial Protocol Modal
        print("[Slide 6/9] Clarify-Once ROI canvas modal...")
        await page.fill("#query-input", "Where is the loading dock?")
        await page.click("#send-btn")
        await page.wait_for_timeout(800)
        canvas = page.locator("#roi-canvas")
        if await canvas.is_visible():
            box = await canvas.bounding_box()
            if box:
                await page.mouse.move(box["x"] + 100, box["y"] + 80)
                await page.mouse.down()
                await page.mouse.move(box["x"] + 450, box["y"] + 240)
                await page.mouse.up()
                await page.wait_for_timeout(300)
        img_bytes = await page.screenshot()
        frames.append(Image.open(io_bytes(img_bytes)).convert("RGB"))
        await page.click("#clarify-cancel-btn")
        await page.wait_for_timeout(400)

        # Slide 7: Tactical Camera Wall Grid
        print("[Slide 7/9] Camera Wall 4-feed grid...")
        await page.click("#tab-wall-btn")
        await page.wait_for_timeout(800)
        img_bytes = await page.screenshot()
        frames.append(Image.open(io_bytes(img_bytes)).convert("RGB"))

        # Slide 8: Clarify-Once SQLite Knowledge Base
        print("[Slide 8/9] Clarify-Once KB table...")
        await page.click("#tab-kb-btn")
        await page.wait_for_timeout(800)
        img_bytes = await page.screenshot()
        frames.append(Image.open(io_bytes(img_bytes)).convert("RGB"))

        # Slide 9: Standing Surveillance Alerts Feed
        print("[Slide 9/9] Standing alerts feed...")
        await page.click("#tab-alerts-btn")
        await page.wait_for_timeout(800)
        img_bytes = await page.screenshot()
        frames.append(Image.open(io_bytes(img_bytes)).convert("RGB"))

        await browser.close()

    print(f"[INFO] Assembling {len(frames)} frames into animated WebP...")
    # Resize frames slightly to optimize file size and crispness (1280x731)
    processed = []
    for f in frames:
        f_resized = f.resize((1280, 731), Image.Resampling.LANCZOS)
        processed.append(f_resized)

    # Save to both demo destinations
    processed[0].save(
        HERO_WEBP,
        save_all=True,
        append_images=processed[1:],
        duration=2200,  # 2.2 seconds per slide
        loop=0,
        quality=88,
        method=6
    )
    processed[0].save(
        DEMO_WEBP_2,
        save_all=True,
        append_images=processed[1:],
        duration=2200,
        loop=0,
        quality=88,
        method=6
    )

    size_mb = HERO_WEBP.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Animated hero demo saved ({size_mb:.2f} MB): {HERO_WEBP}")

def io_bytes(b):
    import io
    return io.BytesIO(b)

if __name__ == "__main__":
    asyncio.run(generate_hero_demo())
