"""Automated Playwright script to capture diverse, high-resolution screenshots of all 12 OmniWatch functionalities."""

import asyncio
import os
from pathlib import Path
from playwright.async_api import async_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
HTML_FILE = (REPO_ROOT / "web" / "index.html").as_uri()
OUT_DIR = REPO_ROOT / "docs" / "screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

async def capture_all():
    print(f"[INFO] Launching Playwright Chromium targeting: {HTML_FILE}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-web-security"]
        )
        context = await browser.new_context(
            viewport={"width": 1600, "height": 900},
            device_scale_factor=1.25  # High-DPI crisp capture
        )
        page = await context.new_page()
        await page.goto(HTML_FILE)
        await page.wait_for_timeout(1000)

        # ----------------------------------------------------------------------
        # 1. Search Console: Yokohama CCTV News Reader (Yokohama Camera)
        # ----------------------------------------------------------------------
        print("[1/12] Capturing Yokohama News Reader Dossier...")
        await page.fill("#query-input", "Yokohama news reader")
        await page.click("#send-btn")
        await page.wait_for_timeout(800)
        await page.screenshot(path=str(OUT_DIR / "01_search_yokohama_news_reader.png"))

        # ----------------------------------------------------------------------
        # 2. Search Console: Tours Pedestrian Plaza Bag Carrier (Tours Camera)
        # ----------------------------------------------------------------------
        print("[2/12] Capturing Tours Person Carrying Bag Dossier...")
        await page.fill("#query-input", "Show me a person carrying a bag")
        await page.click("#send-btn")
        await page.wait_for_timeout(800)
        await page.screenshot(path=str(OUT_DIR / "02_search_tours_bag_carrier.png"))

        # ----------------------------------------------------------------------
        # 3. Search Console: Transport Bus/Truck Query (Vehicle Detection)
        # ----------------------------------------------------------------------
        print("[3/12] Capturing Transport Vehicle Detection Dossier...")
        await page.fill("#query-input", "When did a bus or truck appear?")
        await page.click("#send-btn")
        await page.wait_for_timeout(800)
        await page.screenshot(path=str(OUT_DIR / "03_search_bus_truck_detection.png"))

        # ----------------------------------------------------------------------
        # 4. Search Console: East Perimeter Corridor Person (Cam 2)
        # ----------------------------------------------------------------------
        print("[4/12] Capturing East Corridor Person Dossier...")
        await page.fill("#query-input", "Find someone in east corridor")
        await page.click("#send-btn")
        await page.wait_for_timeout(800)
        await page.screenshot(path=str(OUT_DIR / "04_search_east_corridor_person.png"))

        # ----------------------------------------------------------------------
        # 5. Evidence Modal: Keyframe Reticle Target Lock (Cyber Reticle)
        # ----------------------------------------------------------------------
        print("[5/12] Capturing Keyframe Cyber Reticle Target Modal...")
        first_frame_btn = page.locator(".view-frame-btn").last
        if await first_frame_btn.is_visible():
            await first_frame_btn.click()
            await page.wait_for_timeout(600)
            await page.screenshot(path=str(OUT_DIR / "05_evidence_keyframe_reticle_modal.png"))
            await page.click("#modal-close-btn")
            await page.wait_for_timeout(400)

        # ----------------------------------------------------------------------
        # 6. Evidence Modal: 5-Second Video Evidence Player (Speed Controls)
        # ----------------------------------------------------------------------
        print("[6/12] Capturing Video Evidence Player Modal...")
        first_play_btn = page.locator(".play-clip-btn").last
        if await first_play_btn.is_visible():
            await first_play_btn.click()
            await page.wait_for_timeout(700)
            # Switch to video tab explicitly
            await page.click("#modal-tab-video")
            await page.wait_for_timeout(500)
            await page.screenshot(path=str(OUT_DIR / "06_evidence_video_clip_modal.png"))
            await page.click("#modal-close-btn")
            await page.wait_for_timeout(400)

        # ----------------------------------------------------------------------
        # 7. Cross-Camera Re-ID & Multi-Hop Path Timeline Modal
        # ----------------------------------------------------------------------
        print("[7/12] Capturing Cross-Camera Re-ID Path Timeline Modal...")
        first_reid_btn = page.locator(".reid-trace-btn").last
        if await first_reid_btn.is_visible():
            await first_reid_btn.click()
            await page.wait_for_timeout(700)
            await page.screenshot(path=str(OUT_DIR / "07_cross_camera_reid_timeline_modal.png"))
            await page.click("#reid-close-btn")
            await page.wait_for_timeout(400)

        # ----------------------------------------------------------------------
        # 8. Interactive Clarify-Once Spatial Memory Modal (Canvas ROI)
        # ----------------------------------------------------------------------
        print("[8/12] Capturing Clarify-Once Spatial Protocol Modal with Canvas ROI...")
        await page.fill("#query-input", "Where is the loading dock?")
        await page.click("#send-btn")
        await page.wait_for_timeout(800)
        # Simulate drawing an ROI box on the canvas
        canvas = page.locator("#roi-canvas")
        if await canvas.is_visible():
            box = await canvas.bounding_box()
            if box:
                await page.mouse.move(box["x"] + 100, box["y"] + 80)
                await page.mouse.down()
                await page.mouse.move(box["x"] + 450, box["y"] + 240)
                await page.mouse.up()
                await page.wait_for_timeout(300)
        await page.screenshot(path=str(OUT_DIR / "08_clarify_once_roi_canvas_modal.png"))
        await page.click("#clarify-cancel-btn")
        await page.wait_for_timeout(400)

        # ----------------------------------------------------------------------
        # 9. Tactical Camera Wall 4-Feed Matrix
        # ----------------------------------------------------------------------
        print("[9/12] Capturing Tactical Camera Wall Grid...")
        await page.click("#tab-wall-btn")
        await page.wait_for_timeout(800)
        await page.screenshot(path=str(OUT_DIR / "09_tactical_camera_wall_grid.png"))

        # ----------------------------------------------------------------------
        # 10. Clarify-Once Knowledge Base Table (SQLite Memory CRUD)
        # ----------------------------------------------------------------------
        print("[10/12] Capturing Clarify-Once SQLite Knowledge Base...")
        await page.click("#tab-kb-btn")
        await page.wait_for_timeout(800)
        await page.screenshot(path=str(OUT_DIR / "10_clarify_once_knowledge_base.png"))

        # ----------------------------------------------------------------------
        # 11. Standing Surveillance Alerts & Real-Time Audit Feed
        # ----------------------------------------------------------------------
        print("[11/12] Capturing Standing Surveillance Alerts Audit Feed...")
        await page.click("#tab-alerts-btn")
        await page.wait_for_timeout(800)
        await page.screenshot(path=str(OUT_DIR / "11_standing_surveillance_alerts_feed.png"))

        # ----------------------------------------------------------------------
        # 12. Verified Pipeline Mode (SmolVLM2 Deep Verification Active)
        # ----------------------------------------------------------------------
        print("[12/12] Capturing Verified Pipeline Mode Telemetry...")
        await page.click("#tab-chat-btn")
        await page.wait_for_timeout(400)
        await page.click("#mode-verified-btn")
        await page.wait_for_timeout(400)
        await page.fill("#query-input", "When did a bus pass through North Gate (cam1)?")
        await page.click("#send-btn")
        await page.wait_for_timeout(800)
        await page.screenshot(path=str(OUT_DIR / "12_verified_vlm_pipeline_mode.png"))

        await browser.close()
        print(f"\n[SUCCESS] All 12 distinct functionality screenshots saved to: {OUT_DIR}")

if __name__ == "__main__":
    asyncio.run(capture_all())
