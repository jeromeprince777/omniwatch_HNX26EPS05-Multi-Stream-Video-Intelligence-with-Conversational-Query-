# 07 — FastAPI backend, evidence endpoints, chat UI
**Read:** docs/00_PRD (FR-EVD, FR-UI), docs/01 (6, 8), docs/03.

**Task:**
1. `api/main.py` and `api/schemas.py` implementing every endpoint in docs/01 section 6, with pydantic response models, CORS limited to localhost, `/api/health` listing models, device and active fallbacks.
2. `evidence/thumbs.py` (full frame with bbox + crop) and `evidence/clips.py` (ffmpeg cut, +-2 s padding, H.264 re-encode, cache keyed by camera/start/end; verify the ffmpeg flags by actually producing a playable clip and checking its duration with ffprobe).
3. `web/` single-page UI (vanilla JS or a small framework, no CDN dependencies so it works offline): chat with result cards (thumbnail, camera label, local time with offset, score, matched attributes, play-clip button, mode badge fast/verified), camera wall, KB panel with edit/delete, ingest status bar. If the Stitch MCP is available, use it to explore the layout first, then implement the final UI by hand against the API.
4. **Clarification card:** camera thumbnails to select (multi-select), optional rectangle drawing on the selected camera thumbnail (store normalized coordinates), optional "same as" dropdown of known referents, shows the "Did you mean ...?" suggestion if present, Cancel button (non-persisting). After submit the original question's answer appears automatically.
5. Keep `scripts/serve.sh` working with one command.

**Acceptance:** with the Playwright MCP (if available) or a Playwright script: send a query, verify a result card and a playable clip (video element has non-zero duration); trigger a clarification, draw a region, submit, verify the answer card and the KB panel entry; reload the page and ask again -> no card. Attach screenshots to the DONE REPORT.
