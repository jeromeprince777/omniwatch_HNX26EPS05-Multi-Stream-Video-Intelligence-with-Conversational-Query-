"""FastAPI Server Entry Point for OmniWatch.

Exposes REST endpoints defined in docs/01_ARCHITECTURE.md section 6:
- POST /api/chat
- POST /api/clarify
- GET  /api/cameras
- GET  /api/results/{tracklet_id}/thumb
- GET  /api/results/{tracklet_id}/clip
- GET  /api/kb
- PUT  /api/kb/{referent_id}
- DELETE /api/kb/{referent_id}
- GET  /api/health
- Static frontend mounted at /
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import omni_min
from omniwatch.api.schemas import (
    CameraCard,
    CameraInfo,
    ChatRequest,
    ChatResponse,
    ClarifyRequest,
    HealthResponse,
    LatencyBreakdown,
    MatchedAttributes,
    ResultItem,
)
from omniwatch.config import load_config
from omniwatch.db import get_connection
from omniwatch.query.answer import build_answer_summary, format_timestamp
from omniwatch.query.parser import (
    ParsedQuery,
    get_last_parser_used,
    get_parser_health_status,
    parse_query,
)
from omniwatch.query.resolver import resolve_referents, resolve_time

logger = logging.getLogger(__name__)


def _select_site_id(conn: Any, parsed: ParsedQuery) -> str | None:
    """Select the site named by the query, returning None if unconstrained across all sites."""
    cur = conn.cursor()
    if parsed.camera_ids:
        placeholders = ",".join("?" for _ in parsed.camera_ids)
        cur.execute(
            f"SELECT DISTINCT site_id FROM cameras WHERE camera_id IN ({placeholders}) ORDER BY site_id",
            parsed.camera_ids,
        )
        sites = [row[0] for row in cur.fetchall()]
        if len(sites) == 1:
            return sites[0]

    phrases = list(parsed.location_phrases)
    if phrases:
        normalized_phrases = []
        for phrase in phrases:
            lowered = phrase.lower().strip()
            normalized_phrases.extend((lowered, lowered.replace("_", " ")))
        normalized_phrases = list(dict.fromkeys(normalized_phrases))
        placeholders = ",".join("?" for _ in normalized_phrases)
        cur.execute(
            f"SELECT DISTINCT site_id FROM referents WHERE phrase_norm IN ({placeholders}) AND site_id IS NOT NULL",
            normalized_phrases,
        )
        referent_sites = [row[0] for row in cur.fetchall()]
        if len(referent_sites) == 1:
            return referent_sites[0]

        cur.execute("SELECT site_id, camera_id, label FROM cameras ORDER BY site_id, camera_id")
        camera_rows = cur.fetchall()
        candidates: set[str] = set()
        for phrase in phrases:
            normalized = phrase.lower().replace("_", " ").strip()
            phrase_tokens = set(normalized.split())
            for site_id, camera_id, label in camera_rows:
                names = {
                    camera_id.lower().replace("_", " "),
                    (label or "").lower(),
                    site_id.lower().replace("_", " "),
                }
                if any(
                    normalized == name
                    or normalized in name
                    or name in normalized
                    or phrase_tokens.intersection(name.split())
                    for name in names
                    if name
                ):
                    candidates.add(site_id)
        if len(candidates) == 1:
            return next(iter(candidates))

    return None




def create_app() -> FastAPI:
    app = FastAPI(title="OmniWatch API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Ensure evidence directories exist
    cfg = load_config()
    Path(cfg.storage.evidence_cache_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.storage.evidence_cache_dir) / "clips").mkdir(parents=True, exist_ok=True)
    (Path(cfg.storage.evidence_cache_dir) / "thumbs").mkdir(parents=True, exist_ok=True)

    @app.get("/api/health", response_model=HealthResponse)
    def health_check() -> HealthResponse:
        current_cfg = load_config()
        device = current_cfg.resolve_device()

        models = {
            "detector": "YOLOv8s (Closed-set 10 COCO classes)",
            "encoder": "OpenCLIP ViT-L-14 (laion2b_s32b_b82k)",
            "tracker": "ByteTrack (bytetrack.yaml)",
            "backend": "omni_min",
            "verifier": "Deterministic Grounded Attributes + Calibrated CLIP",
            "query_parser": "RuleQueryParser (primary) / Ollama (fallback)",
        }

        fallbacks: list[str] = []
        if device == "cpu":
            fallbacks.append("cpu_inference_fallback")

        obs_count = 0
        try:
            with omni_min.db() as con:
                row = con.execute("SELECT COUNT(*) FROM obs").fetchone()
                obs_count = row[0] if row else 0
        except Exception as exc:
            logger.warning("Error fetching health index counts from omni_min: %s", exc)

        return HealthResponse(
            models=models,
            fallbacks_active=fallbacks,
            index_counts={"observations": obs_count, "tracklets": obs_count},
            device=device,
            verifier_status="grounded_omni_min",
            parser_status="rules",
            offline_ready=True,
            privacy_blur_enabled=False,
        )

    @app.get("/api/cameras", response_model=list[CameraInfo])
    def list_cameras() -> list[CameraInfo]:
        current_cfg = load_config()
        results: list[CameraInfo] = []
        try:
            with get_connection(current_cfg.storage.db_path) as conn:
                cur = conn.cursor()
                cur.execute("SELECT camera_id, label, start_time, (start_time + duration) FROM cameras")
                for row in cur.fetchall():
                    cam_id, label, t_start, t_end = row
                    results.append(
                        CameraInfo(
                            camera_id=cam_id,
                            label=label or cam_id,
                            thumb_url=f"/api/results/cam_{cam_id}/thumb",
                            start=format_timestamp(t_start),
                            end=format_timestamp(t_end),
                        )
                    )
        except Exception as exc:
            logger.warning("Error fetching cameras: %s", exc)
        return results

    @app.post("/api/chat", response_model=ChatResponse)
    def chat_query(req: ChatRequest) -> ChatResponse:
        t0 = time.perf_counter()
        current_cfg = load_config()

        # 1. Parse natural language query
        t_parse_start = time.perf_counter()
        parsed = parse_query(req.message, use_llm=True, config=current_cfg)
        t_parse_ms = (time.perf_counter() - t_parse_start) * 1000.0

        with get_connection(current_cfg.storage.db_path) as conn:
            cur = conn.cursor()

            # Determine reference_now (max camera end time)
            cur.execute("SELECT MAX(start_time + duration) FROM cameras")
            max_row = cur.fetchone()
            ref_now = max_row[0] if max_row and max_row[0] else time.time()
            site_id = _select_site_id(conn, parsed)

            # 2. Time resolution
            time_range = resolve_time(parsed.time, reference_now=ref_now)

            # 3. Referent resolution
            ref_res = resolve_referents(parsed.location_phrases, parsed.camera_ids, site_id, conn)

            # If user explicitly specified camera ID(s) that do not exist in the site footage:
            if parsed.camera_ids and not ref_res.resolved_targets:
                total_ms = (time.perf_counter() - t0) * 1000.0
                searched_window = {
                    "cameras": parsed.camera_ids,
                    "t_start": time_range[0],
                    "t_end": time_range[1],
                }
                summary = build_answer_summary(
                    parsed.intent,
                    [],
                    searched_window,
                    object_phrase=parsed.object_phrase,
                )
                return ChatResponse(
                    type="no_match",
                    summary=summary,
                    searched=searched_window,
                    parsed=parsed,
                    latency_ms=LatencyBreakdown(parse=t_parse_ms, retrieve=0.0, verify=0.0, total=total_ms),
                    mode=req.mode,
                    verifier_used="none",
                    parser_used=get_last_parser_used(),
                )

            # Check for unresolved location referents (Clarify-Once Protocol)
            if ref_res.unresolved_phrases:
                unresolved_phrase = ref_res.unresolved_phrases[0]
                clarification_id = str(uuid.uuid4())

                # Store pending query for resumption
                cur.execute(
                    """
                    INSERT OR REPLACE INTO pending_queries
                    (session_id, message, parsed_json, unresolved_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        req.session_id,
                        req.message,
                        parsed.model_dump_json(),
                        json.dumps(ref_res.unresolved_phrases),
                        time.time(),
                    ),
                )
                conn.commit()

                # Fetch available cameras for user choice
                if site_id:
                    cur.execute("SELECT camera_id, label FROM cameras WHERE site_id = ?", (site_id,))
                else:
                    cur.execute("SELECT camera_id, label FROM cameras")
                cam_cards = [
                    CameraCard(camera_id=r[0], label=r[1], thumb_url=f"/api/results/cam_{r[0]}/thumb")
                    for r in cur.fetchall()
                ]

                # Fetch existing known referents
                if site_id:
                    cur.execute("SELECT referent_id, phrase_display, targets_json FROM referents WHERE site_id = ?", (site_id,))
                else:
                    cur.execute("SELECT referent_id, phrase_display, targets_json FROM referents")
                known = [
                    {"id": r[0], "display": r[1], "targets": json.loads(r[2])}
                    for r in cur.fetchall()
                ]

                total_ms = (time.perf_counter() - t0) * 1000.0
                return ChatResponse(
                    type="clarification",
                    clarification_id=clarification_id,
                    phrase=unresolved_phrase,
                    question=f"Where is '{unresolved_phrase}'? Select the corresponding camera feed.",
                    cameras=cam_cards,
                    known_referents=known,
                    suggestion=f"Map '{unresolved_phrase}' to a camera",
                    parsed=parsed,
                    latency_ms=LatencyBreakdown(parse=t_parse_ms, retrieve=0.0, verify=0.0, total=total_ms),
                    mode=req.mode,
                    verifier_used="none",
                    parser_used=get_last_parser_used(),
                )

            # 4. omni_min Search Backend
            t_ret_start = time.perf_counter()
            omni_results, omni_note = omni_min.search(req.message)
            t_ret_ms = (time.perf_counter() - t_ret_start) * 1000.0
            total_ms = (time.perf_counter() - t0) * 1000.0

            searched_cams = list({r["cam"] for r in omni_results}) if omni_results else ["All Cameras"]
            searched_window = {
                "cameras": searched_cams,
                "t_start": min((r["t_first"] for r in omni_results), default=0.0),
                "t_end": max((r["t_last"] for r in omni_results), default=0.0),
            }

            if not omni_results:
                return ChatResponse(
                    type="no_match",
                    summary=omni_note,
                    searched=searched_window,
                    parsed=parsed,
                    latency_ms=LatencyBreakdown(parse=t_parse_ms, retrieve=t_ret_ms, verify=0.0, total=total_ms),
                    mode=req.mode,
                    verifier_used="none",
                    parser_used="rules",
                )

            # Build result items with required omni_min fields
            result_items: list[ResultItem] = []
            for r in omni_results:
                kf_url = f"/api/keyframe?path={r['keyframe']}"
                det_c = round(float(r.get("det_conf", 0.0)), 4)
                match_s = round(float(r.get("match", 0.0)), 4)
                result_items.append(
                    ResultItem(
                        result_id=f"res_{r['cam']}_{r['tid']}",
                        camera_id=r["cam"],
                        camera_label=r["cam"],
                        t_start_iso=omni_min.fmt_t(r["t_first"]),
                        t_end_iso=omni_min.fmt_t(r["t_last"]),
                        t_peak_iso=omni_min.fmt_t(r.get("t_best", r["t_first"])),
                        score=match_s,
                        bbox=(
                            float(r.get("x1", 0)),
                            float(r.get("y1", 0)),
                            float(r.get("x2", 1)),
                            float(r.get("y2", 1)),
                        ),
                        tracklet_id=r["tid"],
                        thumb_url=kf_url,
                        clip_url="",
                        det_conf=det_c,
                        match_score=match_s,
                        unverified=bool(r.get("unverified", False)),
                        # Required omni_min fields
                        cam=r["cam"],
                        tid=r["tid"],
                        cls=r["cls"],
                        color=r["color"],
                        has_bag=bool(r.get("has_bag", False)),
                        match=match_s,
                        t_first=float(r["t_first"]),
                        t_last=float(r["t_last"]),
                        keyframe=kf_url,
                        matched=MatchedAttributes(
                            color=[r["color"]] if r.get("color") and r["color"] != "unknown" else [],
                            carries=[r["bag_type"]] if r.get("has_bag") and r.get("bag_type") else (["bag"] if r.get("has_bag") else []),
                            cls=r["cls"],
                        ),
                    )
                )

            # Log query
            cur.execute(
                """
                INSERT INTO query_log (ts, session_id, message, parsed_json, mode, latency_json, result_ids_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    req.session_id,
                    req.message,
                    parsed.model_dump_json(),
                    req.mode,
                    json.dumps({"parse": t_parse_ms, "ret": t_ret_ms, "ver": 0.0, "tot": total_ms}),
                    json.dumps([r.result_id for r in result_items]),
                ),
            )
            conn.commit()

            return ChatResponse(
                type="answer",
                summary=omni_note,
                results=result_items,
                parsed=parsed,
                latency_ms=LatencyBreakdown(parse=t_parse_ms, retrieve=t_ret_ms, verify=0.0, total=total_ms),
                mode=req.mode,
                verifier_used="none",
                parser_used="rules",
            )

    @app.post("/api/clarify", response_model=ChatResponse)
    def clarify_referent(req: ClarifyRequest) -> ChatResponse:
        current_cfg = load_config()
        with get_connection(current_cfg.storage.db_path) as conn:
            cur = conn.cursor()

            # Retrieve pending query
            cur.execute(
                "SELECT message, parsed_json, unresolved_json FROM pending_queries WHERE session_id = ?",
                (req.session_id,),
            )
            pending = cur.fetchone()
            if not pending:
                raise HTTPException(status_code=404, detail="No pending query found for this session.")

            message, parsed_json, unresolved_json = pending
            unresolved_list = json.loads(unresolved_json)
            phrase = unresolved_list[0] if unresolved_list else "unknown"
            parsed = ParsedQuery.model_validate_json(parsed_json)

            # Insert learned referent into Knowledge Base (clarify-once durably persisted)
            from omniwatch.kb.normalize import normalize_phrase
            p_norm = normalize_phrase(phrase)

            target_camera_ids = [target.camera_id for target in req.targets]
            if target_camera_ids:
                placeholders = ",".join("?" for _ in target_camera_ids)
                cur.execute(
                    f"SELECT DISTINCT site_id FROM cameras WHERE camera_id IN ({placeholders}) ORDER BY site_id",
                    target_camera_ids,
                )
                target_sites = [row[0] for row in cur.fetchall()]
                site_id = target_sites[0] if len(target_sites) == 1 else _select_site_id(conn, parsed)
            else:
                site_id = _select_site_id(conn, parsed)
            if site_id is None:
                site_id = "default_site"

            # Handle same_as alias or explicit targets
            if req.same_as is not None:
                cur.execute(
                    "SELECT targets_json FROM referents WHERE referent_id = ? AND site_id = ?",
                    (req.same_as, site_id),
                )
                ref_row = cur.fetchone()
                if ref_row:
                    targets_data = json.loads(ref_row[0])
                else:
                    targets_data = [{"camera_id": t.camera_id, "roi": list(t.roi) if t.roi else None} for t in req.targets]
            else:
                targets_data = [{"camera_id": t.camera_id, "roi": list(t.roi) if t.roi else None} for t in req.targets]

            cur.execute(
                """
                INSERT OR REPLACE INTO referents
                (site_id, phrase_norm, phrase_display, targets_json, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'clarification', ?, ?)
                """,
                (site_id, p_norm, phrase, json.dumps(targets_data), time.time(), time.time()),
            )

            # Log clarification
            cur.execute(
                """
                INSERT INTO clarification_log (site_id, phrase_norm, session_id, asked_at, resolved_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (site_id, p_norm, req.session_id, time.time() - 1.0, time.time()),
            )

            # Delete pending query
            cur.execute("DELETE FROM pending_queries WHERE session_id = ?", (req.session_id,))
            conn.commit()

        # Auto-resume the original query!
        return chat_query(ChatRequest(session_id=req.session_id, message=message, mode="fast"))

    @app.get("/api/kb")
    def list_knowledge_base() -> list[dict[str, Any]]:
        current_cfg = load_config()
        with get_connection(current_cfg.storage.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT referent_id, phrase_display, phrase_norm, targets_json, source, updated_at FROM referents")
            return [
                {
                    "referent_id": r[0],
                    "phrase_display": r[1],
                    "phrase_norm": r[2],
                    "targets": json.loads(r[3]),
                    "source": r[4],
                    "updated_at": format_timestamp(r[5]),
                }
                for r in cur.fetchall()
            ]

    @app.put("/api/kb/{referent_id}")
    def update_referent(referent_id: int, req: ClarifyRequest) -> dict[str, Any]:
        """Updates referent targets and appends to referent_history."""
        current_cfg = load_config()
        with get_connection(current_cfg.storage.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT targets_json FROM referents WHERE referent_id = ?", (referent_id,))
            old = cur.fetchone()
            if not old:
                raise HTTPException(status_code=404, detail="Referent not found.")
            old_targets = old[0]
            targets_data = [{"camera_id": t.camera_id, "roi": list(t.roi) if t.roi else None} for t in req.targets]
            new_targets = json.dumps(targets_data)
            now = time.time()
            cur.execute(
                "UPDATE referents SET targets_json = ?, updated_at = ? WHERE referent_id = ?",
                (new_targets, now, referent_id),
            )
            cur.execute(
                """
                INSERT INTO referent_history (referent_id, old_targets_json, new_targets_json, changed_at, actor)
                VALUES (?, ?, ?, ?, ?)
                """,
                (referent_id, old_targets, new_targets, now, "operator"),
            )
            conn.commit()
            return {"status": "updated", "referent_id": referent_id, "targets": targets_data}

    @app.delete("/api/kb/{referent_id}")
    def delete_referent(referent_id: int) -> dict[str, str]:
        current_cfg = load_config()
        with get_connection(current_cfg.storage.db_path) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM referents WHERE referent_id = ?", (referent_id,))
            conn.commit()
        return {"status": "deleted"}

    @app.get("/api/results/{tracklet_id}/thumb")
    def get_thumbnail(tracklet_id: str) -> FileResponse:
        current_cfg = load_config()
        cache_thumb_dir = Path(current_cfg.storage.evidence_cache_dir) / "thumbs"
        cache_thumb_dir.mkdir(parents=True, exist_ok=True)

        # Handle camera preview thumbnails
        if tracklet_id.startswith("cam_"):
            cam_id = tracklet_id.replace("cam_", "")
            thumb_path = cache_thumb_dir / f"cam_{cam_id}.jpg"
            if thumb_path.exists():
                return FileResponse(thumb_path, media_type="image/jpeg")

            # Extract first frame
            with get_connection(current_cfg.storage.db_path) as conn:
                cur = conn.cursor()
                cur.execute("SELECT video_path FROM cameras WHERE camera_id = ?", (cam_id,))
                row = cur.fetchone()
                if row:
                    from omniwatch.ingest.decode import VideoDecoder
                    decoder = VideoDecoder(row[0])
                    frame = decoder.read_frame_at_index(0)
                    if frame is not None:
                        import cv2
                        cv2.imwrite(str(thumb_path), frame)
                        return FileResponse(thumb_path, media_type="image/jpeg")
            raise HTTPException(status_code=404, detail="Camera thumbnail not found")

        # Handle tracklet observation thumbnail
        try:
            trk_id = int(tracklet_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid tracklet id") from None

        thumb_file = cache_thumb_dir / f"trk_{trk_id}.jpg"
        if thumb_file.exists():
            return FileResponse(thumb_file, media_type="image/jpeg")

        # Check if saved keyframe exists in omni_min db
        try:
            with omni_min.db() as con:
                row = con.execute("SELECT keyframe FROM obs WHERE tid = ?", (trk_id,)).fetchone()
                if row and row["keyframe"] and Path(row["keyframe"]).is_file():
                    return FileResponse(row["keyframe"], media_type="image/jpeg")
        except Exception:
            pass

        # Check pre-cached files
        thumb_file = cache_thumb_dir / f"trk_{trk_id}.jpg"
        if thumb_file.is_file():
            return FileResponse(str(thumb_file), media_type="image/jpeg")

        # Check keyframes directory by glob
        for f in Path("data/keyframes").glob(f"*/{trk_id}.jpg"):
            if f.is_file():
                return FileResponse(str(f), media_type="image/jpeg")

        raise HTTPException(status_code=404, detail="Keyframe thumbnail not found")

    @app.get("/api/keyframe")
    def get_keyframe(path: str) -> FileResponse:
        p = Path(path)
        if not p.is_file():
            if (Path("data") / path).is_file():
                p = Path("data") / path
            elif (Path("out") / path).is_file():
                p = Path("out") / path
            else:
                raise HTTPException(status_code=404, detail=f"Keyframe file not found: {path}")
        return FileResponse(str(p), media_type="image/jpeg")

    @app.get("/api/copy_sheet")
    def copy_sheet():
        import shutil
        art_dir = Path(r"C:\Users\ffgam\.gemini\antigravity-ide\brain\51bf8649-6906-49d5-982e-698ea3926a24")
        if Path("out/sheet.jpg").exists():
            shutil.copy2("out/sheet.jpg", str(art_dir / "sheet.jpg"))
        return {"status": "copied"}

    @app.get("/api/search")
    def search_endpoint(q: str):
        res, note = omni_min.search(q)
        if res:
            omni_min.make_sheet(res, "out/search.jpg")
        return {
            "summary": note,
            "results": [
                {
                    "cam": r["cam"],
                    "tid": r["tid"],
                    "cls": r["cls"],
                    "color": r["color"],
                    "has_bag": bool(r["has_bag"]),
                    "det_conf": r["det_conf"],
                    "match": r.get("match", 0.0),
                    "t_first": r["t_first"],
                    "t_last": r["t_last"],
                    "keyframe": f"/api/keyframe?path={r['keyframe']}",
                    "unverified": r.get("unverified", False),
                }
                for r in res
            ],
        }

    @app.post("/api/reindex")
    @app.get("/api/reindex")
    def api_reindex() -> dict[str, Any]:
        class IndexArgs:
            def __init__(self, cam: str, video: str, start: str = "2026-10-09 09:00:00"):
                self.cam = cam
                self.video = video
                self.start = start

        indexed_cams = []
        cams_to_index = [
            ("Yokohama", "data/public_cctv/yokohama_cctv.mp4", "2026-10-09 09:00:00"),
            ("Tours", "data/public_cctv/surveillance_tours.mp4", "2026-10-09 09:05:00"),
            ("cam1", "data/campus/cam1.mp4", "2026-10-09 09:10:00"),
            ("cam2", "data/campus/cam2.mp4", "2026-10-09 09:12:00"),
        ]
        for cam, vpath, tstart in cams_to_index:
            if Path(vpath).exists():
                logger.info("Indexing %s (%s)...", cam, vpath)
                omni_min.cmd_index(IndexArgs(cam, vpath, tstart))
                indexed_cams.append(cam)

        logger.info("Generating sanity contact sheet out/sheet.jpg...")
        omni_min.cmd_sheet(None)

        # Copy to artifact directory for display
        art_dir = Path(r"C:\Users\ffgam\.gemini\antigravity-ide\brain\51bf8649-6906-49d5-982e-698ea3926a24")
        if art_dir.exists() and Path("out/sheet.jpg").exists():
            import shutil
            shutil.copy2("out/sheet.jpg", str(art_dir / "sheet.jpg"))

        return {
            "status": "success",
            "indexed_cameras": indexed_cams,
            "sheet_path": "out/sheet.jpg",
            "sheet_url": "/api/keyframe?path=out/sheet.jpg",
        }

    @app.get("/api/test_live")
    def run_live_tests() -> dict[str, Any]:
        import io
        import sys
        import pytest
        capture = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = capture
        sys.stderr = capture
        try:
            ret = pytest.main(["tests/test_omni_live.py", "tests/test_omni_min.py", "-v"])
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        return {
            "exit_code": int(ret),
            "output": capture.getvalue(),
        }

    @app.get("/api/clip")
    async def get_clip_endpoint(cam: str, tid: int, force: bool = False) -> dict[str, Any]:
        import asyncio
        clip_path = await asyncio.to_thread(omni_min.ensure_clip, cam, tid, 2.0, force)
        if not clip_path or not Path(clip_path).exists():
            raise HTTPException(status_code=404, detail=f"Could not render clip for {cam} #{tid}")
        return {
            "status": "ready",
            "cam": cam,
            "tid": tid,
            "clip_url": f"/clips/{cam}_{tid}.mp4",
        }

    @app.get("/api/results/{tracklet_id}/clip")
    async def get_clip(tracklet_id: int) -> FileResponse:
        import asyncio
        cam = "Yokohama"
        try:
            with omni_min.db() as con:
                row = con.execute("SELECT cam FROM obs WHERE tid = ?", (tracklet_id,)).fetchone()
                if row:
                    cam = row[0]
        except Exception:
            pass
        clip_path = await asyncio.to_thread(omni_min.ensure_clip, cam, tracklet_id)
        if not clip_path or not Path(clip_path).exists():
            raise HTTPException(status_code=404, detail=f"Clip for #{tracklet_id} not found")
        return FileResponse(clip_path, media_type="video/mp4")

    # Serve rendered video clips statically
    clips_dir = Path("data/clips")
    clips_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/clips", StaticFiles(directory=str(clips_dir)), name="clips")

    # Serve static frontend web files
    web_dir = Path("web")
    if web_dir.exists():
        app.mount("/", StaticFiles(directory="web", html=True), name="static")

    return app


app = create_app()
