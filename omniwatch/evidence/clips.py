"""Video Clip Extraction and ffmpeg Caching Module.

Responsibility:
- Cuts event video clips [t_start - 2s, t_end + 2s] on demand using ffmpeg.
- Re-encodes to fast-start H.264 MP4 for smooth web browser seeking and playback.
- Caches cut clips deterministically by (camera_id, t_start, t_end).
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import imageio_ffmpeg

logger = logging.getLogger(__name__)


def cut_event_clip(
    video_path: str | Path,
    t_start_epoch: float,
    t_end_epoch: float,
    camera_start_epoch: float,
    cache_dir: str | Path = "data/cache/clips",
    pad_seconds: float = 2.0,
) -> Path:
    """Extracts a short playable H.264 MP4 clip for the localized event interval."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    clip_start_rel = max(0.0, (t_start_epoch - camera_start_epoch) - pad_seconds)
    clip_end_rel = (t_end_epoch - camera_start_epoch) + pad_seconds
    duration = max(1.0, clip_end_rel - clip_start_rel)

    out_name = f"clip_{Path(video_path).stem}_{int(t_start_epoch)}_{int(t_end_epoch)}.mp4"
    out_file = cache_path / out_name

    if out_file.exists() and out_file.stat().st_size > 0:
        return out_file

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss", f"{clip_start_rel:.2f}",
        "-i", str(video_path),
        "-t", f"{duration:.2f}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-movflags", "+faststart",
        "-an",
        str(out_file),
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        logger.warning("FFmpeg clip cut error (%s). Falling back to source video.", exc)
        return Path(video_path)

    return out_file


def cut_annotated_event_clip(
    video_path: str | Path,
    t_start_epoch: float,
    t_end_epoch: float,
    camera_start_epoch: float,
    tracklet_id: int,
    db_path: str | Path,
    cache_dir: str | Path = "data/cache/clips",
    pad_seconds: float = 2.0,
) -> Path:
    """Extracts a short playable MP4 clip with bounding boxes burned in."""
    import cv2
    from omniwatch.db import get_connection

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    out_name = f"annotated_clip_{Path(video_path).stem}_{int(t_start_epoch)}_{int(t_end_epoch)}_{tracklet_id}.mp4"
    out_file = cache_path / out_name

    if out_file.exists() and out_file.stat().st_size > 0:
        return out_file

    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT frame_idx, x1, y1, x2, y2 FROM observations WHERE tracklet_id = ?",
            (tracklet_id,)
        ).fetchall()
        
    obs_map = {r["frame_idx"]: (r["x1"], r["y1"], r["x2"], r["y2"]) for r in rows}
    
    if not obs_map:
        return cut_event_clip(video_path, t_start_epoch, t_end_epoch, camera_start_epoch, cache_dir, pad_seconds)
        
    min_frame = min(obs_map.keys())
    max_frame = max(obs_map.keys())
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return cut_event_clip(video_path, t_start_epoch, t_end_epoch, camera_start_epoch, cache_dir, pad_seconds)
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    pad_frames = int(pad_seconds * fps)
    start_f = max(0, min_frame - pad_frames)
    end_f = max_frame + pad_frames
    
    temp_mp4 = cache_path / f"temp_{out_name}"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_mp4), fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    current_f = start_f
    
    while current_f <= end_f:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_f in obs_map:
            x1, y1, x2, y2 = obs_map[current_f]
            px1, py1 = int(x1 * width), int(y1 * height)
            px2, py2 = int(x2 * width), int(y2 * height)
            
            # Tactical bounding box
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
            
            # Corner reticles for high accuracy look
            l = 15
            t = 3
            # Top-left
            cv2.line(frame, (px1, py1), (px1 + l, py1), (0, 255, 0), t)
            cv2.line(frame, (px1, py1), (px1, py1 + l), (0, 255, 0), t)
            # Top-right
            cv2.line(frame, (px2, py1), (px2 - l, py1), (0, 255, 0), t)
            cv2.line(frame, (px2, py1), (px2, py1 + l), (0, 255, 0), t)
            # Bottom-left
            cv2.line(frame, (px1, py2), (px1 + l, py2), (0, 255, 0), t)
            cv2.line(frame, (px1, py2), (px1, py2 - l), (0, 255, 0), t)
            # Bottom-right
            cv2.line(frame, (px2, py2), (px2 - l, py2), (0, 255, 0), t)
            cv2.line(frame, (px2, py2), (px2, py2 - l), (0, 255, 0), t)
            
        out.write(frame)
        current_f += 1
        
    cap.release()
    out.release()
    
    # Transcode to faststart x264 using ffmpeg for web playback
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y",
        "-i", str(temp_mp4),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-movflags", "+faststart",
        "-an",
        str(out_file),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        logger.warning("FFmpeg transcode error: %s", exc)
        if temp_mp4.exists():
            import shutil
            shutil.move(str(temp_mp4), str(out_file))
    finally:
        if temp_mp4.exists():
            temp_mp4.unlink()
            
    return out_file
