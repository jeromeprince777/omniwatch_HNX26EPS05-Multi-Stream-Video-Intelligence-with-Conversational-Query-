"""Manifest Parser and Validator for Multi-Camera Ingestion.

Validates manifest.yaml structure, resolves timezones into UTC epoch seconds,
and supports multi-part video segments per camera ordered by start_time.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


def parse_timestamp_to_epoch(ts: str | float | int | datetime | None) -> float | None:
    """Converts ISO 8601 string, datetime, or numeric seconds to UTC epoch float."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            # Default naive datetimes to UTC
            ts = ts.replace(tzinfo=UTC)
        return ts.timestamp()
    if isinstance(ts, str):
        # Handle ISO format
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.timestamp()
    raise ValueError(f"Unsupported timestamp format: {ts!r}")


class VideoSegment(BaseModel):
    video_path: str
    start_time: float | None = None  # epoch seconds UTC


class CameraConfig(BaseModel):
    camera_id: str
    label: str | None = None
    video: str | list[str] | None = None
    start_time: str | float | int | datetime | None = None
    segments: list[VideoSegment] = Field(default_factory=list)


class ManifestRaw(BaseModel):
    site_id: str
    reference_now: str | float | int | datetime | None = None
    cameras: list[CameraConfig] = Field(default_factory=list)

    @field_validator("site_id")
    @classmethod
    def validate_site_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("site_id cannot be empty")
        return v.strip()


class ResolvedCamera:
    """Normalized camera entity containing ordered video segments."""

    def __init__(self, camera_id: str, label: str | None = None):
        self.camera_id = camera_id
        self.label = label
        self.segments: list[VideoSegment] = []

    @property
    def start_time(self) -> float:
        if not self.segments:
            return 0.0
        return self.segments[0].start_time or 0.0

    @property
    def primary_video_path(self) -> str:
        if not self.segments:
            return ""
        return self.segments[0].video_path


class IngestManifest:
    """Fully validated and normalized multi-camera site manifest."""

    def __init__(
        self,
        site_id: str,
        cameras: dict[str, ResolvedCamera],
        reference_now: float | None = None,
    ):
        self.site_id = site_id
        self.cameras = cameras
        self.reference_now = reference_now


def load_manifest(
    manifest_path: str | Path,
    assume_synced: bool = False,
    default_start_epoch: float | None = None,
) -> IngestManifest:
    """Loads and validates a multi-camera manifest.

    Raises:
        FileNotFoundError: If manifest_path does not exist.
        ValueError: If start_time is missing without assume_synced.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}

    raw_manifest = ManifestRaw(**raw_data)
    ref_now_epoch = parse_timestamp_to_epoch(raw_manifest.reference_now)

    # First pass: if assume_synced is active, find the earliest defined start_time to use as fallback t0
    fallback_t0 = default_start_epoch
    if fallback_t0 is None:
        for cam in raw_manifest.cameras:
            if cam.start_time is not None:
                t = parse_timestamp_to_epoch(cam.start_time)
                if fallback_t0 is None or (t is not None and t < fallback_t0):
                    fallback_t0 = t
        if fallback_t0 is None and assume_synced:
            # If no camera provided a start_time at all, fallback to reference_now or 0.0
            fallback_t0 = ref_now_epoch if ref_now_epoch is not None else 0.0

    resolved_cameras: dict[str, ResolvedCamera] = {}

    for cam in raw_manifest.cameras:
        cam_id = cam.camera_id
        if cam_id not in resolved_cameras:
            resolved_cameras[cam_id] = ResolvedCamera(camera_id=cam_id, label=cam.label)
        elif cam.label and not resolved_cameras[cam_id].label:
            resolved_cameras[cam_id].label = cam.label

        cam_start = parse_timestamp_to_epoch(cam.start_time)
        if cam_start is None:
            if not assume_synced:
                raise ValueError(
                    f"Missing required 'start_time' for camera '{cam_id}'. "
                    f"Provide start_time in manifest or run ingestion with --assume-synced."
                )
            cam_start = fallback_t0

        # Handle video string, list of video strings, or explicit segments
        if cam.segments:
            for seg in cam.segments:
                seg_time = seg.start_time or cam_start
                resolved_cameras[cam_id].segments.append(
                    VideoSegment(video_path=seg.video_path, start_time=seg_time)
                )
        elif isinstance(cam.video, list):
            for v_path in cam.video:
                resolved_cameras[cam_id].segments.append(
                    VideoSegment(video_path=v_path, start_time=cam_start)
                )
        elif isinstance(cam.video, str):
            resolved_cameras[cam_id].segments.append(
                VideoSegment(video_path=cam.video, start_time=cam_start)
            )
        else:
            raise ValueError(f"Camera '{cam_id}' must specify at least one video file.")

    # Sort each camera's segments chronologically by start_time
    for resolved_cam in resolved_cameras.values():
        resolved_cam.segments.sort(key=lambda s: s.start_time or 0.0)

    return IngestManifest(
        site_id=raw_manifest.site_id,
        cameras=resolved_cameras,
        reference_now=ref_now_epoch,
    )
