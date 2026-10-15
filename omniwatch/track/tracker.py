"""Per-Camera ByteTrack-Class Multi-Object Tracking.

Ensures zero state leakage across cameras.
Uses two-stage association (high-confidence IoU matching followed by
low-confidence recovery) with config-driven sample_fps adaptations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from omniwatch.detect.detector import Box

logger = logging.getLogger("omniwatch.track.tracker")


def compute_iou_matrix(
    boxes_a: list[tuple[float, float, float, float]],
    boxes_b: list[tuple[float, float, float, float]],
) -> np.ndarray:
    """Computes IoU cost matrix (1 - IoU) between two sets of boxes."""
    if not boxes_a or not boxes_b:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = np.array(boxes_a, dtype=np.float32)
    b = np.array(boxes_b, dtype=np.float32)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection

    iou = np.where(union > 0, intersection / np.maximum(union, 1e-6), 0.0)
    return 1.0 - iou  # Cost matrix: lower is better match

def compute_appearance_matrix(
    embs_a: list[np.ndarray | None],
    embs_b: list[np.ndarray | None],
) -> np.ndarray:
    """Computes cosine distance matrix between two sets of embeddings."""
    cost = np.ones((len(embs_a), len(embs_b)), dtype=np.float32)
    if not embs_a or not embs_b:
        return cost
        
    for i, ea in enumerate(embs_a):
        for j, eb in enumerate(embs_b):
            if ea is not None and eb is not None:
                n1 = np.linalg.norm(ea)
                n2 = np.linalg.norm(eb)
                if n1 > 1e-6 and n2 > 1e-6:
                    sim = float(np.dot(ea / n1, (eb / n2).T))
                    cost[i, j] = 1.0 - max(0.0, sim)
    return cost


@dataclass
class TrackedDetection:
    frame_idx: int
    timestamp: float
    box: Box
    primary_color: str = "gray"
    secondary_color: str = "gray"
    carries: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = None
    obs_id: int | None = None


@dataclass
class SingleTrack:
    track_id: int
    camera_id: str
    cls: str
    last_bbox: tuple[float, float, float, float]
    history: list[TrackedDetection] = field(default_factory=list)
    last_seen_frame: int = 0
    last_seen_ts: float = 0.0
    state: str = "active"  # "active", "lost", "finished"


class CameraTracker:
    """ByteTrack tracker strictly scoped to a single camera stream."""

    def __init__(
        self,
        camera_id: str,
        track_thresh: float = 0.45,
        match_thresh: float = 0.80,
        max_gap_s: float = 3.0,
        sample_fps: float = 2.0,
    ):
        self.camera_id = camera_id
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_gap_s = max_gap_s
        self.sample_fps = sample_fps

        # Frame buffer before closing lost tracks
        self.max_lost_frames = max(1, int(round(max_gap_s * sample_fps)))

        self._next_track_id: int = 1
        self.active_tracks: list[SingleTrack] = []
        self.lost_tracks: list[SingleTrack] = []
        self.finished_tracks: list[SingleTrack] = []

    def update(
        self,
        detections: list[Box],
        timestamp: float,
        frame_idx: int,
    ) -> list[SingleTrack]:
        """Processes detections for a single frame, advancing track state."""
        # Split detections into high and low confidence pools
        high_det: list[Box] = []
        low_det: list[Box] = []

        for d in detections:
            if d.conf >= self.track_thresh:
                high_det.append(d)
            elif d.conf >= 0.10:
                low_det.append(d)

        # 1. First association: match active tracks with high-confidence detections
        track_boxes = [t.last_bbox for t in self.active_tracks]
        high_boxes = [d.bbox for d in high_det]
        
        # Extract embeddings for ReID
        track_embs = []
        for t in self.active_tracks:
            emb = None
            for det in reversed(t.history):
                if getattr(det.box, "embedding", None) is not None:
                    emb = det.box.embedding
                    break
            track_embs.append(emb)
        high_embs = [getattr(d, "embedding", None) for d in high_det]

        unmatched_tracks: list[SingleTrack] = []
        unmatched_high_dets: list[Box] = list(high_det)

        if track_boxes and high_boxes:
            iou_cost = compute_iou_matrix(track_boxes, high_boxes)
            app_cost = compute_appearance_matrix(track_embs, high_embs)
            
            # Industrial-grade StrongSORT-like fusion: 50% Spatial (IoU), 50% Appearance (ReID)
            cost_matrix = 0.5 * iou_cost + 0.5 * app_cost
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_track_idx = set()
            matched_det_idx = set()

            for r, c in zip(row_ind, col_ind, strict=False):
                # Combined cost threshold check: must be <= 0.8 (e.g. good IoU OR good Appearance)
                if cost_matrix[r, c] <= 0.8:
                    track = self.active_tracks[r]
                    det = high_det[c]
                    track.last_bbox = det.bbox
                    track.last_seen_frame = frame_idx
                    track.last_seen_ts = timestamp
                    track.history.append(
                        TrackedDetection(frame_idx=frame_idx, timestamp=timestamp, box=det)
                    )
                    matched_track_idx.add(r)
                    matched_det_idx.add(c)

            unmatched_tracks = [t for i, t in enumerate(self.active_tracks) if i not in matched_track_idx]
            unmatched_high_dets = [d for i, d in enumerate(high_det) if i not in matched_det_idx]
        else:
            unmatched_tracks = list(self.active_tracks)

        # 2. Second association: match remaining active & lost tracks with low-confidence detections
        remaining_tracks = unmatched_tracks + self.lost_tracks
        low_boxes = [d.bbox for d in low_det]
        
        rem_embs = []
        for t in remaining_tracks:
            emb = None
            for det in reversed(t.history):
                if getattr(det.box, "embedding", None) is not None:
                    emb = det.box.embedding
                    break
            rem_embs.append(emb)
        low_embs = [getattr(d, "embedding", None) for d in low_det]

        unmatched_after_low: list[SingleTrack] = []

        if remaining_tracks and low_boxes:
            iou_cost_low = compute_iou_matrix([t.last_bbox for t in remaining_tracks], low_boxes)
            app_cost_low = compute_appearance_matrix(rem_embs, low_embs)
            
            cost_matrix_low = 0.5 * iou_cost_low + 0.5 * app_cost_low
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix_low)

            matched_track_idx = set()
            for r, c in zip(row_ind, col_ind, strict=False):
                if cost_matrix_low[r, c] <= 0.85:  # Slightly looser threshold for low conf recovery
                    track = remaining_tracks[r]
                    det = low_det[c]
                    track.last_bbox = det.bbox
                    track.last_seen_frame = frame_idx
                    track.last_seen_ts = timestamp
                    track.history.append(
                        TrackedDetection(frame_idx=frame_idx, timestamp=timestamp, box=det)
                    )
                    track.state = "active"
                    matched_track_idx.add(r)

            unmatched_after_low = [t for i, t in enumerate(remaining_tracks) if i not in matched_track_idx]
        else:
            unmatched_after_low = remaining_tracks

        # 3. Initialize new tracks from unmatched high-confidence detections
        new_active: list[SingleTrack] = [t for t in self.active_tracks if t.last_seen_frame == frame_idx]

        for det in unmatched_high_dets:
            new_track = SingleTrack(
                track_id=self._next_track_id,
                camera_id=self.camera_id,
                cls=det.cls,
                last_bbox=det.bbox,
                last_seen_frame=frame_idx,
                last_seen_ts=timestamp,
                state="active",
            )
            new_track.history.append(
                TrackedDetection(frame_idx=frame_idx, timestamp=timestamp, box=det)
            )
            self._next_track_id += 1
            new_active.append(new_track)

        # 4. Update lost tracks and finish expired tracks
        new_lost: list[SingleTrack] = []
        for track in unmatched_after_low:
            age_frames = frame_idx - track.last_seen_frame
            age_seconds = timestamp - track.last_seen_ts

            if age_frames > self.max_lost_frames or age_seconds > self.max_gap_s:
                track.state = "finished"
                self.finished_tracks.append(track)
            else:
                track.state = "lost"
                new_lost.append(track)

        self.active_tracks = new_active
        self.lost_tracks = new_lost

        return list(self.active_tracks)

    def finish_all(self) -> list[SingleTrack]:
        """Flushes all remaining active and lost tracks into finished_tracks."""
        for t in self.active_tracks + self.lost_tracks:
            t.state = "finished"
            self.finished_tracks.append(t)
        self.active_tracks.clear()
        self.lost_tracks.clear()
        return self.finished_tracks
