"""Tracklet Assembly and Aggregation Module.

Computes majority-vote object class, quality-weighted aggregate embeddings,
top-k representative observation selection, and attribute histograms.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from omniwatch.track.tracker import SingleTrack


@dataclass
class TrackletEntity:
    tracklet_id: int | None
    site_id: str
    camera_id: str
    local_track_id: int
    cls: str
    t_start: float
    t_end: float
    n_obs: int
    best_obs_id: int | None = None
    topk_obs_ids: list[int] = field(default_factory=list)
    attrs: dict[str, str | list[str] | dict[str, int]] = field(default_factory=dict)
    aggregate_embedding: np.ndarray | None = None
    emb_row: int | None = None
    global_id: int | None = None

    @property
    def attrs_json(self) -> str:
        return json.dumps(self.attrs)

    @property
    def topk_obs_json(self) -> str:
        return json.dumps(self.topk_obs_ids)


def assemble_tracklet(
    single_track: SingleTrack,
    site_id: str,
    top_k: int = 3,
) -> TrackletEntity:
    """Assembles a completed SingleTrack into a TrackletEntity with aggregated attributes and embedding."""
    history = single_track.history
    if not history:
        raise ValueError(f"Cannot assemble empty tracklet for track {single_track.track_id}")

    # 1. Class by majority vote
    cls_counts = Counter(d.box.cls for d in history)
    dominant_cls = cls_counts.most_common(1)[0][0]

    # 2. Start and end timestamps
    t_start = min(d.timestamp for d in history)
    t_end = max(d.timestamp for d in history)

    # 3. Sort observations by crop quality
    sorted_obs = sorted(history, key=lambda d: d.box.quality, reverse=True)
    best_obs = sorted_obs[0]
    topk_obs = sorted_obs[:top_k]

    # 4. Attribute summary
    primary_colors = [d.primary_color for d in history if d.primary_color != "unknown"]
    color_hist = Counter(primary_colors)
    dominant_color = color_hist.most_common(1)[0][0] if primary_colors else "gray"

    secondary_colors = [d.secondary_color for d in history if d.secondary_color not in ("unknown", dominant_color)]
    dominant_secondary = Counter(secondary_colors).most_common(1)[0][0] if secondary_colors else dominant_color

    # Carries frequency: item carried if observed in >= 20% of frames
    carries_counter: Counter[str] = Counter()
    for d in history:
        for item in d.carries:
            carries_counter[item] += 1

    min_carries_count = max(1, int(len(history) * 0.20))
    confirmed_carries = sorted([
        item for item, count in carries_counter.items() if count >= min_carries_count
    ])

    attrs_summary: dict[str, str | list[str] | dict[str, int]] = {
        "color_primary": dominant_color,
        "color_secondary": dominant_secondary,
        "carries": confirmed_carries,
        "color_histogram": dict(color_hist),
    }

    # 5. Quality-weighted aggregate embedding
    agg_emb: np.ndarray | None = None
    embs_with_weights = []

    for d in history:
        if d.embedding is not None:
            # Quality weight with minimum floor to prevent zeroing
            w = max(d.box.quality, 0.05)
            embs_with_weights.append((d.embedding, w))

    if embs_with_weights:
        weighted_sum = np.zeros_like(embs_with_weights[0][0], dtype=np.float32)
        total_w = 0.0
        for emb, w in embs_with_weights:
            weighted_sum += emb * w
            total_w += w

        if total_w > 0:
            agg_vector = weighted_sum / total_w
            norm = np.linalg.norm(agg_vector)
            if norm > 1e-6:
                agg_emb = (agg_vector / norm).astype(np.float32)

    return TrackletEntity(
        tracklet_id=None,
        site_id=site_id,
        camera_id=single_track.camera_id,
        local_track_id=single_track.track_id,
        cls=dominant_cls,
        t_start=t_start,
        t_end=t_end,
        n_obs=len(history),
        best_obs_id=best_obs.obs_id,
        topk_obs_ids=[d.obs_id for d in topk_obs if d.obs_id is not None],
        attrs=attrs_summary,
        aggregate_embedding=agg_emb,
    )
