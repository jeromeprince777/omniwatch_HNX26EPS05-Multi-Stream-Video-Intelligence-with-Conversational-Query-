"""Pydantic Request and Response Schemas for the OmniWatch API.

Strictly adheres to docs/01_ARCHITECTURE.md section 6.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from omniwatch.query.parser import ParsedQuery


class ChatRequest(BaseModel):
    session_id: str
    message: str
    mode: Literal["fast", "verified"] = "verified"


class TargetRegion(BaseModel):
    camera_id: str
    roi: tuple[float, float, float, float] | None = None  # normalized [x1, y1, x2, y2]


class ClarifyRequest(BaseModel):
    session_id: str
    clarification_id: str
    targets: list[TargetRegion] = Field(default_factory=list)
    same_as: int | None = None  # referent_id if alias


class MatchedAttributes(BaseModel):
    color: list[str] = Field(default_factory=list)
    carries: list[str] = Field(default_factory=list)
    cls: str | None = None


class ResultItem(BaseModel):
    result_id: str
    camera_id: str
    camera_label: str | None = None
    t_start_iso: str
    t_end_iso: str
    t_peak_iso: str = ""
    score: float
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    tracklet_id: int
    matched: MatchedAttributes = Field(default_factory=MatchedAttributes)
    thumb_url: str
    clip_url: str = ""
    det_conf: float = 0.0
    match_score: float = 0.0
    unverified: bool = False
    # omni_min fields
    cam: str | None = None
    tid: int | None = None
    cls: str | None = None
    color: str | None = None
    has_bag: bool = False
    match: float = 0.0
    t_first: float = 0.0
    t_last: float = 0.0
    keyframe: str | None = None


class LatencyBreakdown(BaseModel):
    parse: float
    retrieve: float
    verify: float = 0.0
    total: float


class CameraCard(BaseModel):
    camera_id: str
    label: str | None = None
    thumb_url: str


class ChatResponse(BaseModel):
    type: Literal["answer", "clarification", "no_match", "error"]
    summary: str | None = None
    results: list[ResultItem] = Field(default_factory=list)
    parsed: ParsedQuery | None = None
    latency_ms: LatencyBreakdown | None = None
    mode: Literal["fast", "verified"] | None = None
    verifier_used: str | None = None  # "none" | "smolvlm2" | "openclip_fallback"
    parser_used: str | None = None  # "rules" | "ollama_fallback"
    # Clarification-specific fields
    clarification_id: str | None = None
    phrase: str | None = None
    question: str | None = None
    cameras: list[CameraCard] = Field(default_factory=list)
    known_referents: list[dict[str, Any]] = Field(default_factory=list)
    suggestion: str | None = None
    # No-match specific fields
    searched: dict[str, Any] | None = None
    # Error message
    message: str | None = None


class CameraInfo(BaseModel):
    camera_id: str
    label: str | None = None
    thumb_url: str
    start: str
    end: str


class HealthResponse(BaseModel):
    models: dict[str, str]
    fallbacks_active: list[str]
    index_counts: dict[str, int]
    device: str
    verifier_status: str | None = None
    parser_status: str | None = None
    offline_ready: bool = True
    privacy_blur_enabled: bool = True

