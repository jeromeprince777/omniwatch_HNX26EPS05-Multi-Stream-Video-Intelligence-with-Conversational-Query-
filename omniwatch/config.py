"""OmniWatch Configuration Module.

Loads typed settings from config.yaml with presets for fast, balanced, and accurate modes.
All retrieval weights and decision thresholds are explicitly tagged as placeholders
until tuned on the dev set via eval/run_eval.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

PresetName = Literal["fast", "balanced", "accurate"]


class WeightsConfig(BaseModel):
    w_emb: float = Field(default=0.65, description="Embedding cosine similarity weight")
    w_attr: float = Field(default=0.35, description="Attribute fusion match weight")
    w_time: float = Field(default=0.0, description="Temporal prior weight")


class ThresholdsConfig(BaseModel):
    exists: float = Field(default=0.45, description="Score threshold for existence intent")
    find: float = Field(default=0.45, description="Score threshold for find intent")
    when: float = Field(default=0.45, description="Score threshold for when intent")
    count: float = Field(default=0.50, description="Score threshold for count intent")
    path: float = Field(default=0.45, description="Score threshold for cross-camera path intent")


class PresetConfig(BaseModel):
    sample_fps: float = Field(default=2.0, description="Frame sampling rate per camera")
    motion_gate: bool = Field(default=True, description="Enable motion gating before detection")
    motion_threshold: float = Field(default=0.015, description="Pixel change fraction for motion")
    detector_model: str = Field(default="yolov8s-world.pt", description="Detector checkpoint")
    detector_conf: float = Field(default=0.25, description="Detector confidence threshold")
    encoder_name: str = Field(default="ViT-B-32", description="Vision-language model architecture")
    encoder_pretrained: str = Field(default="laion2b_s34b_b79k", description="Encoder weights tag")
    verify_enabled: bool = Field(default=True, description="Enable VLM verification pass")
    verify_top_k: int = Field(default=3, description="Candidate tracklets sent to VLM")
    weights: WeightsConfig = Field(default_factory=WeightsConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)


class TrackerConfig(BaseModel):
    name: str = Field(default="bytetrack")
    track_thresh: float = Field(default=0.45)
    track_buffer: int = Field(default=30)
    match_thresh: float = Field(default=0.80)


class StorageConfig(BaseModel):
    db_path: str = Field(default="data/omniwatch.db")
    vector_dir: str = Field(default="data/index")
    evidence_cache_dir: str = Field(default="data/cache")


class SystemConfig(BaseModel):
    device: str = Field(default="auto")
    time_tolerance_tau: float = Field(default=5.0)
    assume_synced: bool = Field(default=False)
    log_level: str = Field(default="INFO")


class QueryConfig(BaseModel):
    parser_mode: str = Field(default="rules_first", description="rules_first | llm_first | rules_only")
    ollama_url: str = Field(default="http://localhost:11434")
    parser_model: str = Field(default="llama3.2:1b")
    parser_timeout_s: float = Field(default=6.0)
    fallback_to_rules: bool = Field(default=True)
    dedup_gap_s: float = Field(default=2.0)
    dedup_sim_thresh: float = Field(default=0.90)
    retrieval_top_k: int = Field(default=50, description="Max candidates returned before threshold filtering")


class VLMConfig(BaseModel):
    preferred_verifier: str = Field(default="smolvlm2", description="smolvlm2 | openclip")
    fallback_verifier: str = Field(default="openclip", description="openclip | none")
    model_path: str = Field(default="models/SmolVLM2-2.2B-Instruct")
    device: str = Field(default="auto")
    max_crops: int = Field(default=3)
    enabled: bool = Field(default=True)
    alpha: float = Field(default=0.5, description="Weight for retrieval score")
    beta: float = Field(default=0.5, description="Weight for VLM P(Yes)")


class PrivacyConfig(BaseModel):
    blur: bool = Field(default=True, description="Enable face/person upper-third privacy blur on served evidence")
    method: str = Field(default="hybrid", description="cascade | box_head | hybrid")
    strength: int = Field(default=35, description="Gaussian blur kernel size")


class OmniWatchConfig(BaseModel):
    active_preset: PresetName = Field(default="balanced")
    presets: dict[PresetName, PresetConfig] = Field(default_factory=dict)
    detector_vocab: list[str] = Field(default_factory=list)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)

    @property
    def current(self) -> PresetConfig:
        """Returns the configuration for the active preset."""
        if self.active_preset in self.presets:
            return self.presets[self.active_preset]
        return PresetConfig()

    def resolve_device(self) -> str:
        """Determines the compute device ('cuda' or 'cpu')."""
        if self.system.device != "auto":
            return self.system.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


def load_config(
    config_path: str | Path = "config.yaml",
    preset: PresetName | None = None,
) -> OmniWatchConfig:
    """Load configuration from YAML file, with optional preset override."""
    path = Path(config_path)
    data: dict[str, Any] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    config = OmniWatchConfig(**data)
    if preset:
        config.active_preset = preset
    return config
