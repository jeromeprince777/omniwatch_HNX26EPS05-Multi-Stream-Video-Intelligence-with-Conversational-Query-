"""VLM and OpenCLIP Verification and Reranking Module.

Complies strictly with the OmniWatch Model Strategy:
- Primary retrieval: OpenCLIP ViT-B-32 with attribute fusion.
- Fast mode: No VLM verification, sub-second latency.
- Verified mode: Reranks top 3 candidates only using:
    1. Preferred: SmolVLM2-2.2B-Instruct (models/SmolVLM2-2.2B-Instruct) if memory & image tests pass.
    2. Fallback: OpenCLIP ViT-B-32 + explicit attribute fusion verifier.
- Backup verifier: Qwen2.5-VL-3B-Instruct remains offline/inactive by default (exceeds 6 GB VRAM).
- Never loads SmolVLM2 and Qwen simultaneously.
- Exposes active verifier and fallback status to /api/health and /api/chat.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from omniwatch.query.retrieve import CandidateResult

logger = logging.getLogger(__name__)

# Module-level observability state
LAST_VERIFIER_USED: str = "none"  # "none" | "smolvlm2" | "openclip_fallback"
VERIFIER_FALLBACK_ACTIVE: bool = False
VERIFIER_FALLBACK_REASON: str | None = None


def get_last_verifier_used() -> str:
    """Returns the verifier used during the most recent query."""
    return LAST_VERIFIER_USED


def get_verifier_health_status() -> dict[str, Any]:
    """Returns verifier health metadata for /api/health."""
    return {
        "preferred_verifier": "smolvlm2",
        "active_verifier": LAST_VERIFIER_USED,
        "fallback_active": VERIFIER_FALLBACK_ACTIVE,
        "fallback_reason": VERIFIER_FALLBACK_REASON,
    }


class OpenCLIPVerifier:
    """Fast, zero-extra-VRAM verifier using OpenCLIP ViT-B-32 and explicit attributes."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "auto",
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self._encoder: Any = None

    def _ensure_encoder(self) -> Any:
        if self._encoder is None:
            from omniwatch.embed.encoder import VLEncoder
            self._encoder = VLEncoder(
                model_name=self.model_name,
                pretrained=self.pretrained,
                device=self.device,
            )
        return self._encoder

    def verify_candidate(
        self,
        crop: Image.Image | np.ndarray | str | Path,
        text_hypothesis: str,
        matched_attrs: dict[str, Any] | None = None,
    ) -> float:
        """Evaluates crop match using OpenCLIP cosine similarity and attribute bonuses.

        Returns:
            Probability score P(Yes) in [0.0, 1.0].
        """
        try:
            encoder = self._ensure_encoder()

            if isinstance(crop, (str, Path)):
                img = Image.open(str(crop)).convert("RGB")
                arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            elif isinstance(crop, Image.Image):
                arr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
            elif isinstance(crop, np.ndarray):
                arr = crop
            else:
                return 0.50

            crop_emb = encoder.encode_images([arr])
            text_emb = encoder.encode_text(text_hypothesis)

            if crop_emb.shape[0] == 0 or text_emb.shape[0] == 0:
                return 0.50

            # Cosine similarity between normalized vectors
            sim = float(np.dot(crop_emb[0], text_emb[0]))
            # Calibrate open_clip cosine similarity (typically ~0.15 to ~0.35) into [0.1, 0.9]
            calibrated = float(np.clip((sim - 0.15) / 0.20, 0.10, 0.90))

            # Bonus for explicit attribute matches
            if matched_attrs:
                if matched_attrs.get("color"):
                    calibrated = min(1.0, calibrated + 0.05)
                if matched_attrs.get("carries"):
                    calibrated = min(1.0, calibrated + 0.05)

            return round(calibrated, 4)

        except Exception as exc:
            logger.warning("OpenCLIP verification error: %s", exc)
            return 0.50


class SmolVLM2Verifier:
    """Preferred VLM Verifier using SmolVLM2-2.2B-Instruct."""

    def __init__(
        self,
        model_path: str = "models/SmolVLM2-2.2B-Instruct",
        device: str = "auto",
        min_free_vram_gb: float = 4.5,
    ):
        self.model_path = model_path
        self.device = device
        self.min_free_vram_gb = min_free_vram_gb
        self._processor: Any = None
        self._model: Any = None
        self._loaded: bool = False
        self._failed: bool = False

    def is_available(self) -> bool:
        """Checks if the local model files exist."""
        return Path(self.model_path).exists()

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        if self._failed:
            return False

        path = Path(self.model_path)
        if not path.exists():
            logger.warning("SmolVLM2 model path '%s' not found. Fallback required.", path)
            self._failed = True
            return False

        try:
            import torch
            from transformers import SmolVLMForConditionalGeneration, SmolVLMProcessor

            target_device = "cuda" if (self.device == "auto" and torch.cuda.is_available()) or self.device == "cuda" else "cpu"

            # Check available GPU memory to avoid paging thrashing or OOM
            if target_device == "cuda":
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_gb = free_bytes / (1024**3)
                if free_gb < self.min_free_vram_gb:
                    logger.warning(
                        "SmolVLM2 requires >= %.1f GB free VRAM, but only %.2f GB is available of %.2f GB. "
                        "Skipping CUDA loading to prevent driver crash/severe paging thrashing.",
                        self.min_free_vram_gb, free_gb, total_bytes / (1024**3)
                    )
                    self._failed = True
                    return False

            logger.info("Loading SmolVLM2 from '%s' on %s...", path, target_device)
            self._processor = SmolVLMProcessor.from_pretrained(str(path), local_files_only=True)
            self._model = SmolVLMForConditionalGeneration.from_pretrained(
                str(path),
                torch_dtype=torch.float16 if target_device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            self._model.to(target_device)
            self._model.eval()
            self._loaded = True
            logger.info("SmolVLM2-2.2B-Instruct loaded successfully on %s.", target_device)
            return True

        except Exception as exc:
            logger.warning("Failed to load SmolVLM2 model (%s). Fallback required.", exc)
            self._failed = True
            return False

    def verify_candidate(
        self,
        crop: Image.Image | np.ndarray | str | Path,
        text_hypothesis: str,
    ) -> float:
        """Evaluates whether the crop matches the text hypothesis using SmolVLM2 next-token logits."""
        scores = self.verify_candidates_batch([crop], text_hypothesis)
        return scores[0] if scores else 0.50

    def verify_candidates_batch(
        self,
        crops: list[Image.Image | np.ndarray | str | Path],
        text_hypothesis: str,
    ) -> list[float]:
        """Evaluates multiple crops in a batch using next-token logits of 'Yes' vs 'No'."""
        if not crops:
            return []
        if not self._ensure_loaded():
            raise RuntimeError("SmolVLM2 is not loaded or failed memory requirements.")

        try:
            import torch

            pil_images = []
            for crop in crops:
                if isinstance(crop, (str, Path)):
                    img = Image.open(str(crop)).convert("RGB")
                elif isinstance(crop, np.ndarray):
                    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).convert("RGB")
                elif isinstance(crop, Image.Image):
                    img = crop.convert("RGB")
                else:
                    img = Image.new("RGB", (224, 224), color="gray")
                pil_images.append(img)

            # Format prompt asking Yes or No
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": f"Does this image show {text_hypothesis}? Answer with 'Yes' or 'No'.",
                        },
                    ],
                }
            ]
            prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
            prompts = [prompt] * len(pil_images)

            inputs = self._processor(
                text=prompts,
                images=pil_images,
                return_tensors="pt",
                padding=True,
            ).to(self._model.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                # Next-token logits at position -1: shape (batch_size, vocab_size)
                logits = outputs.logits[:, -1, :]

            # Token IDs for 'Yes'/'yes' and 'No'/'no'
            y_ids = [10539, 10407]
            n_ids = [5230, 4607]

            probs: list[float] = []
            for i in range(len(pil_images)):
                y_max = max(logits[i, y_ids].tolist())
                n_max = max(logits[i, n_ids].tolist())
                # Sigmoid of log-odds diff
                diff = y_max - n_max
                p_yes = 1.0 / (1.0 + np.exp(-diff))
                probs.append(float(np.clip(p_yes, 0.01, 0.99)))

            return probs

        except Exception as exc:
            logger.warning("SmolVLM2 candidate verification error: %s", exc)
            raise


class Qwen2_5_VLVerifier:
    """Backup VLM Verifier using Qwen2.5-VL-3B-Instruct."""

    def __init__(
        self,
        model_path: str = "models/Qwen2.5-VL-3B-Instruct",
        device: str = "auto",
        min_free_vram_gb: float = 6.0,
    ):
        self.model_path = model_path
        self.device = device
        self.min_free_vram_gb = min_free_vram_gb
        self._processor: Any = None
        self._model: Any = None
        self._loaded: bool = False
        self._failed: bool = False

    def is_available(self) -> bool:
        return Path(self.model_path).exists()

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        if self._failed:
            return False

        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            from omniwatch.ingest.pipeline import get_free_vram_gb

            free_vram = get_free_vram_gb()
            if free_vram < self.min_free_vram_gb:
                logger.warning(
                    "Insufficient VRAM for Qwen2.5-VL (need %.1f GB, have %.1f GB).",
                    self.min_free_vram_gb,
                    free_vram,
                )
                self._failed = True
                return False

            # Use qwen-vl compatible auto processor
            self._processor = AutoProcessor.from_pretrained(self.model_path)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map=self.device,
            )
            self._loaded = True
            return True
        except Exception as exc:
            logger.warning("Failed to load Qwen2.5-VL: %s", exc)
            self._failed = True
            return False

    def verify_candidate(
        self,
        crop: Image.Image | np.ndarray | str | Path,
        text_hypothesis: str,
    ) -> float:
        scores = self.verify_candidates_batch([crop], text_hypothesis)
        return scores[0] if scores else 0.50

    def verify_candidates_batch(
        self,
        crops: list[Image.Image | np.ndarray | str | Path],
        text_hypothesis: str,
    ) -> list[float]:
        """Evaluates multiple crops in a batch using next-token logits of 'Yes' vs 'No'."""
        if not crops:
            return []
        if not self._ensure_loaded():
            raise RuntimeError("Qwen2.5-VL is not loaded or failed memory requirements.")

        try:
            import torch

            pil_images = []
            for crop in crops:
                if isinstance(crop, (str, Path)):
                    img = Image.open(str(crop)).convert("RGB")
                elif isinstance(crop, np.ndarray):
                    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).convert("RGB")
                elif isinstance(crop, Image.Image):
                    img = crop.convert("RGB")
                else:
                    img = Image.new("RGB", (224, 224), color="gray")
                pil_images.append(img)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": f"Does this image show {text_hypothesis}? Answer with 'Yes' or 'No'.",
                        },
                    ],
                }
            ]
            prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
            prompts = [prompt] * len(pil_images)

            inputs = self._processor(
                text=prompts,
                images=pil_images,
                return_tensors="pt",
                padding=True,
            ).to(self._model.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits[:, -1, :]

            y_ids = self._processor.tokenizer.encode("Yes")
            n_ids = self._processor.tokenizer.encode("No")
            y_id = y_ids[-1] if y_ids else 0
            n_id = n_ids[-1] if n_ids else 0

            probs: list[float] = []
            for i in range(len(pil_images)):
                y_max = float(logits[i, y_id])
                n_max = float(logits[i, n_id])
                diff = y_max - n_max
                p_yes = 1.0 / (1.0 + np.exp(-diff))
                probs.append(float(np.clip(p_yes, 0.01, 0.99)))

            return probs

        except Exception as exc:
            logger.warning("Qwen2.5-VL candidate verification error: %s", exc)
            raise


class UnifiedReranker:
    """Manages VLM verification and OpenCLIP fallback for top candidates."""

    def __init__(
        self,
        model_path: str = "models/SmolVLM2-2.2B-Instruct",
        device: str = "auto",
        preferred_verifier: str = "smolvlm2",
        fallback_verifier: str = "openclip",
    ):
        self.preferred_verifier = preferred_verifier
        self.fallback_verifier = fallback_verifier
        self.device = device
        self.model_path = model_path
        self._smolvlm: SmolVLM2Verifier | None = None
        self._qwen: Qwen2_5_VLVerifier | None = None
        self._openclip: OpenCLIPVerifier | None = None

    def rerank(
        self,
        candidates: list[CandidateResult],
        text_hypothesis: str,
        alpha: float = 0.5,
        beta: float = 0.5,
        max_k: int = 3,
        crop_resolver: Any = None,
        conn: Any = None,
    ) -> list[CandidateResult]:
        """Reranks top-k candidates using the verified model strategy."""
        global LAST_VERIFIER_USED, VERIFIER_FALLBACK_ACTIVE, VERIFIER_FALLBACK_REASON

        if not candidates or max_k <= 0:
            LAST_VERIFIER_USED = "none"
            return candidates

        # Default crop resolver using SQLite conn if none provided
        if crop_resolver is None and conn is not None:
            def _default_resolver(cand: Any) -> Any:
                if cand.best_obs_id is None:
                    return None
                cur = conn.cursor()
                cur.execute(
                    "SELECT frame_idx, c.video_path FROM observations o "
                    "JOIN cameras c ON o.camera_id = c.camera_id "
                    "WHERE o.obs_id = ?",
                    (cand.best_obs_id,),
                )
                row = cur.fetchone()
                if row:
                    from omniwatch.ingest.decode import VideoDecoder
                    decoder = VideoDecoder(row[1])
                    frame = decoder.read_frame_at_index(row[0])
                    if frame is not None:
                        from omniwatch.evidence.thumbs import crop_detection
                        return crop_detection(frame, cand.bbox)
                return None
            crop_resolver = _default_resolver

        top_cands = candidates[:max_k]
        crops = [crop_resolver(c) if crop_resolver else None for c in top_cands]
        valid_indices = [i for i, cr in enumerate(crops) if cr is not None]

        if not valid_indices:
            LAST_VERIFIER_USED = "none"
            return candidates

        used_verifier = "none"

        if self.preferred_verifier == "smolvlm2":
            if self._smolvlm is None:
                self._smolvlm = SmolVLM2Verifier(model_path=self.model_path, device=self.device)

            if self._smolvlm.is_available():
                try:
                    valid_crops = [crops[i] for i in valid_indices]
                    p_yes_list = self._smolvlm.verify_candidates_batch(valid_crops, text_hypothesis)
                    for idx, p_yes in zip(valid_indices, p_yes_list, strict=False):
                        top_cands[idx].score = float(alpha * top_cands[idx].score + beta * p_yes)
                    used_verifier = "smolvlm2"
                    VERIFIER_FALLBACK_ACTIVE = False
                    VERIFIER_FALLBACK_REASON = None
                except Exception as exc:
                    logger.warning(
                        "SmolVLM2 verification failed (%s). Activating OpenCLIP fallback verifier.", exc
                    )
                    VERIFIER_FALLBACK_ACTIVE = True
                    VERIFIER_FALLBACK_REASON = f"SmolVLM2 unavailable/high VRAM load: {exc}"
                    used_verifier = "openclip_fallback"
            else:
                logger.warning(
                    "SmolVLM2 model files not available. Activating OpenCLIP fallback verifier."
                )
                VERIFIER_FALLBACK_ACTIVE = True
                VERIFIER_FALLBACK_REASON = "SmolVLM2 model files missing"
                used_verifier = "openclip_fallback"
                
        elif self.preferred_verifier == "qwen2.5-vl":
            if self._qwen is None:
                self._qwen = Qwen2_5_VLVerifier(model_path=self.model_path, device=self.device)

            if self._qwen.is_available():
                try:
                    valid_crops = [crops[i] for i in valid_indices]
                    p_yes_list = self._qwen.verify_candidates_batch(valid_crops, text_hypothesis)
                    for idx, p_yes in zip(valid_indices, p_yes_list, strict=False):
                        top_cands[idx].score = float(alpha * top_cands[idx].score + beta * p_yes)
                    used_verifier = "qwen2.5-vl"
                    VERIFIER_FALLBACK_ACTIVE = False
                    VERIFIER_FALLBACK_REASON = None
                except Exception as exc:
                    logger.warning(
                        "Qwen2.5-VL verification failed (%s). Activating OpenCLIP fallback verifier.", exc
                    )
                    VERIFIER_FALLBACK_ACTIVE = True
                    VERIFIER_FALLBACK_REASON = f"Qwen2.5-VL unavailable/high VRAM load: {exc}"
                    used_verifier = "openclip_fallback"
            else:
                logger.warning(
                    "Qwen2.5-VL model files not available. Activating OpenCLIP fallback verifier."
                )
                VERIFIER_FALLBACK_ACTIVE = True
                VERIFIER_FALLBACK_REASON = "Qwen2.5-VL model files missing"
                used_verifier = "openclip_fallback"

        if used_verifier in {"none", "openclip_fallback"}:
            if self._openclip is None:
                self._openclip = OpenCLIPVerifier(device=self.device)

            for idx in valid_indices:
                cand = top_cands[idx]
                p_yes = self._openclip.verify_candidate(
                    crops[idx], text_hypothesis, matched_attrs=cand.matched
                )
                cand.score = float(alpha * cand.score + beta * p_yes)
            used_verifier = "openclip_fallback"
            VERIFIER_FALLBACK_ACTIVE = True

        LAST_VERIFIER_USED = used_verifier
        return sorted(candidates, key=lambda c: c.score, reverse=True)


# Backwards compatibility alias
VLMVerifier = UnifiedReranker


def get_vlm_verifier(config: Any = None) -> UnifiedReranker:
    """Factory function returning configured UnifiedReranker instance."""
    if config is None:
        from omniwatch.config import load_config
        config = load_config()

    return UnifiedReranker(
        model_path=config.vlm.model_path,
        device=config.resolve_device(),
        preferred_verifier=config.vlm.preferred_verifier,
        fallback_verifier=config.vlm.fallback_verifier,
    )

