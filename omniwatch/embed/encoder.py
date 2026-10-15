"""Vision-Language Embedding Encoder Module.

Wraps open_clip models (e.g. ViT-B-32 trained on LAION-2B) for batched,
L2-normalized float32 embeddings across image crops and text prompts.
Supports prompt ensembling to improve multi-class retrieval robustness.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger("omniwatch.embed.encoder")

DEFAULT_PROMPT_TEMPLATES = [
    "a photo of {x}",
    "a surveillance camera image of {x}",
    "a cctv image of {x}",
    "{x}",
]


class VLEncoder:
    """Vision-Language Encoder supporting image crops and text embeddings."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "auto",
        fp16: bool = True,
    ):
        import open_clip

        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.fp16 = fp16 and ("cuda" in self.device)
        self.model_name = model_name
        self.pretrained = pretrained

        logger.info(
            "Loading OpenCLIP '%s' (pretrained='%s') on %s (fp16=%s)",
            model_name, pretrained, self.device, self.fp16
        )

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.model = model.eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

        if self.fp16:
            self.model = self.model.half()

        # Check embedding dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            if self.fp16:
                dummy = dummy.half()
            out = self.model.encode_image(dummy)
            self.dimension = int(out.shape[-1])

    def encode_images(self, crops: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Computes L2-normalized float32 embeddings for a list of BGR image crops.

        Args:
            crops: List of BGR numpy arrays.
            batch_size: Batch size for model inference.

        Returns:
            np.ndarray: Matrix of shape [N, D] with L2-normalized float32 vectors.
        """
        if not crops:
            return np.empty((0, self.dimension), dtype=np.float32)

        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i : i + batch_size]
            tensors = []

            for crop in batch_crops:
                if crop is None or crop.size == 0:
                    # Fallback blank image for invalid crop
                    crop = np.zeros((224, 224, 3), dtype=np.uint8)
                # Convert BGR OpenCV to RGB PIL Image
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                tensor = self.preprocess(pil_img)
                tensors.append(tensor)

            batch_tensor = torch.stack(tensors).to(self.device)
            if self.fp16:
                batch_tensor = batch_tensor.half()

            with torch.no_grad():
                features = self.model.encode_image(batch_tensor)
                features = F.normalize(features.float(), p=2, dim=-1)
                all_embeddings.append(features.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def encode_text(self, prompts: list[str], batch_size: int = 32) -> np.ndarray:
        """Computes L2-normalized float32 embeddings for text prompt strings.

        Args:
            prompts: List of text prompt strings.
            batch_size: Batch size for model inference.

        Returns:
            np.ndarray: Matrix of shape [N, D] with L2-normalized float32 vectors.
        """
        if not prompts:
            return np.empty((0, self.dimension), dtype=np.float32)

        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            tokens = self.tokenizer(batch_prompts).to(self.device)

            with torch.no_grad():
                features = self.model.encode_text(tokens)
                features = F.normalize(features.float(), p=2, dim=-1)
                all_embeddings.append(features.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def encode_prompt_ensemble(
        self,
        text: str,
        templates: list[str] | None = None,
    ) -> np.ndarray:
        """Computes prompt-ensembled L2-normalized embedding for a query phrase.

        Averages embeddings across templates ('a photo of {x}', etc.) and normalizes.

        Returns:
            np.ndarray: 1D vector of shape [D,] normalized to unit length.
        """
        tmpl_list = templates or DEFAULT_PROMPT_TEMPLATES
        formatted = [t.format(x=text) for t in tmpl_list]
        embeddings = self.encode_text(formatted)  # shape [K, D]

        # Average and re-normalize to unit length
        mean_vector = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_vector)
        if norm > 1e-6:
            mean_vector = mean_vector / norm
        return mean_vector.astype(np.float32)
