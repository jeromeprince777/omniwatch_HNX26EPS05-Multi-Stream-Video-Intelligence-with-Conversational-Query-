"""Object Detection Interface and Adapters for Open-Vocabulary and Closed-Set YOLO.

Implements batched inference, FP16 GPU acceleration, normalized [0, 1] bounding boxes,
and crop quality scoring combining detector confidence, sharpness, and relative area.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from omniwatch.config import OmniWatchConfig
from omniwatch.embed.attrs import compute_quality_score

logger = logging.getLogger("omniwatch.detect.detector")


@dataclass
class Box:
    x1: float  # normalized [0, 1]
    y1: float  # normalized [0, 1]
    x2: float  # normalized [0, 1]
    y2: float  # normalized [0, 1]
    cls: str
    conf: float
    quality: float = 0.0
    crop: np.ndarray | None = None
    px1: int = 0
    py1: int = 0
    px2: int = 0
    py2: int = 0
    frame_w: int = 0
    frame_h: int = 0
    carrying_bag: bool = False
    bag_type: str | None = None
    bag_conf: float = 0.0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def pixel_bbox(self) -> tuple[int, int, int, int]:
        return (self.px1, self.py1, self.px2, self.py2)


class BaseDetector(ABC):
    """Abstract Base Detector interface."""

    @abstractmethod
    def detect(self, frames: list[np.ndarray] | np.ndarray) -> list[list[Box]]:
        """Performs batched detection on a list of BGR image arrays.

        Returns:
            list[list[Box]]: Detected boxes per input frame.
        """
        pass


COCO_TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
}


class YOLOWorldDetector(BaseDetector):
    """Open-Vocabulary YOLO-World detector adapter."""

    def __init__(
        self,
        model_name: str | Path = "models/yolov8s-worldv2.pt",
        vocabulary: list[str] | None = None,
        conf_thresh: float = 0.40,
        min_box_height: int = 64,
        min_box_area_ratio: float = 0.002,
        crop_pad_ratio: float = 0.25,
        min_crop_short_side: int = 64,
        device: str = "auto",
        fp16: bool = True,
    ):
        from ultralytics import YOLO

        self.conf_thresh = conf_thresh
        self.min_box_height = min_box_height
        self.min_box_area_ratio = min_box_area_ratio
        self.crop_pad_ratio = crop_pad_ratio
        self.min_crop_short_side = min_crop_short_side
        self.vocabulary = vocabulary or [
            "person", "car", "truck", "bus", "motorcycle", "bicycle",
            "backpack", "handbag", "suitcase", "umbrella", "box"
        ]

        # Resolve device
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.fp16 = fp16 and ("cuda" in self.device)

        logger.info(
            "Loading YOLO-World detector '%s' on %s (fp16=%s, conf=%.2f)",
            model_name, self.device, self.fp16, self.conf_thresh
        )
        self.model = YOLO(str(model_name))
        self.model.set_classes(self.vocabulary)
        self.model_name = str(model_name)

    def detect(self, frames: list[np.ndarray] | np.ndarray) -> list[list[Box]]:
        if isinstance(frames, np.ndarray) and frames.ndim == 3:
            input_frames = [frames]
        else:
            input_frames = list(frames)

        if not input_frames:
            return []

        # Batched prediction with imgsz=640 and conf threshold
        results = self.model.predict(
            source=input_frames,
            conf=self.conf_thresh,
            imgsz=640,
            device=self.device,
            half=self.fp16,
            verbose=False,
        )

        all_frame_boxes: list[list[Box]] = []

        for frame_img, res in zip(input_frames, results, strict=False):
            h, w = frame_img.shape[:2]
            frame_area = w * h
            frame_boxes: list[Box] = []

            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy()
                names = res.names

                for box_coords, conf, cls_id in zip(xyxy, confs, clss, strict=False):
                    bx1, by1, bx2, by2 = box_coords
                    px1 = max(0, int(round(bx1)))
                    py1 = max(0, int(round(by1)))
                    px2 = min(w, int(round(bx2)))
                    py2 = min(h, int(round(by2)))

                    box_h = py2 - py1
                    box_w = px2 - px1
                    box_area = box_w * box_h

                    # Drop boxes under minimum height (64px) or area (<0.2% frame)
                    if box_h < self.min_box_height or box_area < (self.min_box_area_ratio * frame_area):
                        continue

                    # Normalized coordinates [0, 1]
                    norm_x1 = max(0.0, min(1.0, float(px1 / w)))
                    norm_y1 = max(0.0, min(1.0, float(py1 / h)))
                    norm_x2 = max(0.0, min(1.0, float(px2 / w)))
                    norm_y2 = max(0.0, min(1.0, float(py2 / h)))

                    cls_name = names.get(int(cls_id), str(int(cls_id)))

                    # Pad person bbox by 25% on each side so carrying bag stays inside crop
                    pad_w = int(round(self.crop_pad_ratio * box_w)) if cls_name == "person" else int(round(0.05 * box_w))
                    pad_h = int(round(self.crop_pad_ratio * box_h)) if cls_name == "person" else int(round(0.05 * box_h))
                    cpx1 = max(0, px1 - pad_w)
                    cpy1 = max(0, py1 - pad_h)
                    cpx2 = min(w, px2 + pad_w)
                    cpy2 = min(h, py2 + pad_h)

                    crop = None
                    if (cpx2 - cpx1) >= self.min_crop_short_side and (cpy2 - cpy1) >= self.min_crop_short_side:
                        crop = frame_img[cpy1:cpy2, cpx1:cpx2].copy()

                    bbox_tuple = (norm_x1, norm_y1, norm_x2, norm_y2)
                    quality = compute_quality_score(crop, bbox_tuple, float(conf)) if crop is not None else 0.0

                    frame_boxes.append(
                        Box(
                            x1=norm_x1,
                            y1=norm_y1,
                            x2=norm_x2,
                            y2=norm_y2,
                            cls=cls_name,
                            conf=float(conf),
                            quality=quality,
                            crop=crop,
                            px1=px1,
                            py1=py1,
                            px2=px2,
                            py2=py2,
                            frame_w=w,
                            frame_h=h,
                        )
                    )

            all_frame_boxes.append(frame_boxes)

        return all_frame_boxes


class ClosedSetDetector(BaseDetector):
    """Closed-set YOLO detector adapter (YOLOv8s on COCO)."""

    def __init__(
        self,
        model_name: str | Path = "models/yolov8s.pt",
        conf_thresh: float = 0.40,
        min_box_height: int = 64,
        min_box_area_ratio: float = 0.002,
        crop_pad_ratio: float = 0.25,
        min_crop_short_side: int = 64,
        classes: list[int] | None = None,
        device: str = "auto",
        fp16: bool = True,
    ):
        from ultralytics import YOLO

        self.conf_thresh = conf_thresh
        self.min_box_height = min_box_height
        self.min_box_area_ratio = min_box_area_ratio
        self.crop_pad_ratio = crop_pad_ratio
        self.min_crop_short_side = min_crop_short_side
        # COCO target classes: person(0), bicycle(1), car(2), motorcycle(3), bus(5), truck(7),
        # backpack(24), umbrella(25), handbag(26), suitcase(28)
        self.classes = classes if classes is not None else list(COCO_TARGET_CLASSES.keys())

        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.fp16 = fp16 and ("cuda" in self.device)
        logger.info(
            "Loading ClosedSet YOLO detector '%s' on %s (fp16=%s, conf=%.2f, classes=%s)",
            model_name, self.device, self.fp16, self.conf_thresh, self.classes
        )
        self.model = YOLO(str(model_name))
        self.model_name = str(model_name)

    def detect(self, frames: list[np.ndarray] | np.ndarray) -> list[list[Box]]:
        if isinstance(frames, np.ndarray) and frames.ndim == 3:
            input_frames = [frames]
        else:
            input_frames = list(frames)

        if not input_frames:
            return []

        results = self.model.predict(
            source=input_frames,
            conf=self.conf_thresh,
            classes=self.classes,
            imgsz=640,
            device=self.device,
            half=self.fp16,
            verbose=False,
        )

        all_frame_boxes: list[list[Box]] = []

        for frame_img, res in zip(input_frames, results, strict=False):
            h, w = frame_img.shape[:2]
            frame_area = w * h
            frame_boxes: list[Box] = []

            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy()
                names = res.names

                for box_coords, conf, cls_id in zip(xyxy, confs, clss, strict=False):
                    bx1, by1, bx2, by2 = box_coords
                    px1 = max(0, int(round(bx1)))
                    py1 = max(0, int(round(by1)))
                    px2 = min(w, int(round(bx2)))
                    py2 = min(h, int(round(by2)))

                    box_h = py2 - py1
                    box_w = px2 - px1
                    box_area = box_w * box_h

                    # Drop boxes under minimum height or minimum area
                    if box_h < self.min_box_height or box_area < (self.min_box_area_ratio * frame_area):
                        continue

                    norm_x1 = max(0.0, min(1.0, float(px1 / w)))
                    norm_y1 = max(0.0, min(1.0, float(py1 / h)))
                    norm_x2 = max(0.0, min(1.0, float(px2 / w)))
                    norm_y2 = max(0.0, min(1.0, float(py2 / h)))

                    cls_name = names.get(int(cls_id), str(int(cls_id)))

                    # 25% padding for person crops
                    pad_w = int(round(self.crop_pad_ratio * box_w)) if cls_name == "person" else int(round(0.05 * box_w))
                    pad_h = int(round(self.crop_pad_ratio * box_h)) if cls_name == "person" else int(round(0.05 * box_h))
                    cpx1 = max(0, px1 - pad_w)
                    cpy1 = max(0, py1 - pad_h)
                    cpx2 = min(w, px2 + pad_w)
                    cpy2 = min(h, py2 + pad_h)

                    crop = None
                    if (cpx2 - cpx1) >= self.min_crop_short_side and (cpy2 - cpy1) >= self.min_crop_short_side:
                        crop = frame_img[cpy1:cpy2, cpx1:cpx2].copy()

                    bbox_tuple = (norm_x1, norm_y1, norm_x2, norm_y2)
                    quality = compute_quality_score(crop, bbox_tuple, float(conf)) if crop is not None else 0.0

                    frame_boxes.append(
                        Box(
                            x1=norm_x1,
                            y1=norm_y1,
                            x2=norm_x2,
                            y2=norm_y2,
                            cls=cls_name,
                            conf=float(conf),
                            quality=quality,
                            crop=crop,
                            px1=px1,
                            py1=py1,
                            px2=px2,
                            py2=py2,
                            frame_w=w,
                            frame_h=h,
                        )
                    )

            all_frame_boxes.append(frame_boxes)

        return all_frame_boxes


def get_detector(config: OmniWatchConfig) -> BaseDetector:
    """Instantiates detector corresponding to active config preset."""
    preset = config.current
    device = config.resolve_device()
    model_name = preset.detector_model

    # Check local models/ directory first
    local_path = Path("models") / Path(model_name).name
    effective_path = local_path if local_path.exists() else model_name

    if "world" in str(effective_path).lower():
        return YOLOWorldDetector(
            model_name=effective_path,
            vocabulary=config.detector_vocab,
            conf_thresh=preset.detector_conf,
            device=device,
        )
    return ClosedSetDetector(
        model_name=effective_path,
        conf_thresh=preset.detector_conf,
        device=device,
    )
