# 03 — Open-vocabulary detection, crop embeddings, attributes
**Read:** docs/00_PRD (FR-DET), docs/01_ARCHITECTURE (3, 7), docs/02 (baseline needs the same detector).

**Task:**
1. `detect/detector.py`: a `Detector` interface `detect(frames) -> list[list[Box(x1,y1,x2,y2,cls,conf)]]` with two adapters: (a) open-vocabulary YOLO-World-class detector with a vocabulary from `config.yaml` (default list: person, car, truck, bus, motorcycle, bicycle, backpack, handbag, suitcase, box, dog, plus a few others you list in config); (b) closed-set detector selectable by config. Verify the package and checkpoint names exist and load before coding the adapter; record them in `models/MANIFEST.md`. Batch inference, fp16 on GPU if available.
2. `embed/encoder.py`: image and text encoders from the same CLIP/SigLIP-family model (open_clip or transformers, verify), batched, L2-normalized float32, `encode_text(prompts)` with a prompt-ensemble helper ("a photo of {x}", "a surveillance camera image of {x}", "{x}").
3. `embed/attrs.py`: (a) `color_names(crop)`: convert to a perceptual space, cluster dominant colors with a small k, map to a named palette defined in config (red, orange, yellow, green, blue, purple, pink, brown, black, white, gray, silver, beige), ignore near-background pixels with a simple mask; for persons use the torso region, for vehicles the central body region. (b) `carries(person_box, bag_boxes)`: association by overlap/adjacency thresholds from config.
4. A crop quality score (box size, sharpness, detector confidence) saved as `quality`.
5. Unit tests with synthetic colored rectangles for color naming (red/blue/white/black) and carries association; a smoke test that loads detector + encoder on a real frame and checks output shapes.

**Constraints:** no per-query model calls here; this is index-time. Log which models/devices are active to `/api/health` state.
**Acceptance:** run on one real frame; save an annotated image to `benchmark/` showing boxes + color labels; report per-frame detector and encoder timing from your run.
