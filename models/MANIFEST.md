# Model Manifest — OmniWatch

Model files are cached locally before ingest or query. The intended runtime is
offline after the required models have been downloaded and smoke-tested.

## Recommended runtime approach

The primary pipeline uses YOLO-World-S and OpenCLIP on CUDA. OpenCLIP is also
the default verifier because it is already validated on the RTX 4050 and is
fast enough for interactive retrieval. SmolVLM2 is an optional verifier for
the top three candidates only; it is not used during ingestion. Qwen is kept
as a larger backup and is not active on this 6 GB VRAM machine unless a
quantized or offloaded inference test passes.

| Role | Model Name / Checkpoint | Source | Version / SHA256 | License Note | Date Downloaded | Verified Status |
|---|---|---|---|---|---|---|
| Open-vocab detector | YOLO-World-S (`models/yolov8s-world.pt`) | Ultralytics | Ultralytics 8.4.174 / checkpoint file present | AGPL-3.0 | 2026-10-08 | **GPU inference verified** |
| Vision-language encoder and default verifier | OpenCLIP ViT-B-32 / `laion2b_s34b_b79k` | OpenCLIP / HuggingFace | open_clip 3.3.0 / model cache present | MIT / model license to confirm | 2026-10-08 | **CUDA text & image embedding verified** |
| Query parser | Rule-based parser; Ollama `llama3.2:1b` fallback | Local regex / Ollama | Ollama 0.31.1 / `llama3.2:1b` | Meta Llama license | 2026-10-08 | **Deterministic rules primary; Ollama fallback verified** |
| Optional VLM verifier | SmolVLM2-2.2B-Instruct (`models/SmolVLM2-2.2B-Instruct`) | HuggingFaceTB / HuggingFace | checkpoint files present / safetensors | Apache-2.0 | 2026-10-08 | **Image inference verified (110.8s on 6GB GPU; triggers OpenCLIP fallback)** |
| Backup VLM verifier | Qwen2.5-VL-3B-Instruct (`models/Qwen2.5-VL-3B-Instruct`) | Qwen / HuggingFace | checkpoint files present / safetensors | Apache-2.0 | 2026-10-08 | **Inactive by default (exceeds 6 GB VRAM)** |
| Tracker | ByteTrack | Ultralytics | Ultralytics ByteTrack adapter | AGPL-3.0 | 2026-10-08 | **Per-camera tracking verified** |

## Verification Notes

- Hardware observed: NVIDIA GeForce RTX 4050 Laptop GPU, CUDA 12.8, 6 GB VRAM.
- YOLO-World-S loaded and ran CUDA inference on test frames.
- OpenCLIP loaded on CUDA/CPU and produced `(1, 512)` normalized image and text embeddings (dot product cosine similarity verified).
- Rule-based parser verified as primary deterministic parser; Ollama `llama3.2:1b` verified as fallback.
- SmolVLM2-2.2B-Instruct ran full image smoke test: `SmolVLMForConditionalGeneration` generated correct text (`No.`). However, on 6 GB VRAM with system display usage, model consumption reached 5.84 GB / 6.00 GB (0.00 GB free), causing GPU memory paging to host RAM and 110.77s inference time per crop. Thus, OpenCLIP + explicit attributes is configured as the active verified fallback to preserve interactive latency.
- Qwen2.5-VL-3B-Instruct (~11.5 GB) is kept strictly offline/inactive per specification to avoid out-of-memory errors on 6 GB hardware.
- No model is marked fully verified without an executed smoke test confirming actual loading and inference.
