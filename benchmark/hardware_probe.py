"""Hardware Capability and Video Decode Probe for OmniWatch.

Measures and records host hardware specs, Torch device availability,
ffmpeg presence, and video decode throughput on a sample video.
Writes benchmark/hardware.json.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def get_cpu_info() -> dict:
    info = {
        "processor": platform.processor(),
        "architecture": platform.machine(),
        "logical_cores": os.cpu_count() or 1,
    }
    # On Windows, try to get specific CPU model via wmic or PowerShell
    if platform.system() == "Windows":
        try:
            cmd = "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"
            out = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True).strip()
            if out:
                info["cpu_name"] = out
        except Exception:
            pass
    return info


def get_gpu_info() -> dict:
    gpu_data: dict = {"available": False, "devices": []}
    # Check via nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            cmd = [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
            out = subprocess.check_output(cmd, text=True).strip()
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpu_data["devices"].append({
                        "name": parts[0],
                        "memory_total_mb": float(parts[1]),
                        "driver_version": parts[2],
                    })
                    gpu_data["available"] = True
        except Exception:
            pass
    return gpu_data


def get_torch_info() -> dict:
    torch_info: dict = {"installed": False, "cuda_available": False, "devices": []}
    try:
        import torch  # type: ignore[import-untyped]
        torch_info["installed"] = True
        torch_info["version"] = torch.__version__
        torch_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch_info["devices"].append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated_mb": round(torch.cuda.memory_allocated(i) / (1024**2), 2),
                    "memory_total_mb": round(torch.cuda.get_device_properties(i).total_memory / (1024**2), 2),
                })
    except ImportError:
        torch_info["note"] = "PyTorch not yet installed in current environment."
    return torch_info


def get_ffmpeg_info() -> dict:
    ffmpeg_info: dict = {"present": False, "version": None, "path": None}
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        # Check inside virtual environment Scripts
        venv_scripts_ffmpeg = Path(sys.prefix) / "Scripts" / "ffmpeg.exe"
        if venv_scripts_ffmpeg.exists():
            ffmpeg_path = str(venv_scripts_ffmpeg)

    if ffmpeg_path:
        ffmpeg_info["present"] = True
        ffmpeg_info["path"] = ffmpeg_path
        try:
            out = subprocess.check_output([ffmpeg_path, "-version"], text=True, stderr=subprocess.STDOUT)
            first_line = out.splitlines()[0] if out.splitlines() else "unknown"
            ffmpeg_info["version"] = first_line
        except Exception as e:
            ffmpeg_info["error"] = str(e)
    else:
        # Check imageio-ffmpeg as python fallback
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            ffmpeg_info["present"] = True
            ffmpeg_info["path"] = ffmpeg_exe
            ffmpeg_info["version"] = "imageio-ffmpeg bundled binary"
        except Exception:
            pass

    return ffmpeg_info


def generate_synthetic_benchmark_video(out_path: Path, num_frames: int = 150, width: int = 1280, height: int = 720, fps: int = 30) -> Path:
    """Generates a synthetic 5-second test video for decode benchmarking."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    for i in range(num_frames):
        # Draw dynamic color frames with moving bounding box
        frame = np.full((height, width, 3), (i % 255, 120, 200), dtype=np.uint8)
        # Draw moving rectangle
        x = int((i * 8) % (width - 100))
        y = int((i * 4) % (height - 100))
        cv2.rectangle(frame, (x, y), (x + 80, y + 80), (0, 255, 0), -1)
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()
    return out_path


def benchmark_decode_speed(video_path: Path) -> dict:
    """Measures OpenCV video decode throughput and latency."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Failed to open {video_path}"}

    fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0
    t0 = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
    t1 = time.perf_counter()
    cap.release()

    duration = max(t1 - t0, 1e-6)
    decode_fps = frame_count / duration
    latency_per_frame_ms = (duration / max(frame_count, 1)) * 1000.0

    return {
        "video_path": str(video_path),
        "resolution": f"{width}x{height}",
        "total_frames_decoded": frame_count,
        "elapsed_seconds": round(duration, 4),
        "decode_throughput_fps": round(decode_fps, 2),
        "avg_decode_latency_ms": round(latency_per_frame_ms, 3),
        "realtime_factor": round(decode_fps / fps_native, 2),
    }


def run_probe() -> dict:
    print("=" * 60)
    print(" OmniWatch Hardware & Environment Probe")
    print("=" * 60)

    cpu_info = get_cpu_info()
    gpu_info = get_gpu_info()
    torch_info = get_torch_info()
    ffmpeg_info = get_ffmpeg_info()

    benchmark_dir = Path("benchmark")
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    sample_video = benchmark_dir / "sample_probe.mp4"

    if not sample_video.exists():
        print("[Probe] Generating synthetic test video for decode benchmark...")
        generate_synthetic_benchmark_video(sample_video)

    print("[Probe] Measuring OpenCV video decode speed...")
    decode_stats = benchmark_decode_speed(sample_video)

    probe_result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "python_version": sys.version.split()[0],
        },
        "cpu": cpu_info,
        "gpu": gpu_info,
        "torch": torch_info,
        "ffmpeg": ffmpeg_info,
        "video_decode_benchmark": decode_stats,
    }

    out_json = benchmark_dir / "hardware.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(probe_result, f, indent=2)

    # Print human-readable summary
    print(f"\n[CPU] {cpu_info.get('cpu_name', cpu_info['processor'])} ({cpu_info['logical_cores']} logical cores)")
    if gpu_info["available"]:
        for dev in gpu_info["devices"]:
            print(f"[GPU] {dev['name']} - Total Memory: {dev['memory_total_mb']} MB (Driver: {dev['driver_version']})")
    else:
        print("[GPU] No discrete NVIDIA GPU detected via nvidia-smi.")

    print(f"[Torch] Installed: {torch_info['installed']}, CUDA Available: {torch_info['cuda_available']}")
    print(f"[FFmpeg] Present: {ffmpeg_info['present']} (Path: {ffmpeg_info.get('path')})")
    print(f"[Decode] {decode_stats['resolution']} @ {decode_stats['decode_throughput_fps']} FPS "
          f"({decode_stats['realtime_factor']}x realtime, {decode_stats['avg_decode_latency_ms']} ms/frame)")
    print(f"\n[Hardware Probe Complete] Report saved to: {out_json}")
    print("=" * 60)

    return probe_result


if __name__ == "__main__":
    run_probe()
