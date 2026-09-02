#!/usr/bin/env python3
"""Reproducible current-vs-official EasyVtuber performance comparison.

The matrix runner launches every project/backend pair in a fresh Python process
so DirectML, CUDA, and the two source trees cannot contaminate one another.  GPU
telemetry is collected through NVML directly; this script never starts
``nvidia-smi``.

The official TensorRT runtime at the pinned upstream revision rebuilds every
engine on every launch.  Rebuilding is intentionally blocked here.  For the
official case, the worker injects bytes from the already validated,
content-addressed engine cache generated from the same ONNX files.  The
official runtime still performs its own temporary-file write, deserialization,
context creation, allocation, and inference.  Consequently its startup result
is a favorable lower bound, not a measurement of a real cold engine build.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
import types
from typing import Any, Iterable


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    import numpy as np

    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(array.max()),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _working_set() -> dict[str, int] | None:
    """Read the current process working set without a psutil dependency."""
    if os.name != "nt":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("page_fault_count", wintypes.DWORD),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
            ("quota_non_paged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(
        handle,
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        return None
    return {
        "working_set_bytes": int(counters.working_set_size),
        "peak_working_set_bytes": int(counters.peak_working_set_size),
        "pagefile_bytes": int(counters.pagefile_usage),
        "peak_pagefile_bytes": int(counters.peak_pagefile_usage),
    }


class NvmlSampler:
    """Small direct-NVML sampler with sustained safety cutoffs."""

    class Utilization(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class Memory(ctypes.Structure):
        _fields_ = [
            ("total", ctypes.c_ulonglong),
            ("free", ctypes.c_ulonglong),
            ("used", ctypes.c_ulonglong),
        ]

    def __init__(
        self,
        device_index: int = 0,
        interval: float = 0.5,
        utilization_limit: int = 90,
        temperature_limit: int = 80,
    ) -> None:
        self.device_index = device_index
        self.interval = interval
        self.utilization_limit = utilization_limit
        self.temperature_limit = temperature_limit
        self.samples: list[dict[str, float]] = []
        self.error: str | None = None
        self.stop_reason: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._dll = None
        self._handle = ctypes.c_void_p()

    @staticmethod
    def _check(code: int, operation: str) -> None:
        if code != 0:
            raise RuntimeError(f"{operation} failed with NVML code {code}")

    def start(self) -> None:
        try:
            self._dll = ctypes.WinDLL("nvml.dll")
            self._dll.nvmlInit_v2.restype = ctypes.c_int
            self._dll.nvmlDeviceGetHandleByIndex_v2.argtypes = [
                ctypes.c_uint,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            self._dll.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
            self._dll.nvmlDeviceGetUtilizationRates.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(self.Utilization),
            ]
            self._dll.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
            self._dll.nvmlDeviceGetTemperature.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.POINTER(ctypes.c_uint),
            ]
            self._dll.nvmlDeviceGetTemperature.restype = ctypes.c_int
            self._dll.nvmlDeviceGetMemoryInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(self.Memory),
            ]
            self._dll.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
            self._dll.nvmlDeviceGetPowerUsage.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint),
            ]
            self._dll.nvmlDeviceGetPowerUsage.restype = ctypes.c_int
            self._check(self._dll.nvmlInit_v2(), "nvmlInit_v2")
            self._check(
                self._dll.nvmlDeviceGetHandleByIndex_v2(
                    self.device_index,
                    ctypes.byref(self._handle),
                ),
                "nvmlDeviceGetHandleByIndex_v2",
            )
        except Exception as error:
            self.error = str(error)
            self._dll = None
            return

        self._thread = threading.Thread(
            target=self._run,
            name="easyvtuber-nvml-sampler",
            daemon=True,
        )
        self._thread.start()

    def _sample(self) -> dict[str, float]:
        assert self._dll is not None
        utilization = self.Utilization()
        memory = self.Memory()
        temperature = ctypes.c_uint()
        power_mw = ctypes.c_uint()
        self._check(
            self._dll.nvmlDeviceGetUtilizationRates(
                self._handle,
                ctypes.byref(utilization),
            ),
            "nvmlDeviceGetUtilizationRates",
        )
        self._check(
            self._dll.nvmlDeviceGetTemperature(
                self._handle,
                0,
                ctypes.byref(temperature),
            ),
            "nvmlDeviceGetTemperature",
        )
        self._check(
            self._dll.nvmlDeviceGetMemoryInfo(
                self._handle,
                ctypes.byref(memory),
            ),
            "nvmlDeviceGetMemoryInfo",
        )
        power_w = float("nan")
        if self._dll.nvmlDeviceGetPowerUsage(
            self._handle,
            ctypes.byref(power_mw),
        ) == 0:
            power_w = power_mw.value / 1000.0
        return {
            "monotonic_seconds": time.perf_counter(),
            "gpu_utilization_percent": float(utilization.gpu),
            "memory_utilization_percent": float(utilization.memory),
            "temperature_c": float(temperature.value),
            "vram_used_bytes": float(memory.used),
            "power_w": power_w,
        }

    def _run(self) -> None:
        consecutive_over_limit = 0
        try:
            while not self._stop.is_set():
                sample = self._sample()
                self.samples.append(sample)
                if sample["gpu_utilization_percent"] > self.utilization_limit:
                    consecutive_over_limit += 1
                else:
                    consecutive_over_limit = 0
                if consecutive_over_limit >= 2:
                    self.stop_reason = (
                        f"GPU utilization exceeded {self.utilization_limit}% "
                        "for two consecutive NVML samples"
                    )
                    return
                if sample["temperature_c"] >= self.temperature_limit:
                    self.stop_reason = (
                        f"GPU temperature reached {self.temperature_limit} C"
                    )
                    return
                self._stop.wait(self.interval)
        except Exception as error:
            self.error = str(error)

    def ensure_safe(self) -> None:
        if self.stop_reason is not None:
            raise RuntimeError(self.stop_reason)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._dll is not None:
            try:
                self._dll.nvmlShutdown()
            except Exception:
                pass

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {
                "available": False,
                "error": self.error,
                "stop_reason": self.stop_reason,
                "sample_count": 0,
            }

        def finite_values(key: str) -> list[float]:
            return [
                sample[key]
                for sample in self.samples
                if math.isfinite(sample[key])
            ]

        gpu = finite_values("gpu_utilization_percent")
        memory = finite_values("memory_utilization_percent")
        temperature = finite_values("temperature_c")
        vram = finite_values("vram_used_bytes")
        power = finite_values("power_w")
        return {
            "available": True,
            "error": self.error,
            "stop_reason": self.stop_reason,
            "sample_count": len(self.samples),
            "gpu_average_percent": sum(gpu) / len(gpu),
            "gpu_peak_percent": max(gpu),
            "memory_average_percent": sum(memory) / len(memory),
            "temperature_peak_c": max(temperature),
            "vram_peak_bytes": int(max(vram)),
            "power_peak_w": max(power) if power else None,
        }


class DutyLimiter:
    def __init__(self, percent: float) -> None:
        self.percent = percent
        self._previous_end: float | None = None
        self._previous_active = 0.0

    def wait(self) -> None:
        if self.percent >= 100 or self._previous_end is None:
            return
        cooldown = self._previous_active * (100.0 / self.percent - 1.0)
        remaining = self._previous_end + cooldown - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

    def record(self, started_at: float, finished_at: float) -> None:
        self._previous_active = max(0.0, finished_at - started_at)
        self._previous_end = finished_at


class CurrentPostprocessor:
    def __init__(self, shape: tuple[int, int, int]) -> None:
        from src.utils.frame_transform import (
            apply_output_transform,
            build_output_transform,
        )

        self.apply_output_transform = apply_output_transform
        self.build_output_transform = build_output_transform
        self.bgra = __import__("numpy").empty(shape, dtype="uint8")
        self.destination = __import__("numpy").empty(shape[:2] + (3,), dtype="uint8")

    def __call__(self, frame, position):
        import cv2

        transform = self.build_output_transform(
            position,
            frame.shape,
            extend_movement=True,
            bongo=False,
        )
        bgra = self.apply_output_transform(frame, transform, dst=self.bgra)
        _draw_fixed_debug_overlay(bgra)
        cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR, dst=self.destination)
        return self.destination


class OfficialPostprocessor:
    def __init__(self, shape: tuple[int, int, int]) -> None:
        import numpy as np

        self.destination = np.empty(shape[:2] + (3,), dtype=np.uint8)

    def __call__(self, frame, position):
        import cv2
        import numpy as np

        # Keep NumPy-scalar arithmetic in the same order as the official live
        # path; eagerly converting to Python float shifts the affine matrix by
        # a few millionths and can change edge interpolation bytes.
        scale = position[2] * math.sqrt(True) + 1
        angle = -position[0] * 10 * True
        dx = position[0] * 400 * scale * True
        dy = -position[1] * 600 * scale * True
        matrix = cv2.getRotationMatrix2D(
            (frame.shape[1] / 2, frame.shape[0] / 2),
            angle,
            scale,
        )
        matrix[0, 2] += dx
        matrix[1, 2] += dy
        bgra = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))
        _draw_fixed_debug_overlay(bgra)
        converted = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        # The official live process subsequently copies this allocated result
        # into its shared-memory slot.
        np.copyto(self.destination, converted)
        return self.destination


def _draw_fixed_debug_overlay(frame) -> None:
    import cv2

    lines = (
        "INFER/S: 30.0000",
        "INPUT/S: 60.0000",
        "OUTPUT/S: 30.0000",
        "CALC: 16.00 ms",
        "MEM CACHE: 50.00%",
        "GPU CACHE: 50.00%",
    )
    y = 16
    for line in lines:
        cv2.putText(
            frame,
            line,
            (0, y),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 255, 0),
            1,
        )
        y += 16


def _position(index: int):
    import numpy as np

    phase = index / 60.0
    return np.asarray(
        [
            0.08 * math.sin(phase),
            0.05 * math.cos(phase * 0.83),
            0.04 * math.sin(phase * 0.47),
            0.0,
        ],
        dtype=np.float32,
    )


def _cache_snapshot(core) -> dict[str, Any]:
    def snapshot(cache) -> dict[str, Any] | None:
        if cache is None:
            return None
        result: dict[str, Any] = {}
        for name in (
            "hits",
            "miss",
            "size",
            "current_size",
            "current_bytes",
            "max_bytes",
            "allocations",
            "reuses",
            "evictions",
            "pool_releases",
            "retained_bytes",
        ):
            value = getattr(cache, name, None)
            if isinstance(value, (int, float)):
                result[name] = value
        return result

    return {
        "ram": snapshot(getattr(core, "cacher", None)),
        "sr_ram": snapshot(getattr(core, "sr_cacher", None)),
        "vram": snapshot(getattr(getattr(core, "tha", None), "cacher", None)),
    }


def _disable_frame_caches(core) -> None:
    core.cacher = None
    tha = getattr(core, "tha", None)
    if tha is not None and hasattr(tha, "cacher"):
        tha.cacher = None


def _inference(core, poses, optimized_output: bool):
    if optimized_output:
        return core.inference(poses, copy_output=False)
    return core.inference(poses)


def _run_phase(
    *,
    name: str,
    core,
    poses,
    frame_count: int,
    start_index: int,
    postprocessor,
    duty: DutyLimiter,
    sampler: NvmlSampler,
    optimized_output: bool,
    fps: float | None,
    pose_span: int | None = None,
    checkpoint_count: int = 30,
) -> dict[str, Any]:
    infer_ms: list[float] = []
    post_ms: list[float] = []
    total_ms: list[float] = []
    raw_digest = hashlib.sha256()
    final_digest = hashlib.sha256()
    checkpoint_step = max(1, frame_count // max(1, checkpoint_count))
    deadline_ms = None if fps is None else 1000.0 / fps
    deadline_misses = 0
    phase_wall_started = time.perf_counter()
    phase_cpu_started = time.process_time()
    next_deadline = phase_wall_started

    for offset in range(frame_count):
        pose_offset = offset if pose_span is None else offset % pose_span
        index = (start_index + pose_offset) % len(poses)
        duty.wait()
        frame_started = time.perf_counter()
        inference_started = frame_started
        output = _inference(
            core,
            [poses[index].reshape(1, 45).copy()],
            optimized_output,
        )
        inference_finished = time.perf_counter()
        duty.record(inference_started, inference_finished)
        processed = postprocessor(output[0], _position(index))
        frame_finished = time.perf_counter()

        infer_duration_ms = (inference_finished - inference_started) * 1000.0
        post_duration_ms = (frame_finished - inference_finished) * 1000.0
        total_duration_ms = (frame_finished - frame_started) * 1000.0
        infer_ms.append(infer_duration_ms)
        post_ms.append(post_duration_ms)
        total_ms.append(total_duration_ms)
        if deadline_ms is not None and total_duration_ms > deadline_ms:
            deadline_misses += 1

        if offset % checkpoint_step == 0 or offset == frame_count - 1:
            raw_digest.update(memoryview(output[0]).cast("B"))
            final_digest.update(memoryview(processed).cast("B"))

        if offset % 300 == 0:
            sampler.ensure_safe()
            print(
                f"BENCH {name} progress {offset}/{frame_count}",
                flush=True,
            )

        if fps is not None:
            next_deadline += 1.0 / fps
            remaining = next_deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)

    sampler.ensure_safe()
    wall_seconds = time.perf_counter() - phase_wall_started
    cpu_seconds = time.process_time() - phase_cpu_started
    return {
        "name": name,
        "frames": frame_count,
        "paced_fps": fps,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "effective_fps": frame_count / wall_seconds,
        "deadline_ms": deadline_ms,
        "deadline_misses": deadline_misses,
        "inference": _stats(infer_ms),
        "postprocess": _stats(post_ms),
        "processing_total": _stats(total_ms),
        "checkpoint_step": checkpoint_step,
        "raw_checkpoint_sha256": raw_digest.hexdigest(),
        "final_bgr_checkpoint_sha256": final_digest.hexdigest(),
        "cache_after": _cache_snapshot(core),
        "memory_after": _working_set(),
    }


def _model_hashes(model_root: Path, backend: str) -> dict[str, str]:
    model_dir = model_root / "tha3" / "seperable" / "fp32"
    names = (
        ("decomposer.onnx", "merge_no_eyebrow.onnx")
        if backend == "directml"
        else (
            "decomposer.onnx",
            "combiner.onnx",
            "morpher.onnx",
            "rotator.onnx",
            "editor.onnx",
        )
    )
    return {name: _sha256_file(model_dir / name) for name in names}


def _directml_devices() -> list[dict[str, Any]]:
    import onnxruntime as ort

    devices: list[dict[str, Any]] = []
    for ep_device in ort.get_ep_devices():
        if getattr(ep_device, "ep_name", "") != "DmlExecutionProvider":
            continue
        options = dict(getattr(ep_device, "ep_options", {}) or {})
        hardware = getattr(ep_device, "device", None)
        metadata = dict(getattr(hardware, "metadata", {}) or {})
        devices.append(
            {
                "device_id": options.get("device_id"),
                "description": metadata.get("Description"),
                "discrete": metadata.get("Discrete"),
                "dxgi_high_performance_index": metadata.get(
                    "DxgiHighPerformanceIndex"
                ),
                "vendor": getattr(hardware, "vendor", None),
            }
        )
    return devices


def _inject_or_guard_engine_builds(
    label: str,
    engine_cache: Path,
) -> dict[str, str]:
    import ezvtb_rt.trt_utils as trt_utils

    def traced_get_engine_cache_path(path: str) -> Path:
        print(
            f"BENCH resolving engine cache path for {Path(path).name}",
            flush=True,
        )
        print("BENCH querying TensorRT/GPU cache identity", flush=True)
        cache_identity = trt_utils._trt_cache_identity()
        print(
            f"BENCH resolved TensorRT/GPU cache identity: {cache_identity}",
            flush=True,
        )
        resolved_path = trt_utils._get_engine_cache_path(path, cache_identity)
        print(
            f"BENCH resolved engine cache path for {Path(path).name}: "
            f"{resolved_path}",
            flush=True,
        )
        return resolved_path

    trt_utils.get_engine_cache_path = traced_get_engine_cache_path

    resolved: dict[str, str] = {}
    for stem in ("decomposer", "combiner", "morpher", "rotator", "editor"):
        candidates = sorted(engine_cache.glob(f"{stem}-*.trt"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one validated {stem} engine in "
                f"{engine_cache}, found {len(candidates)}"
            )
        resolved[stem] = str(candidates[0])

    if label == "official":
        def inject(path: str) -> bytes:
            stem = Path(path).stem
            if stem not in resolved:
                raise RuntimeError(f"No validated cached engine for {path}")
            print(f"BENCH injecting validated engine bytes for {stem}", flush=True)
            return Path(resolved[stem]).read_bytes()

        trt_utils.build_engine = inject
    else:
        def reject(path: str) -> bytes:
            raise RuntimeError(
                f"Safety guard blocked an unexpected TensorRT build: {path}"
            )

        trt_utils.build_engine = reject
    return resolved


def worker(args: argparse.Namespace) -> int:
    source_root = _path(args.source_root)
    model_root = _path(args.model_root)
    image_path = _path(args.image_path)
    pose_path = _path(args.pose_path)
    engine_cache = _path(args.engine_cache)
    result_path = _path(args.result)
    result: dict[str, Any] = {
        "status": "running",
        "label": args.label,
        "backend": args.backend,
        "variant": args.variant,
        "source_root": str(source_root),
        "model_root": str(model_root),
        "image_path": str(image_path),
        "pose_path": str(pose_path),
        "settings": {
            "model": "THA3 separable FP32",
            "eyebrow": False,
            "rife": False,
            "super_resolution": False,
            "simplify": 2,
            "ram_cache_gib": 2.0,
            "ram_cache_mode": "raw" if args.label == "current" else "brotli",
            "vram_cache_gib": 2.0,
            "fps": args.fps,
            "gpu_duty_percent": args.duty,
            "dml_device_id": (
                0
                if args.backend == "directml" and args.variant == "native"
                else (1 if args.backend == "directml" else None)
            ),
            "debug_output": True,
            "extend_movement": True,
        },
    }
    sampler = NvmlSampler(
        utilization_limit=int(args.max_gpu_utilization),
        temperature_limit=int(args.max_temperature),
    )
    core = None
    try:
        print(
            f"BENCH worker start label={args.label} backend={args.backend}",
            flush=True,
        )
        sys.path.insert(0, str(source_root / "ezvtuber-rt"))
        sys.path.insert(0, str(source_root))
        # src.args parses argv during import.  Use the exact launcher setting
        # relevant to this benchmark and hide benchmark-only arguments.
        sys.argv = [sys.argv[0], "--simplify", "2"]

        import cv2
        import numpy as np
        from PIL import Image

        pose_import_started = time.perf_counter()
        from src.utils.pose_simplify import pose_simplify
        result["pose_simplify_import_ms"] = (
            time.perf_counter() - pose_import_started
        ) * 1000.0

        raw_poses = np.asarray(
            json.loads(pose_path.read_text(encoding="utf-8")),
            dtype=np.float32,
        )
        first_pose_started = time.perf_counter()
        pose_simplify(raw_poses[0].copy())
        result["pose_simplify_first_call_ms"] = (
            time.perf_counter() - first_pose_started
        ) * 1000.0
        simplify_started = time.perf_counter()
        simplified_poses = np.stack(
            [pose_simplify(pose.copy())[0] for pose in raw_poses],
            axis=0,
        )
        result["pose_simplify_5000_ms"] = (
            time.perf_counter() - simplify_started
        ) * 1000.0
        pose_bytes = np.ascontiguousarray(simplified_poses).tobytes()
        normalized = simplified_poses.copy()
        normalized[normalized == 0] = 0.0
        result["pose_sha256"] = hashlib.sha256(pose_bytes).hexdigest()
        result["pose_normalized_zero_sha256"] = hashlib.sha256(
            np.ascontiguousarray(normalized).tobytes()
        ).hexdigest()
        result["pose_unique_count"] = len(
            {pose.tobytes() for pose in simplified_poses}
        )

        preprocess_started = time.perf_counter()
        image = Image.open(image_path).convert("RGBA")
        if args.label == "current":
            from src.utils.preprocess import clear_transparent_rgb

            image = clear_transparent_rgb(image)
        else:
            # Reproduce the official src.main pixel loop.  It is deliberately
            # kept here instead of importing a helper because the official
            # revision has no clear_transparent_rgb function.
            width, _ = image.size
            for pixel_index, pixel in enumerate(image.getdata()):
                if pixel[3] <= 0:
                    y = pixel_index // width
                    x = pixel_index % width
                    image.putpixel((x, y), (0, 0, 0, 0))
        input_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGBA2BGRA)
        result["image_preprocess_ms"] = (
            time.perf_counter() - preprocess_started
        ) * 1000.0
        result["preprocessed_image_sha256"] = hashlib.sha256(
            np.ascontiguousarray(input_image).tobytes()
        ).hexdigest()

        result["model_sha256"] = _model_hashes(model_root, args.backend)
        requested_device_id = (
            "0"
            if args.backend == "directml" and args.variant == "native"
            else ("1" if args.backend == "directml" else "0")
        )
        os.environ["EZVTB_DEVICE_ID"] = requested_device_id
        os.environ["EZVTB_GPU_DUTY_LIMIT"] = str(args.duty)
        os.environ.pop("EZVTB_TRT_RUNTIME_CACHE", None)

        sampler.start()
        runtime_import_started = time.perf_counter()
        official_dml_backend_isolation = (
            args.label == "official"
            and args.backend == "directml"
            and args.variant == "same-device"
        )
        if official_dml_backend_isolation:
            # Official ezvtb_rt eagerly initializes CUDA with EZVTB_DEVICE_ID.
            # On this laptop CUDA ID 1 is invalid while DirectML/DXGI ID 1 is
            # the RTX.  A package shell lets us load the unmodified official
            # CoreORT modules without executing that unrelated eager CUDA path.
            package = types.ModuleType("ezvtb_rt")
            package.__path__ = [
                str(source_root / "ezvtuber-rt" / "ezvtb_rt")
            ]
            package.EZVTB_DATA = str(model_root)

            def init_model_path(custom_path: str) -> None:
                package.EZVTB_DATA = custom_path

            package.init_model_path = init_model_path
            sys.modules["ezvtb_rt"] = package
            ezvtb_rt = package
        else:
            import ezvtb_rt

        ezvtb_rt.init_model_path(str(model_root))
        result["official_dml_backend_isolation"] = (
            official_dml_backend_isolation
        )
        result["directml_devices_at_core_creation"] = (
            _directml_devices() if args.backend == "directml" else None
        )
        result["runtime_import_ms"] = (
            time.perf_counter() - runtime_import_started
        ) * 1000.0
        result["engine_sources"] = None

        if args.backend == "tensorrt":
            result["engine_sources"] = _inject_or_guard_engine_builds(
                args.label,
                engine_cache,
            )
            Core = ezvtb_rt.CoreTRT
        else:
            if official_dml_backend_isolation:
                from ezvtb_rt.core_ort import CoreORT

                Core = CoreORT
            else:
                Core = ezvtb_rt.CoreORT

        core_started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "tha_model_version": "v3",
            "tha_model_seperable": True,
            "tha_model_fp16": False,
            "rife_model_enable": False,
            "sr_model_enable": False,
            "vram_cache_size": 2.0,
            "cache_max_giga": 2.0,
            "use_eyebrow": False,
        }
        if args.label == "current":
            kwargs["cache_storage_mode"] = "raw"
        core = Core(**kwargs)
        result["core_initialization_ms"] = (
            time.perf_counter() - core_started
        ) * 1000.0
        sampler.ensure_safe()

        # The current runtime already cools down after indivisible TensorRT
        # startup calls.  The official runtime has no such hook, so apply a
        # conservative whole-initialization cooldown before adding more work.
        official_startup_cooldown = 0.0
        if args.label == "official" and args.backend == "tensorrt":
            official_startup_cooldown = (
                result["core_initialization_ms"] / 1000.0
            ) * (100.0 / args.duty - 1.0)
            time.sleep(official_startup_cooldown)
        result["official_startup_cooldown_seconds"] = official_startup_cooldown

        set_image_started = time.perf_counter()
        core.setImage(input_image)
        result["set_image_ms"] = (
            time.perf_counter() - set_image_started
        ) * 1000.0

        optimized_output = args.label == "current"
        warmup_started = time.perf_counter()
        _inference(
            core,
            [np.zeros((1, 45), dtype=np.float32)],
            optimized_output,
        )
        warmup_finished = time.perf_counter()
        result["warmup_ms"] = (warmup_finished - warmup_started) * 1000.0
        warmup_cooldown = (warmup_finished - warmup_started) * (
            100.0 / args.duty - 1.0
        )
        if warmup_cooldown > 0:
            time.sleep(warmup_cooldown)
        sampler.ensure_safe()

        postprocessor = (
            CurrentPostprocessor(tuple(input_image.shape))
            if args.label == "current"
            else OfficialPostprocessor(tuple(input_image.shape))
        )
        duty = DutyLimiter(args.duty)
        result["cache_before_stress"] = _cache_snapshot(core)
        result["phases"] = {}
        print("BENCH entering paced stress phase", flush=True)
        stress = _run_phase(
            name="paced_stress",
            core=core,
            poses=simplified_poses,
            frame_count=args.stress_frames,
            start_index=0,
            postprocessor=postprocessor,
            duty=duty,
            sampler=sampler,
            optimized_output=optimized_output,
            fps=args.fps,
        )
        result["phases"]["paced_stress"] = stress

        replay_start = max(0, args.stress_frames - args.cache_replay_span)
        print("BENCH entering warm-cache replay phase", flush=True)
        cache_replay = _run_phase(
            name="warm_cache_replay",
            core=core,
            poses=simplified_poses,
            frame_count=args.cache_replay_frames,
            start_index=replay_start,
            postprocessor=postprocessor,
            duty=duty,
            sampler=sampler,
            optimized_output=optimized_output,
            fps=None,
            pose_span=args.cache_replay_span,
        )
        result["phases"]["warm_cache_replay"] = cache_replay

        _disable_frame_caches(core)
        no_cache_start = args.stress_frames % len(simplified_poses)
        print("BENCH entering no-cache diagnostic phase", flush=True)
        no_cache = _run_phase(
            name="no_cache_diagnostic",
            core=core,
            poses=simplified_poses,
            frame_count=args.no_cache_frames,
            start_index=no_cache_start,
            postprocessor=postprocessor,
            duty=duty,
            sampler=sampler,
            optimized_output=optimized_output,
            fps=None,
        )
        result["phases"]["no_cache_diagnostic"] = no_cache
        result["environment"] = {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "opencv_contrib_python": _package_version("opencv-contrib-python"),
            "onnxruntime_directml": _package_version("onnxruntime-directml"),
            "tensorrt_rtx": _package_version("tensorrt-rtx"),
            "pycuda": _package_version("pycuda"),
        }
        result["memory_final"] = _working_set()
        result["status"] = "ok"
        print("BENCH worker completed", flush=True)
        return_code = 0
    except Exception as error:
        result["status"] = "error"
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
        print(f"BENCH worker failed: {error}", flush=True)
        return_code = 1
    finally:
        sampler.stop()
        result["gpu"] = sampler.summary()
        result["finished_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _atomic_json(result_path, result)
        core = None
        gc.collect()
    return return_code


def matrix(args: argparse.Namespace) -> int:
    current_root = _path(args.current_root)
    official_root = _path(args.official_root)
    python = _path(args.python)
    output_dir = _path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    cases = (
        ("current", "tensorrt", "standard", current_root, args.stress_frames),
        ("official", "tensorrt", "standard", official_root, args.stress_frames),
        ("current", "directml", "auto", current_root, args.stress_frames),
        (
            "official",
            "directml",
            "native",
            official_root,
            min(args.stress_frames, args.official_native_stress_frames),
        ),
        (
            "official",
            "directml",
            "same-device",
            official_root,
            args.stress_frames,
        ),
    )
    if args.cases:
        selected = set(args.cases)
        cases = tuple(
            case
            for case in cases
            if "-".join(case[:3]) in selected
        )
    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case_index, (label, backend, variant, source_root, stress_frames) in enumerate(cases):
        stem = f"{label}-{backend}-{variant}"
        result_path = output_dir / f"{stem}.json"
        log_path = output_dir / f"{stem}.log"
        command = [
            str(python),
            "-B",
            str(script),
            "worker",
            "--label",
            label,
            "--backend",
            backend,
            "--variant",
            variant,
            "--source-root",
            str(source_root),
            "--model-root",
            str(_path(args.model_root)),
            "--image-path",
            str(_path(args.image_path)),
            "--pose-path",
            str(_path(args.pose_path)),
            "--engine-cache",
            str(_path(args.engine_cache)),
            "--result",
            str(result_path),
            "--stress-frames",
            str(stress_frames),
            "--cache-replay-frames",
            str(args.cache_replay_frames),
            "--cache-replay-span",
            str(args.cache_replay_span),
            "--no-cache-frames",
            str(args.no_cache_frames),
            "--fps",
            str(args.fps),
            "--duty",
            str(args.duty),
            "--max-gpu-utilization",
            str(args.max_gpu_utilization),
            "--max-temperature",
            str(args.max_temperature),
        ]
        python_root = python.parent
        environment_root = python_root.parent
        runtime_path_entries = [
            python_root,
            python_root / "Scripts",
            python_root / "Library" / "bin",
        ]
        runtime_path_entries.extend(
            sorted(environment_root.glob("TensorRT-RTX*/bin"))
        )
        child_environment = os.environ.copy()
        child_environment["PATH"] = os.pathsep.join(
            [str(path) for path in runtime_path_entries]
            + [child_environment.get("PATH", "")]
        )
        case_temp_dir = output_dir / "_temporary" / stem
        case_temp_dir.mkdir(parents=True, exist_ok=True)
        child_environment["TEMP"] = str(case_temp_dir)
        child_environment["TMP"] = str(case_temp_dir)
        child_environment["EZVTB_TRT_CACHE_DIR"] = str(
            _path(args.engine_cache)
        )
        print(f"BENCH matrix starting {stem}", flush=True)
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            process = subprocess.Popen(
                command,
                cwd=source_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=child_environment,
            )
            timed_out = threading.Event()

            def terminate_timed_out_worker() -> None:
                if process.poll() is None:
                    timed_out.set()
                    process.kill()

            watchdog = threading.Timer(
                args.case_timeout_seconds,
                terminate_timed_out_worker,
            )
            watchdog.daemon = True
            watchdog.start()
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith("BENCH"):
                    print(f"{stem}: {line.rstrip()}", flush=True)
            return_code = process.wait()
            watchdog.cancel()
        shutil.rmtree(case_temp_dir, ignore_errors=True)

        if not result_path.is_file():
            failure = {
                "label": label,
                "backend": backend,
                "variant": variant,
                "status": "error",
                "error": (
                    f"worker produced no result before the "
                    f"{args.case_timeout_seconds:.1f}s timeout"
                    if timed_out.is_set()
                    else "worker exited without producing a result"
                ),
                "timed_out": timed_out.is_set(),
                "case_timeout_seconds": args.case_timeout_seconds,
            }
            runs.append(failure)
            failures.append(failure)
            aggregate = {
                "status": "error",
                "failed_case": stem,
                "timed_out": timed_out.is_set(),
                "case_timeout_seconds": args.case_timeout_seconds,
                "runs": runs,
            }
            _atomic_json(output_dir / "comparison.json", aggregate)
            print(
                f"BENCH matrix failed {stem}: no result "
                f"(timed_out={timed_out.is_set()})",
                flush=True,
            )
            if not args.continue_on_error:
                return 1
            if case_index != len(cases) - 1 and args.cooldown_seconds > 0:
                print(
                    f"BENCH cooling down for {args.cooldown_seconds:.1f} seconds",
                    flush=True,
                )
                time.sleep(args.cooldown_seconds)
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        runs.append(payload)
        if return_code != 0 or payload.get("status") != "ok":
            failures.append(payload)
            aggregate = {
                "status": "error",
                "failed_case": stem,
                "runs": runs,
            }
            _atomic_json(output_dir / "comparison.json", aggregate)
            if not args.continue_on_error:
                return 1
            if case_index != len(cases) - 1 and args.cooldown_seconds > 0:
                print(
                    f"BENCH cooling down for {args.cooldown_seconds:.1f} seconds",
                    flush=True,
                )
                time.sleep(args.cooldown_seconds)
            continue
        print(f"BENCH matrix completed {stem}", flush=True)
        if case_index != len(cases) - 1 and args.cooldown_seconds > 0:
            print(
                f"BENCH cooling down for {args.cooldown_seconds:.1f} seconds",
                flush=True,
            )
            time.sleep(args.cooldown_seconds)

    def equality(current, official) -> dict[str, bool]:
        return {
            "pose_normalized_zero_equal": (
                current["pose_normalized_zero_sha256"]
                == official["pose_normalized_zero_sha256"]
            ),
            "preprocessed_image_equal": (
                current["preprocessed_image_sha256"]
                == official["preprocessed_image_sha256"]
            ),
            "paced_raw_checkpoints_equal": (
                current["phases"]["paced_stress"]["raw_checkpoint_sha256"]
                == official["phases"]["paced_stress"]["raw_checkpoint_sha256"]
            ),
            "paced_final_checkpoints_equal": (
                current["phases"]["paced_stress"]["final_bgr_checkpoint_sha256"]
                == official["phases"]["paced_stress"]["final_bgr_checkpoint_sha256"]
            ),
            "no_cache_raw_checkpoints_equal": (
                current["phases"]["no_cache_diagnostic"]["raw_checkpoint_sha256"]
                == official["phases"]["no_cache_diagnostic"]["raw_checkpoint_sha256"]
            ),
            "no_cache_final_checkpoints_equal": (
                current["phases"]["no_cache_diagnostic"]["final_bgr_checkpoint_sha256"]
                == official["phases"]["no_cache_diagnostic"]["final_bgr_checkpoint_sha256"]
            ),
        }

    def find_run(label: str, backend: str, variant: str) -> dict[str, Any] | None:
        return next(
            (
                run
                for run in runs
                if run.get("label") == label
                and run.get("backend") == backend
                and run.get("variant") == variant
                and run.get("status") == "ok"
            ),
            None,
        )

    def compare_if_available(
        current: dict[str, Any] | None,
        official: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if current is None or official is None:
            return {
                "status": "unavailable",
                "reason": "one or both benchmark cases did not complete",
            }
        return {"status": "ok", **equality(current, official)}

    current_dml = find_run("current", "directml", "auto")
    official_dml_native = find_run("official", "directml", "native")
    official_dml_same = find_run("official", "directml", "same-device")
    current_trt = find_run("current", "tensorrt", "standard")
    official_trt = find_run("official", "tensorrt", "standard")
    comparisons: dict[str, Any] = {
        "directml_native_default": compare_if_available(
            current_dml,
            official_dml_native,
        ),
        "directml_same_device_control": compare_if_available(
            current_dml,
            official_dml_same,
        ),
        "tensorrt": compare_if_available(current_trt, official_trt),
    }

    aggregate = {
        "status": "partial" if failures else "ok",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script": str(script),
        "matrix": {
            "stress_frames_per_case": args.stress_frames,
            "official_native_directml_stress_frames": min(
                args.stress_frames,
                args.official_native_stress_frames,
            ),
            "cache_replay_frames_per_case": args.cache_replay_frames,
            "no_cache_frames_per_case": args.no_cache_frames,
            "fps": args.fps,
            "duty_percent": args.duty,
            "cooldown_seconds_between_cases": args.cooldown_seconds,
            "selected_cases": args.cases or "all",
        },
        "comparisons": comparisons,
        "failures": failures,
        "runs": runs,
    }
    _atomic_json(output_dir / "comparison.json", aggregate)
    try:
        (output_dir / "_temporary").rmdir()
    except OSError:
        pass
    print("BENCH matrix complete", flush=True)
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--label", choices=("current", "official"), required=True)
    worker_parser.add_argument("--backend", choices=("directml", "tensorrt"), required=True)
    worker_parser.add_argument(
        "--variant",
        choices=("auto", "native", "same-device", "standard"),
        default="standard",
    )
    worker_parser.add_argument("--source-root", required=True)
    worker_parser.add_argument("--model-root", required=True)
    worker_parser.add_argument("--image-path", required=True)
    worker_parser.add_argument("--pose-path", required=True)
    worker_parser.add_argument("--engine-cache", required=True)
    worker_parser.add_argument("--result", required=True)
    worker_parser.add_argument("--stress-frames", type=int, default=3600)
    worker_parser.add_argument("--cache-replay-frames", type=int, default=900)
    worker_parser.add_argument("--cache-replay-span", type=int, default=300)
    worker_parser.add_argument("--no-cache-frames", type=int, default=300)
    worker_parser.add_argument("--fps", type=float, default=30.0)
    worker_parser.add_argument("--duty", type=float, default=90.0)
    worker_parser.add_argument("--max-gpu-utilization", type=float, default=90.0)
    worker_parser.add_argument("--max-temperature", type=float, default=80.0)
    worker_parser.set_defaults(entrypoint=worker)

    matrix_parser = subparsers.add_parser("matrix")
    matrix_parser.add_argument("--current-root", required=True)
    matrix_parser.add_argument("--official-root", required=True)
    matrix_parser.add_argument("--python", required=True)
    matrix_parser.add_argument("--model-root", required=True)
    matrix_parser.add_argument("--image-path", required=True)
    matrix_parser.add_argument("--pose-path", required=True)
    matrix_parser.add_argument("--engine-cache", required=True)
    matrix_parser.add_argument("--output-dir", required=True)
    matrix_parser.add_argument(
        "--cases",
        nargs="+",
        choices=(
            "current-tensorrt-standard",
            "official-tensorrt-standard",
            "current-directml-auto",
            "official-directml-native",
            "official-directml-same-device",
        ),
        help="Run only the named matrix cases (default: all)",
    )
    matrix_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed or timed-out worker and continue remaining cases",
    )
    matrix_parser.add_argument("--stress-frames", type=int, default=3600)
    matrix_parser.add_argument(
        "--official-native-stress-frames",
        type=int,
        default=450,
        help="Bound the slow official default DirectML/Intel diagnostic",
    )
    matrix_parser.add_argument("--cache-replay-frames", type=int, default=900)
    matrix_parser.add_argument("--cache-replay-span", type=int, default=300)
    matrix_parser.add_argument("--no-cache-frames", type=int, default=300)
    matrix_parser.add_argument("--fps", type=float, default=30.0)
    matrix_parser.add_argument("--duty", type=float, default=90.0)
    matrix_parser.add_argument("--max-gpu-utilization", type=float, default=90.0)
    matrix_parser.add_argument("--max-temperature", type=float, default=80.0)
    matrix_parser.add_argument("--cooldown-seconds", type=float, default=15.0)
    matrix_parser.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=300.0,
    )
    matrix_parser.set_defaults(entrypoint=matrix)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.entrypoint(args)


if __name__ == "__main__":
    raise SystemExit(main())
