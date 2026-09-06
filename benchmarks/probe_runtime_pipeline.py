"""Worker implementation for probe_runtime_jit_cache.py --pipeline."""

import hashlib
import json
import os
from pathlib import Path
import sys
import time


def run_pipeline(args, case, stage, durations):
    import cv2
    import numpy as np
    from PIL import Image
    from probe_runtime_jit_cache import write_json

    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "ezvtuber-rt"))
    sys.path.insert(0, str(project))
    os.environ["EZVTB_GPU_DUTY_LIMIT"] = "80"
    os.environ["EZVTB_DEVICE_ID"] = str(args.device)
    os.environ["EZVTB_TRT_RUNTIME_CACHE"] = "" if args.runtime_cache == "off" else "1"
    cache_dir = case / "runtime-cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ["EZVTB_TRT_CACHE_DIR"] = str(cache_dir)

    import ezvtb_rt
    import ezvtb_rt.trt_utils as utils
    import ezvtb_rt.trt_engine as engine_module
    from ezvtb_rt.core_trt import CoreTRT
    from src.utils.preprocess import clear_transparent_rgb

    baseline = json.loads((project / "benchmark_results/2026-09-02/current-tensorrt-standard.json").read_text())
    engine_paths = {name: Path(path) for name, path in baseline["engine_sources"].items()}
    if any(not path.is_file() for path in engine_paths.values()):
        raise RuntimeError("An existing engine is missing; rebuilding is prohibited")
    model_root = project / "data/models"
    for name in engine_paths:
        model = model_root / "tha3/seperable/fp32" / f"{name}.onnx"
        if hashlib.sha256(model.read_bytes()).hexdigest() != baseline["model_sha256"][f"{name}.onnx"]:
            raise RuntimeError(f"Model changed: {model}")

    class EngineProxy:
        def __init__(self, engine, name):
            self._engine, self._name = engine, name

        def __getattr__(self, name):
            return getattr(self._engine, name)

        def create_execution_context(self, config):
            if args.graph == "off":
                config.cuda_graph_strategy = utils.trt.CudaGraphStrategy.DISABLED
            return stage(f"{self._name}.context", lambda: self._engine.create_execution_context(config))

    def load_existing(source):
        name = Path(source).stem
        engine = stage(f"{name}.deserialize", lambda: utils._deserialize_engine(engine_paths[name]))
        if engine is None:
            raise RuntimeError(f"Cannot deserialize {name}; rebuilding is prohibited")
        return EngineProxy(engine, name)

    def forbid_build(*args, **kwargs):
        raise RuntimeError("Engine building is prohibited in this probe")

    original_load_cache = engine_module.load_runtime_cache
    original_save_cache = engine_module.save_runtime_cache
    engine_module.load_engine = load_existing
    utils.build_engine = forbid_build
    engine_module.get_runtime_cache_path = lambda source: cache_dir / f"{Path(source).stem}.runtime.cache"
    engine_module.load_runtime_cache = lambda config, path: stage(
        f"{Path(path).stem}.load", lambda: original_load_cache(config, path))
    engine_module.save_runtime_cache = lambda cache, path: stage(
        f"{Path(path).stem}.save", lambda: original_save_cache(cache, path))

    ezvtb_rt.init_model_path(str(model_root))
    core = stage("core.construct", lambda: CoreTRT(
        tha_model_version="v3", tha_model_seperable=True, tha_model_fp16=False,
        vram_cache_size=0, cache_max_giga=0, use_eyebrow=False))
    image = clear_transparent_rgb(Image.open(project / "data/images/lambda_00.png").convert("RGBA"))
    input_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
    stage("core.set_image", lambda: core.setImage(input_image))
    rng = np.random.default_rng(20260905)
    poses = [np.zeros((1, 45), dtype=np.float32)]
    poses.extend(rng.uniform(0, 0.15, (1, 45)).astype(np.float32) for _ in range(3))
    hashes = []
    for i, pose in enumerate(poses):
        started = time.perf_counter()
        frames = stage(f"core.inference.{i}", lambda: core.inference([pose.copy()], copy_output=True))
        hashes.append(hashlib.sha256(np.asarray(frames).tobytes()).hexdigest())
        time.sleep(max(0.1, (time.perf_counter() - started) * 0.25))
    result = {"status": "ok", "pipeline": "THA3 separable FP32", "context": args.context,
              "runtime_cache": args.runtime_cache, "graph": args.graph,
              "input_image_sha256": hashlib.sha256(input_image.tobytes()).hexdigest(),
              "pose_sha256": hashlib.sha256(np.asarray(poses).tobytes()).hexdigest(),
              "output_sha256": hashes, "durations_ms": durations,
              "cache_sizes": {path.name: path.stat().st_size for path in cache_dir.glob("*.runtime.cache")}}
    write_json(case / "worker.json", result)
    del core, frames
