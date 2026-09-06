"""Read-only CPU audit of model shapes, cache identities, and installed versions.

Loads ONNX protobufs and queries NVML metadata; never imports TensorRT/PyCUDA,
creates a CUDA context, deserializes a TensorRT engine/cache, or writes caches.
"""

import argparse
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
from datetime import datetime, timezone


def gpu_metadata():
    nvml = ctypes.WinDLL("nvml.dll")
    def check(code):
        if code:
            raise RuntimeError(f"NVML returned {code}")

    check(nvml.nvmlInit_v2())
    try:
        handle = ctypes.c_void_p()
        check(nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)))
        name = ctypes.create_string_buffer(128)
        driver = ctypes.create_string_buffer(128)
        major, minor, temperature = ctypes.c_int(), ctypes.c_int(), ctypes.c_uint()
        check(nvml.nvmlDeviceGetName(handle, name, len(name)))
        check(nvml.nvmlSystemGetDriverVersion(driver, len(driver)))
        check(nvml.nvmlDeviceGetCudaComputeCapability(handle, ctypes.byref(major), ctypes.byref(minor)))
        check(nvml.nvmlDeviceGetTemperature(handle, 0, ctypes.byref(temperature)))
        return {"name": name.value.decode(), "driver": driver.value.decode(),
                "compute_capability": [major.value, minor.value], "temperature_c": temperature.value}
    finally:
        nvml.nvmlShutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--model-root", type=Path, help="Defaults to PROJECT/data/models")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    project, cache_dir = args.project.resolve(), args.cache_dir.resolve()
    spec = importlib.util.spec_from_file_location("audited_cache", project / "ezvtuber-rt/ezvtb_rt/trt_cache.py")
    cache = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cache)
    import onnx

    versions = {name: importlib.metadata.version(name) for name in ("tensorrt-rtx", "pycuda", "onnx", "numpy")}
    gpu = gpu_metadata()
    cc = tuple(gpu["compute_capability"])
    identity = f"{versions['tensorrt-rtx']}|device-id=0|name={gpu['name']}|cc={cc}"
    baseline = json.loads((project / "benchmark_results/2026-09-02/current-tensorrt-standard.json").read_text())
    model_root = (args.model_root or project / "data/models").resolve()
    models = []
    for name in ("decomposer", "combiner", "morpher", "rotator", "editor"):
        source = model_root / "tha3/seperable/fp32" / f"{name}.onnx"
        data = source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        model = onnx.load_model_from_string(data)
        initializer_names = {item.name for item in model.graph.initializer}
        inputs = []
        for item in model.graph.input:
            if item.name in initializer_names:
                continue
            dimensions = [dim.dim_value if dim.HasField("dim_value") else (dim.dim_param or "?")
                          for dim in item.type.tensor_type.shape.dim]
            inputs.append({"name": item.name, "shape": dimensions})
        tokens = {schema: cache._cache_token(schema, identity, cache.BUILDER_CONFIG_FINGERPRINT, digest)
                  for schema in ("1", "2")}
        engine_path = cache_dir / f"{name}-{tokens[cache.ENGINE_CACHE_SCHEMA]}.trt"
        runtime_paths = {schema: cache_dir / f"{name}-{token}.runtime.cache" for schema, token in tokens.items()}
        models.append({"name": name, "source": str(source), "onnx_sha256": digest,
                       "matches_baseline_onnx": digest == baseline["model_sha256"][f"{name}.onnx"],
                       "inputs": inputs, "dynamic_inputs": [item["name"] for item in inputs
                                                                if any(not isinstance(d, int) or d < 0 for d in item["shape"])],
                       "engine_exists": engine_path.is_file(), "engine_path": str(engine_path),
                       "runtime": {schema: {"path": str(path), "exists": path.is_file(),
                                            "bytes": path.stat().st_size if path.is_file() else 0}
                                   for schema, path in runtime_paths.items()}})
        del data, model

    forbidden = [name for name in sys.modules if name == "tensorrt_rtx" or name.startswith("pycuda")]
    assert not forbidden, forbidden
    result = {"audited_at_utc": datetime.now(timezone.utc).isoformat(), "mode": "cpu_onnx_and_nvml_metadata_only",
              "python": sys.version, "versions": versions, "gpu": gpu, "identity": identity,
              "engine_schema": cache.ENGINE_CACHE_SCHEMA, "runtime_schema": cache.RUNTIME_CACHE_SCHEMA,
              "models": models,
              "gpu_inference_performed": False, "gpu_context_created": False, "cache_files_modified": False}
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"versions": versions, "gpu": gpu, "models": [
        {"name": item["name"], "inputs": item["inputs"], "engine_exists": item["engine_exists"],
         "schema1_exists": item["runtime"]["1"]["exists"], "schema2_exists": item["runtime"]["2"]["exists"]}
        for item in models], "output": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
