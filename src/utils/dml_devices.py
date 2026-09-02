"""DirectML adapter discovery and selection helpers.

ONNX Runtime exposes the exact DXGI adapter index through ``get_ep_devices``.
Keep that import lazy so the wx launcher does not load the DirectML runtime in
its own process; discovery happens in the inference child immediately before
the DirectML backend is created.
"""

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from .gpu_detect import display_adapter_names


@dataclass(frozen=True)
class DirectMLAdapter:
    device_id: int
    description: str
    discrete: Optional[bool] = None
    high_performance_index: Optional[int] = None
    vendor: str = ""

    @property
    def display_label(self) -> str:
        kind = "独显" if self.discrete is True else "核显" if self.discrete is False else "GPU"
        return f"GPU {self.device_id}: {self.description}（{kind}）"


def _optional_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    return None


def discover_directml_adapters(ort_module=None) -> List[DirectMLAdapter]:
    """Return DirectML adapters using ONNX Runtime's actual DXGI indices."""
    if ort_module is None:
        import onnxruntime as ort_module

    get_ep_devices = getattr(ort_module, "get_ep_devices", None)
    if get_ep_devices is None:
        return []

    adapters = []
    for ep_device in get_ep_devices():
        if getattr(ep_device, "ep_name", "") != "DmlExecutionProvider":
            continue
        ep_options = getattr(ep_device, "ep_options", {}) or {}
        device_id = _optional_int(ep_options.get("device_id"))
        if device_id is None or device_id < 0:
            continue

        hardware = getattr(ep_device, "device", None)
        metadata: Mapping = getattr(hardware, "metadata", {}) or {}
        description = str(
            metadata.get("Description")
            or getattr(hardware, "vendor", "")
            or f"DirectML GPU {device_id}"
        )
        adapters.append(
            DirectMLAdapter(
                device_id=device_id,
                description=description,
                discrete=_optional_bool(metadata.get("Discrete")),
                high_performance_index=_optional_int(
                    metadata.get("DxgiHighPerformanceIndex")
                ),
                vendor=str(getattr(hardware, "vendor", "") or ""),
            )
        )

    # Explicit choices are easiest to understand in their DXGI/device order.
    return sorted(adapters, key=lambda adapter: adapter.device_id)


def preferred_directml_adapter(
    adapters: Sequence[DirectMLAdapter],
) -> DirectMLAdapter:
    """Prefer a discrete/high-performance adapter, with stable fallbacks."""
    if not adapters:
        raise RuntimeError("ONNX Runtime did not report any DirectML GPU adapters")

    return min(
        adapters,
        key=lambda adapter: (
            0 if adapter.discrete is True else 1,
            adapter.high_performance_index
            if adapter.high_performance_index is not None
            else 1_000_000,
            adapter.device_id,
        ),
    )


def select_directml_adapter(
    requested_device_id: Optional[int] = None,
    adapters: Optional[Sequence[DirectMLAdapter]] = None,
    ort_module=None,
) -> DirectMLAdapter:
    """Resolve an explicit adapter or automatically choose the fastest one."""
    if adapters is None:
        adapters = discover_directml_adapters(ort_module)
    if requested_device_id is None:
        return preferred_directml_adapter(adapters)

    requested_device_id = int(requested_device_id)
    for adapter in adapters:
        if adapter.device_id == requested_device_id:
            return adapter
    available = ", ".join(str(adapter.device_id) for adapter in adapters) or "none"
    raise ValueError(
        f"DirectML device {requested_device_id} is unavailable; available device IDs: {available}"
    )


def launcher_directml_choices(
    adapter_names: Optional[Iterable[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Build lightweight launcher choices without importing ONNX Runtime."""
    if adapter_names is None:
        adapter_names = display_adapter_names()
    names = [str(name) for name in adapter_names if str(name)]
    labels = ["Auto（优先高性能独显，推荐）"]
    mappings = ["auto"]
    for device_id, name in enumerate(names):
        labels.append(f"GPU {device_id}: {name}")
        mappings.append(str(device_id))
    if not names:
        labels.append("GPU 0（默认适配器）")
        mappings.append("0")
    return labels, mappings
