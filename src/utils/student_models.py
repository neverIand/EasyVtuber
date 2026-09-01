"""Safe discovery and character lookup for custom THA4 student packages."""

from pathlib import Path
from typing import List, Union


PathLike = Union[str, Path]
REQUIRED_STUDENT_FILES = (
    "face_morpher.onnx",
    "body_morpher.onnx",
    "character.png",
)


def is_student_model_directory(path: PathLike) -> bool:
    model_dir = Path(path)
    return model_dir.is_dir() and all(
        (model_dir / filename).is_file()
        for filename in REQUIRED_STUDENT_FILES
    )


def scan_student_models(root: PathLike) -> List[str]:
    """Return runnable ONNX student packages in stable name order."""
    model_root = Path(root)
    try:
        candidates = list(model_root.iterdir())
    except OSError:
        return []
    return sorted(
        candidate.name
        for candidate in candidates
        if is_student_model_directory(candidate)
    )


def student_model_directory(root: PathLike, model_name: str) -> Path:
    """Resolve one direct child of ``root`` without permitting traversal."""
    if (
            not model_name
            or Path(model_name).name != model_name
            or model_name in (".", "..")
    ):
        raise ValueError("Invalid THA4 student model name")

    model_root = Path(root).resolve()
    model_dir = (model_root / model_name).resolve()
    if model_dir.parent != model_root:
        raise ValueError("THA4 student model must stay inside the model root")
    return model_dir


def student_character_path(root: PathLike, model_name: str) -> Path:
    model_dir = student_model_directory(root, model_name)
    character_path = model_dir / "character.png"
    if not character_path.is_file():
        raise FileNotFoundError(
            f"THA4 student character image is missing: {character_path}"
        )
    return character_path
