import tempfile
from pathlib import Path
import unittest

from src.utils.student_models import (
    scan_student_models,
    student_character_path,
    student_model_directory,
)


class StudentModelPackageTests(unittest.TestCase):
    @staticmethod
    def make_package(root: Path, name: str, files) -> Path:
        model_dir = root / name
        model_dir.mkdir()
        for filename in files:
            (model_dir / filename).write_bytes(b"test")
        return model_dir

    def test_scan_accepts_complete_onnx_packages_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_package(
                root,
                "z-valid",
                ("face_morpher.onnx", "body_morpher.onnx", "character.png"),
            )
            self.make_package(
                root,
                "a-valid",
                ("face_morpher.onnx", "body_morpher.onnx", "character.png"),
            )
            self.make_package(
                root,
                "trt-only",
                ("face_morpher.trt", "body_morpher.trt", "character.png"),
            )
            self.make_package(
                root,
                "missing-character",
                ("face_morpher.onnx", "body_morpher.onnx"),
            )

            self.assertEqual(scan_student_models(root), ["a-valid", "z-valid"])

    def test_character_path_comes_from_the_selected_student_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = self.make_package(
                root,
                "demo",
                ("face_morpher.onnx", "body_morpher.onnx", "character.png"),
            )

            self.assertEqual(
                student_character_path(root, "demo"),
                (package / "character.png").resolve(),
            )

    def test_model_name_cannot_escape_the_configured_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "Invalid"):
                student_model_directory(temp_dir, "../outside")


if __name__ == "__main__":
    unittest.main()
