from importlib.metadata import version
from pathlib import Path
import unittest


class RequirementLockTests(unittest.TestCase):
    def setUp(self):
        project_root = Path(__file__).resolve().parents[1]
        self.requirement_lines = [
            line.strip()
            for line in (project_root / 'requirements.txt').read_text(
                encoding='utf-8'
            ).splitlines()
            if line.strip() and not line.lstrip().startswith('#')
        ]

    def test_all_runtime_dependencies_are_exactly_pinned_and_installed(self):
        for requirement in self.requirement_lines:
            with self.subTest(requirement=requirement):
                self.assertEqual(requirement.count('=='), 1)
                package_name, expected_version = requirement.split('==')
                self.assertEqual(version(package_name), expected_version)

    def test_only_one_opencv_distribution_is_requested(self):
        opencv_requirements = [
            requirement
            for requirement in self.requirement_lines
            if requirement.lower().startswith('opencv-')
        ]
        self.assertEqual(
            opencv_requirements,
            ['opencv-contrib-python==4.13.0.90'],
        )


if __name__ == '__main__':
    unittest.main()
