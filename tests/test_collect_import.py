import unittest
import subprocess
import sys
import importlib.util
from pathlib import Path


class TestCollectImport(unittest.TestCase):
    def test_missing_mediapipe_shows_friendly_message(self):
        if importlib.util.find_spec("mediapipe") is not None:
            self.skipTest("mediapipe installed")
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent.parent / "collect.py")],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("未安装 mediapipe", proc.stderr)


if __name__ == "__main__":
    unittest.main()
