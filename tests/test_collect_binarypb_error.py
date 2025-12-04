import unittest
from unittest.mock import patch
import importlib
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestBinarypbFallback(unittest.TestCase):
    def setUp(self):
        self.collect = importlib.import_module("collect")

    def test_hands_init_file_not_found_uses_ascii_fallback(self):
        if not self.collect.MP_AVAILABLE:
            self.skipTest("mediapipe not installed")
        ascii_site_packages = r"C:\\Users\\32028\\RPSrobotEnv\\myenv312\\Lib\\site-packages"
        if not os.path.isdir(ascii_site_packages):
            self.skipTest("ascii site-packages missing")
        with patch.object(self.collect.mp_hands, "Hands", side_effect=FileNotFoundError("missing")):
            import sys
            os.environ["MP_RESOURCE_DIR"] = ascii_site_packages
            # avoid interactive capture
            self.collect.GESTURES = []
            sys.argv = ["collect.py"]
            self.collect.main()


if __name__ == "__main__":
    unittest.main()
