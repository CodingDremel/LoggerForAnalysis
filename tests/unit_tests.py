# unit_tests.py
# Unit tests for core logic and utilities

import unittest
from utils import integrity, path_utils

class TestIntegrityUtils(unittest.TestCase):
    def test_md5_and_crc32(self):
        data = b"test data"
        md5 = integrity.compute_md5(data)
        crc = integrity.compute_crc32(data)
        self.assertIsInstance(md5, str)
        self.assertIsInstance(crc, int)

class TestPathUtils(unittest.TestCase):
    def test_normalize_path(self):
        path = "C:/folder//subfolder\\file.txt"
        normalized = path_utils.normalize_path(path)
        self.assertIn("folder", normalized)

if __name__ == "__main__":
    unittest.main()
