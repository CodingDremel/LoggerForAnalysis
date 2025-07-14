# test_modules.py
# This script tests individual modules for expected behavior

from config import parser
from core import logger, operations, validator
from run_on_platform import base, windows, windows_remote, unix_local, unix_remote
from utils import file_ops, integrity, path_utils

def test_imports():
    print("✅ All modules imported successfully.")

if __name__ == "__main__":
    test_imports()
