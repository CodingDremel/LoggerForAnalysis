# test_modules.py
# This script tests individual modules for expected behavior

from safe_remote_ops.config import parser
from safe_remote_ops.core import logger, operations, validator
from safe_remote_ops.platform import base, windows, windows_remote, unix_local, unix_remote
from safe_remote_ops.utils import file_ops, integrity, path_utils

def test_imports():
    print("✅ All modules imported successfully.")

if __name__ == "__main__":
    test_imports()
