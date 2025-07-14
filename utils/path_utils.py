import os
from pathlib import Path
from typing import List, Tuple

def normalize_path(path: str) -> str:
    return os.path.normpath(path)

def strip_common_path_prefix(paths: List[str]) -> Tuple[str, List[str]]:
    if not paths:
        return "", []
    split_paths = [Path(p).parts for p in paths]
    common = os.path.commonprefix(split_paths)
    common_prefix = os.path.join(*common)
    stripped = [os.path.join(*p[len(common):]) for p in split_paths]
    return common_prefix, stripped
