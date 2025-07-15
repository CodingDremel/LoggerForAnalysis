import os
import sys
import stat
import logging

from typing import List, Dict, Tuple
from pathlib import Path

from core.logger import SafeLogger

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

def strip_common_prefix_sources(sources: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Strips the common path prefix from the 'source' field in a list of dicts (sources).
    Returns a new list with updated 'source' values, using project path utilities.
    """
    if not sources:
        return sources

    # Normalize all source paths
    source_paths = [normalize_path(s['source']) for s in sources if 'source' in s]
    if not source_paths:
        return sources

    common_prefix, stripped_paths = strip_common_path_prefix(source_paths)
    # Map the stripped paths back into the dicts
    result = []
    for s, new_source in zip(sources, stripped_paths):
        updated = s.copy()
        updated['source'] = new_source
        result.append(updated)
    return result

def check_path_status(path_str: str,
                      system_type: str = None,  # "unix" or "windows"
                      check_types_to_ignore=None,  # e.g. ["symlink", "stat", "reparse"]
                      exceptions_to_ignore=None,    # e.g. ["PermissionError", "FileNotFoundError"]
                      path_status_logger: SafeLogger = None
    ) -> bool:
    """
    True if given absolute path exists and is accessible as file, dir or mount point,
    or is a symlink, device, FIFO, or socket (unless ignored). Handles exceptions as specified.
    Prints diagnostic info at each step.
    """
    if check_types_to_ignore is None:
        check_types_to_ignore = []
    if exceptions_to_ignore is None:
        exceptions_to_ignore = []

    path = Path(path_str)
    sys_type = system_type or ("windows" if os.name == "nt" else "unix")

    def handle_exception(e, check_type, exception_handling_logger=path_status_logger):
        exc_name = type(e).__name__
        if exc_name in exceptions_to_ignore:
            if exception_handling_logger.log_level <= logging.WARNING:
                exception_handling_logger.logger.debug(
                    f"[IGNORED] {exc_name} during {check_type} check at {path_str}: {e}")
            return False
        exception_handling_logger.logger.debug(f"[EXCEPTION] {exc_name} during {check_type} check at {path_str}: {e}")
        raise

    path_status_logger.logger.debug(f"[DEBUG] Checking status for: {path_str} (system_type={sys_type})")

    # Standard existence and type checks
    try:
        if path.exists():
            path_status_logger.logger.debug("[DEBUG] Path exists.")
            if path.is_file():
                path_status_logger.logger.debug("[DEBUG] Path is a file.")
                return True
            elif path.is_dir():
                path_status_logger.logger.debug("[DEBUG] Path is a directory.")
                return True
            elif path.is_mount():
                path_status_logger.logger.debug("[DEBUG] Path is a mount point.")
                return True
        else:
            path_status_logger.logger.debug("[DEBUG] Path does NOT exist.")
            return False
    except Exception as e:
        return handle_exception(e, "exists/file/dir/mount", path_status_logger)

    # Symlink check (unless ignored)
    if "symlink" not in check_types_to_ignore:
        try:
            if path.is_symlink():
                path_status_logger.logger.debug("[DEBUG] Path is a symlink.")
                try:
                    resolved = path.resolve(strict=True)
                    if resolved.exists() and (resolved.is_file() or resolved.is_dir() or resolved.is_mount()):
                        path_status_logger.logger.debug("[DEBUG] Symlink target exists and is of valid type.")
                        return True
                except Exception as e:
                    return handle_exception(e, "symlink_resolve", path_status_logger)
        except Exception as e:
            return handle_exception(e, "symlink", path_status_logger)

    # Stat-based special file checks (unless ignored)
    if "stat" not in check_types_to_ignore:
        try:
            mode = path.stat().st_mode
            if stat.S_ISFIFO(mode):
                path_status_logger.logger.debug("[DEBUG] Path is a FIFO (named pipe).")
                return True
            if stat.S_ISSOCK(mode):
                path_status_logger.logger.debug("[DEBUG] Path is a socket.")
                return True
            if stat.S_ISCHR(mode):
                path_status_logger.logger.debug("[DEBUG] Path is a character device.")
                return True
            if stat.S_ISBLK(mode):
                path_status_logger.logger.debug("[DEBUG] Path is a block device.")
                return True
        except Exception as e:
            return handle_exception(e, "stat", path_status_logger)

    # Windows-specific reparse point check (unless ignored)
    if sys_type == "windows" and "reparse" not in check_types_to_ignore:
        try:
            import ctypes
            FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
            if attrs != -1 and (attrs & FILE_ATTRIBUTE_REPARSE_POINT):
                path_status_logger.logger.debug("[DEBUG] Path is a reparse point (Windows).")
                return True
        except Exception as e:
            return handle_exception(e, "reparse", path_status_logger)

    path_status_logger.logger.debug("[DEBUG] No valid file type or special case matched, returning False.")
    return False
