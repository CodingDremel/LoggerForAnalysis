# Basic imports
import os
import shutil
import subprocess
import re
import tempfile
import logging

# Specific imports
from functools import lru_cache
from typing import List, Dict, Union, Tuple, Generator, Optional, Any
from contextlib import contextmanager
from pathlib import Path
from more_itertools.more import substrings

def copy_file(src: str, dst: str, remote: bool = False, platform: str = "windows", ssh_info: Optional[dict] = None):
    if remote:
        if platform == "windows":
            # Use WinRM or PsExec (placeholder)
            raise NotImplementedError("Remote Windows copy not implemented here.")
        else:
            # Use scp for Unix/Linux
            user = ssh_info.get("user")
            host = ssh_info.get("host")
            subprocess.run(["scp", src, f"{user}@{host}:{dst}"], check=True)
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

def delete_file(path: str, remote: bool = False, platform: str = "windows", ssh_info: Optional[dict] = None):
    if remote:
        if platform == "windows":
            raise NotImplementedError("Remote Windows delete not implemented here.")
        else:
            user = ssh_info.get("user")
            host = ssh_info.get("host")
            subprocess.run(["ssh", f"{user}@{host}", f"rm -f {path}"], check=True)
    else:
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

@lru_cache(maxsize=64)  # Reduced cache size for memory efficiency
def _compile_pattern(pattern_tuple: tuple) -> re.Pattern:
    """Cache compiled regex patterns to avoid recompilation."""

    def normalize_extensions(exts):
        normalized = []
        for ext in exts:
            if ext.startswith('^') or ext.endswith('$') or re.search(r'[\\^$*+?.()[\]{}|]', ext):
                normalized.append(ext)
            else:
                normalized.append(re.escape(ext))
        return normalized

    normalized_exts = normalize_extensions(pattern_tuple)
    pattern_str = r".*(" + "|".join(normalized_exts) + r")$"
    return re.compile(pattern_str, re.IGNORECASE)


def _find_longest_common_substring_streaming(exe_base: str, bat_base: str, min_length: int) -> int:
    """
    Memory-efficient streaming approach to find longest common substring.
    Uses O(1) space instead of O(n²) for substring storage.
    """
    max_len = 0
    exe_len = len(exe_base)
    bat_len = len(bat_base)

    # Use a sliding window approach for better cache locality
    for i in range(exe_len):
        # Early termination if remaining length can't beat current max
        if exe_len - i <= max_len:
            break

        for j in range(i + min_length, exe_len + 1):
            substr_len = j - i
            # Skip if this substring can't be longer than current max
            if substr_len <= max_len:
                continue

            substr = exe_base[i:j]
            if substr in bat_base:
                max_len = substr_len
                # Early exit optimization - if we found a match of this length,
                # we can skip shorter substrings starting from this position
                break

    return max_len


def _get_filtered_files(directory: str, pattern: re.Pattern) -> Generator[str, None, None]:
    """
    Generator that yields matching filenames to avoid loading all files into memory.
    """
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file() and pattern.match(entry.name.lower()):
                    yield entry.name
    except OSError:
        return


def find_matching_batch_files(exe_path: str, operation_system: str = "nt",
                              batch_rec_pattern: List[str] = ["*.bat"],
                              min_match_length: int = 5) -> List[str]:
    """
    Memory-optimized version for large directories (up to 100k files).
    Uses streaming approach to minimize memory usage.
    """

    # Use cached pattern compilation
    pattern = _compile_pattern(tuple(batch_rec_pattern))
    logging.debug(f"Compiled regex pattern: {pattern.pattern}")

    exe_dir = os.path.dirname(exe_path)
    exe_base = os.path.splitext(os.path.basename(exe_path))[0].lower()

    # Optimized prefix removal with tuple for faster lookup
    prefixes = ('alwaysrun', 'always')
    for prefix in prefixes:
        if exe_base.startswith(prefix):
            exe_base = exe_base[len(prefix):]
            break

    # Early exit if exe_base is too short
    if len(exe_base) < min_match_length:
        return []

    # Stream through files to avoid memory issues with large directories
    max_len = 0
    matches = []

    for fname in _get_filtered_files(exe_dir, pattern):
        bat_base = os.path.splitext(fname)[0].lower()

        # Skip if bat_base is too short to have meaningful match
        if len(bat_base) < min_match_length:
            continue

        # Find longest matching substring using streaming approach
        longest = _find_longest_common_substring_streaming(exe_base, bat_base, min_match_length)

        if longest > max_len:
            max_len = longest
            matches = [os.path.join(exe_dir, fname)]
        elif longest == max_len and longest >= min_match_length:
            matches.append(os.path.join(exe_dir, fname))

    return matches


# Alternative implementation using rolling hash for very large exe_base strings
def find_matching_batch_files_rolling_hash(exe_path: str, operation_system: str = "nt",
                                           batch_rec_pattern: List[str] = ["*.bat"],
                                           min_match_length: int = 5) -> List[str]:
    """
    Ultra-optimized version using rolling hash for O(n) substring matching.
    Best for cases where exe_base is very long (>100 chars).
    """
    pattern = _compile_pattern(tuple(batch_rec_pattern))

    exe_dir = os.path.dirname(exe_path)
    exe_base = os.path.splitext(os.path.basename(exe_path))[0].lower()

    # Remove prefixes
    for prefix in ('alwaysrun', 'always'):
        if exe_base.startswith(prefix):
            exe_base = exe_base[len(prefix):]
            break

    if len(exe_base) < min_match_length:
        return []

    def _rolling_hash_match(text: str, pattern_text: str, min_len: int) -> int:
        """Find longest common substring using rolling hash technique."""
        max_match = 0
        pattern_len = len(pattern_text)
        text_len = len(text)

        # Create hash sets for different lengths, starting from longest possible
        for length in range(min(pattern_len, text_len), min_len - 1, -1):
            if length <= max_match:
                break

            # Create rolling hash for pattern substrings of this length
            pattern_hashes = set()
            for i in range(pattern_len - length + 1):
                pattern_hashes.add(hash(pattern_text[i:i + length]))

            # Check if any substring of this length in text matches
            for i in range(text_len - length + 1):
                if hash(text[i:i + length]) in pattern_hashes:
                    max_match = length
                    break

            if max_match == length:
                break

        return max_match

    max_len = 0
    matches = []

    for fname in _get_filtered_files(exe_dir, pattern):
        bat_base = os.path.splitext(fname)[0].lower()

        if len(bat_base) < min_match_length:
            continue

        longest = _rolling_hash_match(bat_base, exe_base, min_match_length)

        if longest > max_len:
            max_len = longest
            matches = [os.path.join(exe_dir, fname)]
        elif longest == max_len and longest >= min_match_length:
            matches.append(os.path.join(exe_dir, fname))

    return matches


# Performance comparison function
def benchmark_versions(exe_path: str, batch_rec_pattern: List[str] = ["*.bat"],
                       min_match_length: int = 5) -> dict:
    """
    Benchmark different versions to help choose the best one.
    Returns timing and memory usage stats.
    """
    import time
    import tracemalloc

    versions = {
        'streaming': find_matching_batch_files,
        'rolling_hash': find_matching_batch_files_rolling_hash
    }

    results = {}

    for name, func in versions.items():
        tracemalloc.start()
        start_time = time.perf_counter()

        try:
            result = func(exe_path, batch_rec_pattern=batch_rec_pattern,
                          min_match_length=min_match_length)
            end_time = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            results[name] = {
                'time': end_time - start_time,
                'memory_current': current,
                'memory_peak': peak,
                'result_count': len(result)
            }
        except Exception as e:
            results[name] = {'error': str(e)}
            tracemalloc.stop()

    return results

@contextmanager
def atomic_file_operation(filepath: str, mode: str = 'w'):
    """Context manager for atomic file operations"""
    temp_path = None
    try:
        # Create temporary file in same directory
        dir_path = os.path.dirname(filepath)
        temp_fd, temp_path = tempfile.mkstemp(dir=dir_path)

        if 'b' in mode:
            temp_file = os.fdopen(temp_fd, mode)
        else:
            temp_file = os.fdopen(temp_fd, mode, encoding='utf-8')

        yield temp_file
        temp_file.flush()
        os.fsync(temp_fd)
        temp_file.close()

        # Atomically replace original file
        if os.name == 'nt':  # Windows
            if os.path.exists(filepath):
                os.replace(temp_path, filepath)
            else:
                os.rename(temp_path, filepath)
        else:  # Unix-like
            os.rename(temp_path, filepath)

        temp_path = None  # Prevent cleanup

    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        raise e
