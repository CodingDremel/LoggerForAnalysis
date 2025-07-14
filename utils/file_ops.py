import os
import shutil
import subprocess
from pathlib import Path
import tempfile
from typing import Optional
from contextlib import contextmanager

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
