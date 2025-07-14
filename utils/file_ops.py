import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

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
