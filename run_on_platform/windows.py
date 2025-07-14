"""
Windows-specific implementation of platform operations.
"""

import os
import subprocess
import shutil
import psutil
import csv
from io import StringIO
from typing import List, Dict, Union, Tuple

def get_running_processes_windows() -> List[str]:
    try:
        output = subprocess.check_output(['tasklist', '/FO', 'CSV'], text=True)
        reader = csv.DictReader(StringIO(output))
        return [row['Image Name'] for row in reader if row['Image Name'].lower().endswith('.exe')]
    except Exception:
        return []

def kill_process_by_name_windows(app_name: str, logger=None) -> bool:
    try:
        subprocess.run(['taskkill', '/f', '/im', app_name], check=True, capture_output=True)
        if logger:
            logger.info(f"Killed process: {app_name}")
        return True
    except subprocess.CalledProcessError:
        if logger:
            logger.warning(f"Process {app_name} not found or could not be killed.")
        return False
    except Exception as e:
        if logger:
            logger.error(f"Error killing process {app_name}: {e}")
        return False

def start_application_windows(app_path: str, logger=None) -> bool:
    try:
        ext = os.path.splitext(app_path)[1].lower()
        if ext in ['.bat', '.cmd']:
            subprocess.Popen(['cmd.exe', '/c', app_path], shell=True)
        else:
            subprocess.Popen([app_path], creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
        if logger:
            logger.info(f"Started application: {app_path}")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Error starting application {app_path}: {e}")
        return False

def copy_file_with_integrity(src: str, dst: str, logger=None) -> bool:
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        if logger:
            logger.debug(f"Copied file from {src} to {dst}")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Failed to copy file from {src} to {dst}: {e}")
        return False

def delete_file_or_dir(path: str, logger=None) -> bool:
    try:
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
        if logger:
            logger.info(f"Deleted: {path}")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Failed to delete {path}: {e}")
        return False
