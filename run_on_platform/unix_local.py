import os
import subprocess
import shutil
import logging

logger = logging.getLogger(__name__)

def start_application(app_path):
    try:
        subprocess.Popen([app_path])
        logger.info(f"Started application: {app_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to start application {app_path}: {e}")
        return False

def kill_application(app_name):
    try:
        subprocess.run(['pkill', '-f', app_name], check=True)
        logger.info(f"Killed application: {app_name}")
        return True
    except subprocess.CalledProcessError:
        logger.warning(f"Application {app_name} not running")
        return True
    except Exception as e:
        logger.error(f"Failed to kill application {app_name}: {e}")
        return False

def copy_file(src, dst):
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        logger.info(f"Copied file from {src} to {dst}")
        return True
    except Exception as e:
        logger.error(f"Failed to copy file from {src} to {dst}: {e}")
        return False

def delete_file(path):
    try:
        if os.path.isfile(path):
            os.remove(path)
            logger.info(f"Deleted file: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            logger.info(f"Deleted directory: {path}")
        else:
            logger.warning(f"Path not found: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete {path}: {e}")
        return False
