import subprocess
import logging

logger = logging.getLogger(__name__)

def run_remote_command(host, user, command):
    ssh_command = ['ssh', f'{user}@{host}', command]
    try:
        result = subprocess.run(ssh_command, capture_output=True, text=True, check=True)
        logger.info(f"Executed remote command on {host}: {command}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Remote command failed on {host}: {e.stderr}")
        return None

def start_application(host, user, app_path):
    return run_remote_command(host, user, f'nohup {app_path} &')

def kill_application(host, user, app_name):
    return run_remote_command(host, user, f'pkill -f {app_name}')

def copy_file_to_remote(src, host, user, dst):
    scp_command = ['scp', src, f'{user}@{host}:{dst}']
    try:
        subprocess.run(scp_command, check=True)
        logger.info(f"Copied file to remote {host}:{dst}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to copy file to remote {host}:{dst}: {e}")
        return False

def delete_file(host, user, path):
    return run_remote_command(host, user, f'rm -rf {path}')
