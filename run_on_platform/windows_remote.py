
import pywinrm
import os
from typing import List, Tuple

class RemoteWindowsOperator:
    def __init__(self, ip: str, username: str, password: str):
        self.session = pywinrm.Session(ip, auth=(username, password))

    def run_cmd(self, command: str) -> Tuple[int, str, str]:
        try:
            result = self.session.run_cmd(command)
            return result.status_code, result.std_out.decode(), result.std_err.decode()
        except Exception as e:
            return -1, "", str(e)

    def start_application(self, app_path: str) -> Tuple[int, str, str]:
        return self.run_cmd(f'start "" "{app_path}"')

    def kill_application(self, app_name: str) -> Tuple[int, str, str]:
        return self.run_cmd(f'taskkill /f /im {app_name}')

    def delete_file(self, file_path: str) -> Tuple[int, str, str]:
        return self.run_cmd(f'del /f /q "{file_path}"')

    def delete_directory(self, dir_path: str) -> Tuple[int, str, str]:
        return self.run_cmd(f'rmdir /s /q "{dir_path}"')

    def copy_file(self, src: str, dst: str) -> Tuple[int, str, str]:
        return self.run_cmd(f'copy "{src}" "{dst}"')

    def list_processes(self) -> Tuple[int, str, str]:
        return self.run_cmd('tasklist')
