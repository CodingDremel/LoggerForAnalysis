import json
import copy
from typing import Dict, Any, List
from dataclasses import dataclass

def load_default_config():
    """Load default config values"""
    return {
        'max_file_size': 1000 * 1024 * 1024,
        'max_files_per_operation': 5000,
        'backup_before_delete': True,
        'verify_paths': True,
        'simulation_mode': False,
        'default_log_file_extensions': ['.txt', '.log', '.csv', '.\\d+', '.sum', '.tmp'],
        'safe_apps': [],
        'critical_apps': [],
        'destination': "C:\\Logs"
    }

class FlushSignal:
    pass

class ConfigParser:
    """Parse and validate config structure"""
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.raw_config = {}
        self.computers = {}
        self.operations = {}
        self.global_settings = {}
        self.defaults = {**load_default_config(), 'safe_apps': [], 'critical_apps': []}

    def load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as f:
                self.raw_config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {self.config_path} not found")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        self._parse_config()
        return self._build_legacy_format()

    def _parse_config(self):
        if 'computers' in self.raw_config and isinstance(self.raw_config['computers'], dict):
            self.computers = self.raw_config['computers']
        for key, value in self.raw_config.items():
            if key == 'computers':
                continue
            elif isinstance(value, list) and '_' in key:
                self.operations[key] = value
            else:
                self.global_settings[key] = value

    def _build_legacy_format(self) -> Dict[str, Any]:
        legacy_config = copy.deepcopy(self.defaults)
        legacy_config.update(self.global_settings)
        computers_dict = {}
        for comp_name, comp_config in self.computers.items():
            computer = {
                'ip': comp_config.get('ip', '127.0.0.1'),
                'user_name': comp_config.get('user_name', ''),
                'password': comp_config.get('password', ''),
                'destination_path': comp_config.get('destination_path', ''),
                'name': comp_config.get('name', comp_name),
                'exe_path': comp_config.get('exe_path', ''),
                'save_session_path': comp_config.get('save_session_path', ''),
                'save_process_list_path': comp_config.get('save_process_list_path', ''),
                'kill_all_path': comp_config.get('kill_all_path', ''),
                'log_src_path': comp_config.get('log_src_path', []),
                'files_to_delete': comp_config.get('files_to_delete', []),
                'files_to_copy': comp_config.get('files_to_copy', []),
                'files_path': comp_config.get('files_path', []),
                'applications_to_start': comp_config.get('applications_to_start', []),
                'applications_to_kill': comp_config.get('applications_to_kill', []),
                'safe_apps': comp_config.get("safe_apps", []),
                'critical_apps': comp_config.get("critical_apps", []),
                'apps': comp_config.get('apps', []),
                'common_app_order': comp_config.get('common_app_order', {}),
                'src_path': comp_config.get('src_path', ''),
                'target_path': comp_config.get('target_path', '') + comp_config.get('destination_path', ''),
                'pings_to_run': comp_config.get('pings_to_run', 1),
                'remote_drive_connection_attempts': comp_config.get('remote_drive_connection_attempts', 1),
                'remote_copy_attempts': comp_config.get('remote_copy_attempts', 1)
            }
            if 'LCS' in computer['name'].upper():
                computer['lcs_restart_delay'] = self.global_settings.get('lcs_restart_delay', 0)
            computers_dict[comp_name] = computer
        legacy_config['computers'] = computers_dict
        legacy_config['file_extensions'] = self.global_settings.get('file_extensions', self.defaults['default_log_file_extensions'])
        legacy_config['destination'] = self.raw_config.get('all_logs_save', self.defaults['destination'])
        return legacy_config

@dataclass
class OperationResult:
    """Data class to track operation results"""
    operation: str
    computer_name: str
    success_count: int
    failure_count: int
    errors: List[str]
    duration: float
    timestamp: str
