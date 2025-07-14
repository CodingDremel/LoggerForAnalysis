# General imports
import os
import json
import hashlib
import zlib
import logging
import logging.handlers
import colorlog
import argparse
import multiprocessing
import threading
import time
import shutil
import fcntl
import tempfile
import queue
import subprocess
import psutil
import copy
import re

# Concrete imports
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# Local imports
from core.logger import SafeLogger
from config.parser import load_default_config
from run_on_platform.base import OperationResult

class SafeRemoteOperations:
    """Safe version of remote operations with proper atomicity and logging"""

    def __init__(self, config: Dict[str, Any], logger=None, simulation_mode: bool = False):
        self.config = config
        self.simulation_mode = simulation_mode
        self.file_extensions = config.get("file_extensions", config.get('default_log_file_extensions', ['.txt']))

        self.safe_logger = logger

        # Build safe_apps and critical_apps with computer-specific defaults
        self.safe_apps = self._build_safe_apps_list(config, self.safe_logger)
        self.critical_apps = self._build_critical_apps_list(config, self.safe_logger)

        # Use config values instead of global SAFE_MODE_CONFIG
        self.max_file_size = config.get('max_file_size', 1000 * 1024 * 1024)  # 1000 Mb
        self.max_files_per_operation = config.get('max_files_per_operation', 1000)
        self.backup_before_delete = config.get('backup_before_delete', True)

    def _build_safe_apps_list(self, config: Dict[str, Any], logger: SafeLogger) -> List[str]:
        """Build safe_apps list with fallback to computer-specific apps"""
        # If safe_apps is not explicitly defined in config, use default it
        joined_safe_apps = copy.deepcopy(config['safe_apps'])
        if not config['safe_apps'] or set(config['safe_apps']).issubset(set(load_default_config()['safe_apps'])):
            joined_safe_apps.extend(load_default_config()['safe_apps'])
            data_string = "\t" + "\n\t".join(joined_safe_apps)
            self.safe_logger.warning(
                f"Using default safe_apps from config: {len(joined_safe_apps)} apps\n{data_string}")
            return joined_safe_apps

        # Otherwise, build from default safe_apps + all computer apps
        default_safe_apps = load_default_config()['safe_apps']
        computer_apps = self._extract_all_computer_apps(config)

        # Combine and deduplicate
        combined_apps = list(set(default_safe_apps + computer_apps))

        self.safe_logger.debug(f"Computer apps found: {computer_apps}")
        self.safe_logger.debug(
            f"Built safe_apps list: {len(default_safe_apps)} default + {len(computer_apps)} "
            f"from computers = {len(combined_apps)} total\n\t" + "\n\t".join(combined_apps) + "\n")
        return combined_apps

    def _build_critical_apps_list(self, config: Dict[str, Any], logger: SafeLogger) -> List[str]:
        """Build critical_apps list with fallback to defaults only"""
        # If critical_apps is explicitly defined in config, use it
        if config['critical_apps'] == load_default_config()['critical_apps']:
            data_string = "\t" + "\n\t".join(config['critical_apps'])
            self.safe_logger.warning(
                f"Using explicit critical_apps from config: {len(config['critical_apps'])} apps\n{data_string}")
            return config['critical_apps']

        # For critical apps, we only use the default list (don't add computer apps)
        # as we don't want to accidentally mark user apps as critical
        default_critical_apps = load_default_config()['critical_apps']
        data_string = "\t" + "\n\t".join(default_critical_apps)

        self.safe_logger.debug(f"Using default critical_apps list: {len(default_critical_apps)} apps")
        self.safe_logger.debug(f"Built-in critical_apps list:\n{data_string}\n")
        return default_critical_apps

    def _extract_all_computer_apps(self, config: Dict[str, Any]) -> List[str]:
        """Extract all apps from all computer configurations"""
        all_apps = []
        computers = config.get('computers', {})

        # Handle both dict and list formats
        if isinstance(computers, dict):
            computer_configs = computers.values()
        elif isinstance(computers, list):
            computer_configs = computers
        else:
            return all_apps

        for computer in computer_configs:
            if not isinstance(computer, dict):
                continue

            # Extract apps from various fields
            apps_fields = ['apps', 'applications_to_start', 'applications_to_kill']

            for field in apps_fields:
                apps = computer.get(field, [])

                if isinstance(apps, str):
                    # Single app as string
                    app_name = self._extract_app_name(apps)
                    if app_name:
                        all_apps.append(app_name)
                elif isinstance(apps, list):
                    # List of apps
                    for app in apps:
                        if isinstance(app, str):
                            app_name = self._extract_app_name(app)
                            if app_name:
                                all_apps.append(app_name)

        # Remove duplicates and return
        return list(set(all_apps))

    def _extract_app_name(self, app_path: str) -> str:
        """Extract just the executable name from a full path"""
        if not app_path:
            return ""

        # Handle both Windows and Unix paths
        app_name = os.path.basename(app_path)

        # If it's just a name without path, return as-is
        if app_name == app_path:
            return app_name

        return app_name

    def _is_safe_application(self, app: str) -> bool:
        """Check if application is safe to start"""
        app_name = self._extract_app_name(app)
        # In case you want to change logic to allow list of safe apps - uncomment 2 lines below
        # return any(safe_app.lower() in app_name.lower() or app_name.lower() in safe_app.lower()
        #            for safe_app in self.safe_apps)
        return True

    def _is_safe_to_kill(self, app: str) -> bool:
        """Check if application is safe to kill"""
        app_name = self._extract_app_name(app)
        # In case you want to change logic to allow persistent apps - uncomment 2 lines below
        # return not any(critical_app.lower() in app_name.lower() or app_name.lower() in critical_app.lower()
        #                for critical_app in self.critical_apps)
        return True

    def get_all_files(self, path: str, filter=False) -> List[str]:
        """Safely get all files matching extensions"""
        files = []
        try:
            if not os.path.exists(path):
                self.safe_logger.warning(f"Path does not exist: {path}")
                return files
            pattern = '|'.join(self.file_extensions)

            if os.path.isdir(path):
                for root, _, filenames in os.walk(path):
                    for filename in filenames:
                        if filter:
                            if re.search(pattern, filename):
                                full_path = os.path.join(root, filename)
                        else:
                            full_path = os.path.join(root, filename)

                        # Safety checks using instance variables
                        if os.path.getsize(full_path) > self.max_file_size:
                            self.safe_logger.warning(f"Skipping large file: {full_path}")
                            continue

                        files.append(full_path)

                        if len(files) >= self.max_files_per_operation:
                            self.safe_logger.warning(f"Reached max files limit: {self.max_files_per_operation}")
                            break

            elif os.path.isfile(path):
                if os.path.getsize(path) <= self.max_file_size:
                    files.append(path)
                else:
                    self.safe_logger.warning(f"File path too large: {path}")

        except Exception as e:
            self.safe_logger.error(f"Error getting files from {path}: {str(e)}")

        return files

    def safe_delete_files(self, computer: Dict[str, Any]) -> OperationResult:
        """Safely delete files with backup option"""
        start_time = time.time()
        name = computer['name']
        ip = computer['ip']

        self.safe_logger.info(f"[{name} | {ip}] Starting safe_delete_files operation...")
        self.safe_logger.info(f"\n{'-' * 128}\n")

        success, fail = 0, 0
        errors = []

        # Handle LOG_SRC_PATH as dict, list fo dicts or str
        log_src_list = []
        if isinstance(computer['log_src_path'], dict):
            log_src_list = [val for key, val in computer['log_src_path'].items() if key == 'source']
        elif isinstance(computer['log_src_path'], list):
            log_src_list = [val['source'] for val in computer['log_src_path']]
        elif isinstance(computer['log_src_path'], str):
            log_src_list = [computer['log_src_path']]
        else:
            error_message = (f"Unknown format of [{name}]['log_src_path'] entry, not LIST, DICT or STR, "
                             f"value is: {computer['log_src_path']}, exiting delete_files operation...")
            errors.append(error_message)
            self.safe_logger.error(error_message)
            self.safe_logger.info(f"\n{'-' * 128}\n")

            return OperationResult(
                operation="delete_files",
                computer_name=name,
                success_count=0,
                failure_count=1,
                errors=errors,
                duration=0,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )

        lst_src_str = "\n\t".join(log_src_list)
        lst_fdl_str = "\n\t".join(computer['files_to_delete'])
        self.safe_logger.info(f"Creating a generalized list of files to delete from {computer['name']} "
                              f"values:\nlog_src_path:\n\t{lst_src_str if log_src_list else '[]'}\nAND"
                              f"\nfiles_to_delete: \n\t{lst_fdl_str if computer['files_to_delete'] else '[]'}\n...")

        all_files_to_delete_list = copy.deepcopy(computer['files_to_delete'])
        all_files_to_delete_list.extend(log_src_list)
        amount_of_files_to_delete = len(all_files_to_delete_list)

        # No files to delete in specifically mentioned dir - delete contents of log_src_path
        if amount_of_files_to_delete == 0:
            self.safe_logger.warning(f"Files_to_delete list and Log_src_path on: {name} are both empty, so nothing "
                                     f"to delete ...")
            if hasattr(self.safe_logger, 'flush'):
                self.safe_logger.flush()
        else:
            for i, path in enumerate(all_files_to_delete_list):
                # Validate path safety
                is_safe, safety_msg = SafePathValidator.is_safe_path(path)
                if not is_safe:
                    errors.append(f"Unsafe path {path}: {safety_msg}")
                    self.safe_logger.error(f"[{name}] {safety_msg}")
                    fail += 1
                    continue

                if self.simulation_mode:
                    self.safe_logger.info(f"[{name}] [SIMULATION] Would delete {path}")
                    success += 1
                    continue

                abs_path = os.path.abspath(path)
                if self.backup_before_delete:
                    main_backup_dir = self.config.get('destination', "") + '.backup'
                    os.makedirs(main_backup_dir, exist_ok=True)
                    backup_path = os.path.join(main_backup_dir, f"{abs_path}.bak")

                # Atomic deletion (move to temp, then delete)
                #         temp_path = file + '.tmp_delete'
                #         os.rename(file, temp_path)
                #         os.remove(temp_path)
                #
                #         self.safe_logger.debug(f"[{name}] Safely deleted {file}")
                #         success += 1

                if os.path.isfile(abs_path):
                    # Atomic file deletion
                    try:
                        if self.backup_before_delete:
                            shutil.copy2(abs_path, backup_path)
                            self.safe_logger.debug(f"[{name}] Created backup of FILE {abs_path} at: {backup_path}")
                        # Simple for debug
                        # os.remove(abs_path)
                        temp_path = abs_path + '.tmp_delete'
                        os.rename(abs_path, temp_path)
                        os.remove(temp_path)
                        self.safe_logger.info(f"[{abs_path}] Deleted file     : {abs_path}")
                        success += 1
                    except Exception as e:
                        self.safe_logger.error(f"Failed to delete     file    : {abs_path}")
                        fail += 1
                elif os.path.isdir(abs_path):
                    # Atomic directory deletion
                    try:
                        if self.backup_before_delete:
                            shutil.copytree(abs_path, backup_path)
                            self.safe_logger.debug(f"[{name}] Created backup of DIR {abs_path} at: {backup_path}")

                        parent_dir = os.path.dirname(path)
                        tmp_dir = tempfile.mkdtemp(dir=parent_dir)
                        tmp_path = os.path.join(tmp_dir, os.path.basename(path))

                        shutil.move(abs_path, tmp_path)
                        shutil.rmtree(tmp_path)
                        os.rmdir(tmp_dir)
                        self.safe_logger.info(f"Deleted DIR: {abs_path}")
                        # Simple for debug
                        # for item in os.listdir(abs_path):
                        #     item_path = os.path.join(abs_path, item)
                        #     if os.path.isfile(item_path) or os.path.islink(item_path):
                        #         os.remove(item_path)
                        #     elif os.path.isdir(item_path):
                        #         shutil.rmtree(item_path)
                        # self.safe_logger.info(f"Emptied given    directory     : {abs_path}")

                        success += 1
                    except Exception as e:
                        self.safe_logger.error(f"Failed to empty directory     : {abs_path}")
                        fail += 1
                else:
                    self.safe_logger.warning(f"Not file/dir or invalid path: {abs_path}")
                    fail += 1

                # Delete all files in directory supplied
                # files = self.get_all_files(path, filter=False)
                #
                # for file in files:
                #     try:
                #         if self.simulation_mode:
                #             self.safe_logger.info(f"[{name}] [SIMULATION] Would delete {file}")
                #             success += 1
                #             continue
                #
                #         # Create backup if configured
                #         if self.backup_before_delete:
                #             backup_dir = os.path.join(os.path.dirname(file), '.backup')
                #             os.makedirs(backup_dir, exist_ok=True)
                #             backup_path = os.path.join(backup_dir, f"{os.path.basename(file)}.bak")
                #             shutil.copy2(file, backup_path)
                #             self.safe_logger.debug(f"[{name}] Created backup: {backup_path}")
                #
                #         # Atomic deletion (move to temp, then delete)
                #         temp_path = file + '.tmp_delete'
                #         os.rename(file, temp_path)
                #         os.remove(temp_path)
                #
                #         self.safe_logger.debug(f"[{name}] Safely deleted {file}")
                #         success += 1
                #
                #     except Exception as e:
                #         error_msg = f"Failed to delete {file}: {str(e)}"
                #         errors.append(error_msg)
                #         self.safe_logger.error(f"[{name}] {error_msg}")
                #         fail += 1

        duration = time.time() - start_time
        self.safe_logger.info(f"[{name}] safe_delete_files operation completed: {success} succeeded, "
                              f"{fail} failed in {duration:.2f}s")
        self.safe_logger.info(f"\n{'-' * 128}\n")

        return OperationResult(
            operation="delete_files",
            computer_name=name,
            success_count=success,
            failure_count=fail,
            errors=errors,
            duration=duration,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )

    def safe_copy_files(self, computer: Dict[str, Any], copy_type=True) -> OperationResult:
        """Safely copy files with integrity verification
        copy_type - should we apply log filtering or not,
            if True - copy recursively origin structure, but only filtered files
            else - copy recursively origin structure and all files
        """
        start_time = time.time()
        name = computer['name']
        ip = computer['ip']

        self.safe_logger.info(f"[{name} | {ip}] Starting safe_copy_files")
        self.safe_logger.info(f"\n{'-' * 128}\n")

        success, fail = 0, 0
        errors = []
        list_of_paths_to_copy = computer.get("files_path")
        log_src_paths = computer.get("log_src_path")

        if isinstance(list_of_paths_to_copy, str):
            list_of_paths_to_copy = [list_of_paths_to_copy]

        all_paths_to_copy = copy.deepcopy(list_of_paths_to_copy)

        if not ((len(log_src_paths) == 1 and
                 (log_src_paths == [""] or log_src_paths == [{'source': "", 'destination': ""}]))):
            all_paths_to_copy += log_src_paths

        if not all_paths_to_copy:
            self.safe_logger.info(f"[{name}] safe_copy_files operation: List of all paths to copy is empty"
                                  f" - nothing to do...")
            duration = time.time() - start_time
            self.safe_logger.info(f"\n{'-' * 128}\n")

            return OperationResult(
                operation="copy_files",
                computer_name=name,
                success_count=success,
                failure_count=fail,
                errors=errors,
                duration=duration,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
        dir_name_prefix = "_LOG__" + datetime.datetime.fromtimestamp(start_time).strftime("%Y_%m_%d_%H_%M_%S")

        all_paths_to_copy = strip_common_prefix_sources(all_paths_to_copy)
        tabber = "\t" * 5 + "  "
        for entry in all_paths_to_copy:
            src = entry["source"]
            if not check_path_status(src):
                errors.append(f"Path:\t{src}\n{tabber}cannot be instantiated, accessed or doesn't exist")
                self.safe_logger.warning(f"Path:\t{src}\n{tabber}cannot be instantiated, accessed or doesn't exist")
                fail += 1
                continue
            mount_point_pattern = re.compile(r"^(?:[a-zA-Z]:\\?)$")
            if not mount_point_pattern.match(src):
                trimmed_source = entry["source"][2:] \
                    if os.path.isabs(entry["source"]) else os.path.abspath(entry["source"])[2:]
            else:
                trimmed_source = "\\Mount_Point__" + src[:1]

            dst = entry["destination"] + "\\" + computer['name'] + dir_name_prefix + trimmed_source
            Path(dst).mkdir(parents=True, exist_ok=True)

            # Validate paths
            for path in [src, dst]:
                is_safe, safety_msg = SafePathValidator.is_safe_path(path)
                if not is_safe:
                    errors.append(f"Unsafe path {path}: {safety_msg}")
                    self.safe_logger.error(f"[{name}] {safety_msg}")
                    fail += 1
                    continue

            # get files filtered or not - depending on supplied param
            files = self.get_all_files(src, filter=copy_type)

            for file in files:
                try:
                    if self.simulation_mode:
                        self.safe_logger.info(f"[{name}] [SIMULATION] Would copy {file} to {dst}")
                        success += 1
                        continue

                    # Ensure destination directory exists
                    dst_dir = os.path.dirname(dst) if os.path.isfile(dst) else dst
                    os.makedirs(dst_dir, exist_ok=True)

                    # Determine final destination path
                    if os.path.isdir(dst):
                        final_dst = os.path.join(dst, os.path.basename(file))
                    else:
                        final_dst = dst

                    # Atomic copy using temporary file
                    temp_dst = final_dst + '.tmp_copy'

                    # Copy with verification
                    with open(file, 'rb') as fsrc:
                        with open(temp_dst, 'wb') as fdst:
                            shutil.copyfileobj(fsrc, fdst)
                            fdst.flush()
                            os.fsync(fdst.fileno())

                    # Verify integrity before final move
                    if self._verify_file_integrity(file, temp_dst):
                        os.rename(temp_dst, final_dst)
                        self.safe_logger.debug(f"[{name}] Safely copied\n\t{file}\n\tto\n\t{final_dst}")
                        success += 1
                    else:
                        os.remove(temp_dst)
                        raise Exception("Integrity verification failed")

                except Exception as e:
                    error_msg = f"Failed to copy\n\t{file}\n\tto\n\t{dst}: {str(e)}"
                    errors.append(error_msg)
                    self.safe_logger.error(f"[{name}] {error_msg}")
                    fail += 1

        duration = time.time() - start_time
        self.safe_logger.info(f"[{name}] safe_copy_files completed: "
                              f"{success} succeeded, {fail} failed in {duration:.2f}s")
        self.safe_logger.info(f"\n{'-' * 128}\n")

        return OperationResult(
            operation="copy_files",
            computer_name=name,
            success_count=success,
            failure_count=fail,
            errors=errors,
            duration=duration,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )

    def safe_start_applications(self, computer: Dict[str, Any], start_mode: str = "ALL",
                                search_for_apps: bool = False) -> OperationResult:
        """Safely start applications with validation"""
        start_time = time.time()
        name = computer['name']
        ip = computer['ip']
        apps = computer.get("applications_to_start", [])

        self.safe_logger.info(f"\n{'-' * 128}\n")
        # self.safe_logger.info(f"[{name} | {ip}] Starting applications: {apps}")

        success, fail = 0, 0
        errors = []

        mode_keys = {
            "ALL": ["critical_apps", "apps", "applications_to_start"],
            "SAFE": ["apps", "applications_to_start"],
            "CRITICAL": ["critical_apps"]
        }
        if start_mode not in mode_keys:
            self.safe_logger.warning(f"[{name}] Invalid kill mode requested: {start_mode}, resetting it to \"SAFE\"")
            start_mode = "SAFE"
        self.safe_logger.info(f"Using following keys to search for applications to start: {mode_keys[start_mode]}")
        selected_keys = mode_keys.get(start_mode, [])
        apps = []
        # Process computer instance keys to search for values in them according to ones in selected_keys
        for sel_key in selected_keys:
            value = computer.get(sel_key, [])
            if isinstance(value, str):
                apps.append(value)
            elif isinstance(value, list):
                apps.extend(value)
        # Remove dupes and paths reducing to only filenames
        apps_print = [os.path.basename(p).lower() for p in list(set(apps))]
        apps_to_str = "\n\t".join(apps_print)
        positions = len(str(len(apps_print)))
        self.safe_logger.info(f"[{name} | {ip}] Starting applications:\n\t{apps_to_str}")

        def find_executable_abs_paths(filenames: List[str], errors_list: List[str] = [],
                                      parent_dir: Union[str, None] = None,
                                      logger=None) -> Union[List[str], Dict[str, List[str]]]:
            """
            Search for all abs paths to given exe names.
            If parent_dir is not given - search all mounted drives
            Else - limit search to parent dir
            Return:
                List of abs paths to each filename supplied if ALL have only one instance found
                If ANY got 0 or more than 1 - return empty list and print error message
            :param logger:
            :param errors_list:
            :param filenames:
            :param parent_dir:
            :return:
            """
            # print only errors generated here
            error_list_position_start = len(errors_list) if errors_list else 0
            error_list_position_end = error_list_position_start

            matches: Dict[str, List[str]] = {os.path.basename(name).lower(): [] for name in filenames}
            search_roots = []
            if parent_dir:
                search_roots = [parent_dir]
            else:
                if os.name == 'nt':
                    # Win System
                    search_roots = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
                    search_roots.append("C:\\Windows\\System32")
                else:
                    # Unix system
                    search_roots = ['/', '/usr', '/bin', '/opt', '/sbin']
            for root in search_roots:
                for dirpath, _, files in os.walk(root, topdown=True, followlinks=False):
                    # if logger:
                    #     logger.debug(f"Searching in: {dirpath}")
                    try:
                        for file in files:
                            for target in matches:
                                if file.lower() == target:
                                    full_path = os.path.join(dirpath, file)
                                    matches[target].append(full_path)
                    except (PermissionError, OSError) as e:
                        msg = f"Permission denied: {dirpath} ({e})"
                        if logger:
                            logger.warning(msg)
                            errors_list.append(msg)
                            error_list_position_end += 1
                            continue
            invalid = {k: v for k, v in matches.items() if len(v) != 1}
            if invalid:
                div = "\n\t\t"
                longest_fname = max([len(exe) for exe in invalid.keys()])
                for exe, paths in invalid.items():
                    print_paths = f"{div}".join(paths)
                    msg = (f"{exe:{longest_fname}}: {len(paths)} matches found." +
                           f"{div + (print_paths if len(paths) > 1 else '[]')}")
                    # if logger:
                    #     logger.error(msg)
                    errors_list.append(msg)
                    error_list_position_end += 1

            if not invalid:
                logger.info("\u2705 - Found all abs paths to all exe files we looked for")
                abs_paths = [paths[0] for paths in matches.values()]
                logger.info("\n\t".join(abs_paths))
                return abs_paths
            else:
                error_msg_details = errors_list[error_list_position_start: error_list_position_end]
                print_error_msg_details = "\n\t".join(error_msg_details)
                if logger:
                    logger.error(f"\u274C - Following errors found:\n\t{print_error_msg_details}")
                return []

        def spinner_wrapper(func, *args, **kwargs):
            """
            Wrap a function with spinner if it takes more than 2s to complete

            :param func:
            :param args:
            :param kwargs:
            :return:
            """
            result = [None]
            done = threading.Event()

            def spinner():
                for c in itertools.cycle('|/-\\'):
                    if done.is_set():
                        break
                    print(f'\r\u23F3 Searching...{c}', end='', flush=True)
                    time.sleep(0.1)

            def target():
                result[0] = func(*args, **kwargs)
                done.set()

            thread = threading.Thread(target=target)
            thread.start()

            thread.join(timeout=2)
            if not done.is_set():
                spin_thread = threading.Thread(target=spinner)
                spin_thread.start()
                thread.join()
                done.set()
                spin_thread.join()
                print("\r\u2705 - Search finished.     ")
            else:
                print("\r\u2705 - Search completed in less than 2s")
            return result[0]

        # apps = ['NotePad++.exe']
        if not all(os.path.isabs(app) for app in apps):
            if search_for_apps:
                paths = spinner_wrapper(find_executable_abs_paths, apps, errors_list=errors,
                                        logger=self.safe_logger, parent_dir=computer['exe_path'])
                if not paths:
                    duration = time.time() - start_time
                    return OperationResult(
                        operation="start_applications",
                        computer_name=name,
                        success_count=0,
                        failure_count=1,
                        errors=errors,
                        duration=duration,
                        timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
                    )
            else:
                app_names_not_abs = [app for app in apps if not os.path.isabs(app)]
                err_msg = (f"Unable to continue, following apps are not supplied as ABSOLUTE paths, "
                           f"and settings prevent from searching for them...\n\t") + "\n\t".join(app_names_not_abs)
                errors.append(err_msg)
                duration = time.time() - start_time
                self.safe_logger.error(err_msg)
                return OperationResult(
                    operation="start_applications",
                    computer_name=name,
                    success_count=0,
                    failure_count=1,
                    errors=errors,
                    duration=duration,
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
                )
        paths = apps

        for i, app in enumerate(apps, 1):
            try:
                if self.simulation_mode:
                    self.safe_logger.info(f"[{name}] [SIMULATION] Would start {i}. {app}")
                    success += 1
                    continue

                # Validate application path/name
                if not self._is_safe_application(paths[i - 1]):
                    raise Exception(f"Application {app} is not in safe list")

                # Use subprocess for better control
                if os.name == 'nt':  # Windows
                    ext = os.path.splitext(paths[i - 1])[1].lower()
                    if ext in [".bat", ".cmd"]:
                        subprocess.Popen(['start', paths[i - 1]], shell=True)
                    if (ext == ".exe" and
                            any(os.path.basename(paths[i - 1]).lower().startswith(a) for a in ('alwaysrun', 'always'))):
                        base_dir = os.path.dirname(paths[i - 1])
                        bat_files_list = [os.path.join(base_dir, f) for f in os.listdir(base_dir)
                                          if f.lower().endswith('.bat')]
                        if bat_files_list:
                            bats_in_dir = find_matching_bat_files(paths[i - 1], 3)
                            self.safe_logger.warning(f"For persistent app:\n\t{paths[i - 1]}\n"
                                                     f"running it using:\n\t{bats_in_dir[0]}")
                            # os.path.join(base_dir, bats_in_dir[0])
                            subprocess.Popen(['cmd.exe', '/c', bats_in_dir[0]], cwd=base_dir,
                                             creationflags=subprocess.CREATE_NO_WINDOW)
                        else:
                            self.safe_logger.info(f"No batches found, running app directly:\n\t{paths[i - 1]}")
                            subprocess.Popen([paths[i - 1]],
                                             creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
                    else:

                        subprocess.Popen([paths[i - 1]],
                                         creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
                else:  # Unix-like
                    subprocess.Popen([paths[i - 1]])

                self.safe_logger.debug(f"[{name}] Started {i}. {app}")
                success += 1


            except Exception as e:
                error_msg = f"Failed to start {i}. {app}: {str(e)}"
                errors.append(error_msg)
                self.safe_logger.error(f"[{name}] {error_msg}")
                fail += 1

        duration = time.time() - start_time

        verification, error_list = verify_apps_running(computer=computer,
                                                       list_of_keys_to_test=selected_keys, logger=self.safe_logger)
        if verification:
            self.safe_logger.info(
                f"[{name}] safe_start_applications completed: {success} succeeded, {fail} failed in {duration:.2f}s")
        else:
            print_err = "".join(error_list)
            self.safe_logger.error(f"Following applications failed to start:\n{print_err}")

        self.safe_logger.info(f"\n{'-' * 128}\n")

        return OperationResult(
            operation="start_applications",
            computer_name=name,
            success_count=success,
            failure_count=fail,
            errors=errors,
            duration=duration,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )

    def start_apps(self, computer: Dict[str, Any]) -> OperationResult:
        """Start applications with process checking and logging"""
        start_time = time.time()
        name = computer['name']
        ip = computer['ip']

        self.safe_logger.info(f"\n{'-' * 128}\n")
        self.safe_logger.info(f"[{name} | {ip}] Starting start_apps operation")

        success, fail = 0, 0
        errors = []

        # Step 1: Get running processes and save/store them
        try:
            running_processes = self._get_running_processes()

            # Save to file if save_process_list_path exists and is not empty
            save_path = computer.get("save_process_list_path", "")
            if save_path and save_path.strip():
                try:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, 'w') as f:
                        for process in running_processes:
                            f.write(f"{process}\n")
                    self.safe_logger.info(f"[{name}] Process list saved to {save_path}")
                except Exception as e:
                    self.safe_logger.warning(f"[{name}] Could not save process list to {save_path}: {str(e)}")
            else:
                self.safe_logger.info(f"[{name}] Process list kept in memory ({len(running_processes)} processes)")

        except Exception as e:
            error_msg = f"Failed to get running processes: {str(e)}"
            errors.append(error_msg)
            self.safe_logger.error(f"[{name}] {error_msg}")
            fail += 1
            running_processes = []

        # Step 2: Process apps from config
        apps = computer.get("apps", [])
        if isinstance(apps, str):
            apps = [apps]  # Convert single app to list

        if not apps:
            self.safe_logger.info(f"[{name}] No apps configured to start")
            self.safe_logger.info(f"\n{'-' * 128}\n")

            return OperationResult(
                operation="start_apps",
                computer_name=name,
                success_count=success,
                failure_count=fail,
                errors=errors,
                duration=time.time() - start_time,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )

        for app in apps:
            try:
                app_name = os.path.basename(app) if '\\' in app or '/' in app else app

                # Check if app is already running
                if self._is_process_running(app_name, running_processes):
                    self.safe_logger.info(f"[{name}] App {app_name} is already running")
                    success += 1
                    continue

                # Step 2.1: Start the app
                if self.simulation_mode:
                    self.safe_logger.info(f"[{name}] [SIMULATION] Would start {app}")
                    success += 1
                    continue

                # Get full exe_path if available
                exe_path = computer.get("exe_path", "")
                if exe_path and not os.path.isabs(app):
                    full_app_path = os.path.join(exe_path, app)
                else:
                    full_app_path = app

                # Start the application
                if os.name == 'nt':  # Windows
                    subprocess.Popen([full_app_path], shell=True)
                else:  # Unix-like
                    subprocess.Popen([full_app_path])

                self.safe_logger.info(f"[{name}] Successfully started {app}")
                success += 1

            except Exception as e:
                error_msg = f"Failed to start {app}: {str(e)}"
                errors.append(error_msg)
                self.safe_logger.error(f"[{name}] {error_msg}")
                fail += 1
                # Continue with next app instead of raising exception

        duration = time.time() - start_time
        self.safe_logger.info(f"[{name}] start_apps completed: {success} succeeded, {fail} failed in {duration:.2f}s")
        self.safe_logger.info(f"\n{'-' * 128}\n")

        return OperationResult(
            operation="start_apps",
            computer_name=name,
            success_count=success,
            failure_count=fail,
            errors=errors,
            duration=duration,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )

    def safe_kill_applications(self, computer: Dict[str, Any], kill_mode: str = "SAFE") -> OperationResult:
        """Safely kill applications with validation"""
        start_time = time.time()
        name = computer['name']
        ip = computer['ip']

        mode_keys = {
            "ALL": ["critical_apps", "safe_apps", "apps", "applications_to_kill"],
            "SAFE": ["safe_apps", "apps"],
            "CRITICAL": ["critical_apps", "safe_apps", "apps"],
            "KILL": ["safe_apps", "applications_to_kill", "apps"]
        }
        if kill_mode not in mode_keys:
            self.safe_logger.warning(f"[{name}] Invalid kill mode requested: {kill_mode}, resetting it to \"SAFE\"")
            kill_mode = "SAFE"

        self.safe_logger.info(f"Using following keys to search for applications to kill: {mode_keys[kill_mode]}")
        selected_keys = mode_keys.get(kill_mode, [])
        apps = []
        # Process computer instance keys to search for values in them according to ones in selected_keys
        for sel_key in selected_keys:
            value = computer.get(sel_key, [])
            if isinstance(value, str):
                apps.append(value)
            elif isinstance(value, list):
                apps.extend(value)
        # Remove dupes and paths reducing to only filenames
        apps = [os.path.basename(p).lower() for p in list(set(apps))]

        # Simpler DEBUG version to get all
        # apps = (computer.get("applications_to_kill", []) + computer.get("apps", []) +
        #         computer.get("safe_apps", []) + computer.get("critical_apps", []))
        apps_to_str = "\n\t".join(apps)
        positions = len(str(len(apps)))
        self.safe_logger.info(f"[{name} | {ip}] Killing applications:\n\t{apps_to_str}")
        self.safe_logger.info(f"\n{'-' * 128}\n")

        success, fail = 0, 0
        errors = []

        def kill_win_processes_by_name_substring(substring: str, logger=None):
            """Kills windows processes by part of name or name in case-insensitive way"""
            sub = substring.lower()
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_name = proc.info['name']
                    if proc_name and sub in proc_name.lower():
                        logger.info(f"PsUtil Killing: {proc_name} (PID: {proc.pid})")
                        proc.kill()
                        logger.info(f"PsUtil Killed: {proc_name} (PID: {proc.pid})")
                except psutil.NoSuchProcess:
                    logger.warning(f"Process: {proc_name} not found...")
                    continue
                except psutil.AccessDenied:
                    logger.warning(f"Process: {proc_name} got Access Denied error...")
                    continue

        for i, app in enumerate(apps, 1):
            try:
                if self.simulation_mode:
                    self.safe_logger.info(f"[{name}] [SIMULATION] Would kill app no.{i:{positions}}: {app}")
                    success += 1
                    continue

                # Validate application is safe to kill
                if not self._is_safe_to_kill(app):
                    raise Exception(f"Application {app} is critical and cannot be killed")

                # Use cross-platform process killing
                import subprocess
                if os.name == 'nt':  # Windows
                    # TODO: choose method - psutil or taskkill, make sure to execute taskkill with ADMIN privileges
                    subprocess.run(['taskkill', '/f', '/im', app], check=True, capture_output=True)
                    # kill_win_processes_by_name_substring(app, self.safe_logger)
                else:  # Unix-like
                    subprocess.run(['pkill', '-f', app], check=True, capture_output=True)

                self.safe_logger.debug(f"[{name}] Killed {i}. {app}")
                success += 1

            except subprocess.CalledProcessError:
                # Process might not be running, consider it success
                self.safe_logger.debug(f"[{name}] Process {app} was not running")
                success += 1
            except Exception as e:
                error_msg = f"Failed to kill {i}. {app}: {str(e)}"
                errors.append(error_msg)
                self.safe_logger.error(f"[{name}] {error_msg}")
                fail += 1

        duration = time.time() - start_time
        self.safe_logger.info(f"[{name}] safe_kill_applications completed: {success} succeeded, "
                              f"{fail} failed in {duration:.2f}s")
        self.safe_logger.info(f"\n{'-' * 128}\n")

        return OperationResult(
            operation="kill_applications",
            computer_name=name,
            success_count=success,
            failure_count=fail,
            errors=errors,
            duration=duration,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )

    def safe_restart_application(self, computer: Dict[str, Any]) -> OperationResult:
        """Safely restart applications by killing running ones and starting them again"""
        start_time = time.time()
        name = computer['name']
        ip = computer['ip']

        self.safe_logger.info(f"[{name} | {ip}] Starting safe_restart_application")

        total_success, total_fail = 0, 0
        all_errors = []

        # Step 1: Get apps from config (support both 'apps' and 'applications_to_start')
        apps_to_restart = []

        # Check 'apps' field (from config structure)
        apps_config = computer.get("apps", [])
        if isinstance(apps_config, str):
            apps_to_restart.append(apps_config)
        elif isinstance(apps_config, list):
            apps_to_restart.extend(apps_config)

        # Also check 'applications_to_start' field
        apps_start_config = computer.get("applications_to_start", [])
        if isinstance(apps_start_config, list):
            apps_to_restart.extend(apps_start_config)

        if not apps_to_restart:
            self.safe_logger.info(f"[{name}] No applications configured for restart")
            return OperationResult(
                operation="restart_application",
                computer_name=name,
                success_count=0,
                failure_count=0,
                errors=[],
                duration=time.time() - start_time,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )

        # Step 2: Check if any apps are running and kill them if they are
        try:
            running_processes = self._get_running_processes()
            apps_to_kill = []

            for app in apps_to_restart:
                app_name = os.path.basename(app) if ('\\' in app or '/' in app) else app
                if self._is_process_running(app_name, running_processes):
                    apps_to_kill.append(app)
                    self.safe_logger.info(f"[{name}] Found running app to restart: {app_name}")

            # Kill running applications if any found
            if apps_to_kill:
                # Create temporary computer config for killing
                temp_computer_kill = computer.copy()
                temp_computer_kill["applications_to_kill"] = apps_to_kill

                kill_result = self.safe_kill_applications(temp_computer_kill)
                total_success += kill_result.success_count
                total_fail += kill_result.failure_count
                all_errors.extend(kill_result.errors)

                # Wait a moment for processes to fully terminate
                if not self.simulation_mode and apps_to_kill:
                    time.sleep(2)
                    self.safe_logger.debug(f"[{name}] Waited 2 seconds for processes to terminate")
            else:
                self.safe_logger.info(f"[{name}] No running applications found to kill")

        except Exception as e:
            error_msg = f"Error during kill phase: {str(e)}"
            all_errors.append(error_msg)
            self.safe_logger.error(f"[{name}] {error_msg}")
            total_fail += 1

        # Step 3: Start all applications
        try:
            # Create temporary computer config for starting
            temp_computer_start = computer.copy()
            temp_computer_start["applications_to_start"] = apps_to_restart

            start_result = self.safe_start_applications(temp_computer_start)
            total_success += start_result.success_count
            total_fail += start_result.failure_count
            all_errors.extend(start_result.errors)

        except Exception as e:
            error_msg = f"Error during start phase: {str(e)}"
            all_errors.append(error_msg)
            self.safe_logger.error(f"[{name}] {error_msg}")
            total_fail += 1

        duration = time.time() - start_time
        self.safe_logger.info(f"[{name}] safe_restart_application completed: {total_success} succeeded, "
                              f"{total_fail} failed in {duration:.2f}s")
        self.safe_logger.info(f"\n{'-' * 128}\n")
        return OperationResult(
            operation="restart_application",
            computer_name=name,
            success_count=total_success,
            failure_count=total_fail,
            errors=all_errors,
            duration=duration,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )

    def _verify_file_integrity(self, src_path: str, dst_path: str) -> bool:
        """Verify file integrity using multiple methods"""
        try:
            with open(src_path, 'rb') as fsrc, open(dst_path, 'rb') as fdst:
                src_data = fsrc.read()
                dst_data = fdst.read()
                accumulated_error = ""

                # Size check
                if len(src_data) != len(dst_data):
                    accumulated_error += f"\nSize mismatch:\n\t{src_path}\n\tvs\n\t{dst_path}"

                # MD5 check
                if hashlib.md5(src_data).hexdigest() != hashlib.md5(dst_data).hexdigest():
                    accumulated_error += f"\nMD5 mismatch:\n\t{src_path}\n\tvs\n\t{dst_path}"

                # CRC32 check
                if zlib.crc32(src_data) != zlib.crc32(dst_data):
                    accumulated_error += f"\nCRC32 mismatch:\n\t{src_path}\n\tvs\n\t{dst_path}"

                if accumulated_error != "":
                    self.safe_logger.error(accumulated_error)
                    return False

                return True

        except Exception as e:
            self.safe_logger.error(f"Integrity verification error: {str(e)}")
            return False

    def _is_safe_application(self, app: str) -> bool:
        """Check if application is safe to start"""
        return True
        # return any(safe_app in app.lower() for safe_app in self.safe_apps)

    def _is_safe_to_kill(self, app: str) -> bool:
        """Check if application is safe to kill"""
        return True
        # return not any(critical_app in app.lower() for critical_app in self.critical_apps)

    def _get_running_processes(self) -> List[str]:
        """Get list of currently running processes"""
        processes = []
        try:
            if os.name == 'nt':  # Windows
                result = subprocess.run(['tasklist', '/fo', 'csv'],
                                        capture_output=True, text=True, check=True)
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        process_name = line.split(',')[0].strip('"')
                        processes.append(process_name)
            else:  # Unix-like
                result = subprocess.run(['ps', '-eo', 'comm'],
                                        capture_output=True, text=True, check=True)
                processes = [line.strip() for line in result.stdout.strip().split('\n')[1:]]

        except Exception as e:
            self.safe_logger.warning(f"Could not get process list via system command: {str(e)}")
            # Fallback to psutil if available
            try:
                for proc in psutil.process_iter(['name']):
                    processes.append(proc.info['name'])
            except Exception as e2:
                self.safe_logger.error(f"Could not get process list via psutil: {str(e2)}")

        return processes

    def _is_process_running(self, app_name: str, running_processes: List[str]) -> bool:
        """Check if a process is currently running"""
        return any(app_name.lower() in process.lower() for process in running_processes)


