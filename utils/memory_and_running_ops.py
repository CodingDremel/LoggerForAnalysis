# General imports
import logging
import os
import csv
import platform
import subprocess
import signal
import time
import weakref
import psutil
import subprocess
# import sys
# import json
# import hashlib
# import zlib
# import logging
# import logging.handlers#
# import colorlog
# import argparse
# import multiprocessing
# import threading
# import time
# import shutil
# import fcntl
# import tempfile
# import queue
# import copy
# import re
# import datetime
# import string
# import itertools

# Concrete imports
# from pathlib import Path
# from contextlib import contextmanager
# from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Union, Any, Optional
from io import StringIO
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import lru_cache
from contextlib import contextmanager
from pathlib import Path
from enum import Enum
from threading import RLock
from collections import defaultdict
from xml.etree import ElementTree as ET

# Local imports
from core.logger import SafeLogger
from operations import spinner_wrapper
# from config.parser import load_default_config
# from run_on_platform.base import OperationResult
# from utils.path_validator import SafePathValidator
# from utils.path_utils import check_path_status, strip_common_prefix_sources
# from file_ops import find_matching_batch_files

class OSType(Enum):
    """Operating system types for better type safety."""
    WINDOWS = "windows"
    LINUX = "linux"
    DARWIN = "darwin"
    UNIX = "unix"

class ProcessStatus(Enum):
    """Process status enumeration."""
    RUNNING = "running"
    STOPPED = "stopped"
    ZOMBIE = "zombie"
    SLEEPING = "sleeping"

# Connection timeouts and retry settings
DEFAULT_TIMEOUT = 30
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1


@dataclass
class AppVerificationResult:
    """Result of single application verification."""
    success: bool
    errors: List[str]
    app_name: str
    missing_dependencies: List[str]


@lru_cache(maxsize=1)
def get_current_os() -> OSType:
    """
    Get current operating system type with caching.

    Returns:
        OSType: The current operating system type
    """
    system = platform.system().lower()
    if system == 'windows':
        return OSType.WINDOWS
    elif system == 'linux':
        return OSType.LINUX
    elif system == 'darwin':
        return OSType.DARWIN
    else:
        return OSType.UNIX


@contextmanager
def timeout_context(seconds: int):
    """
    Context manager for handling timeouts in operations.

    seconds: Timeout duration in seconds

    Raises: TimeoutError: If operation exceeds timeout
    """

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def retry_on_failure(max_attempts: int = RETRY_ATTEMPTS, delay: float = RETRY_DELAY):
    """
    Decorator for retrying operations on failure.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
                    continue
            raise last_exception

        return wrapper

    return decorator


@dataclass
class ProcessInfo:
    """
    Standardized process information with enhanced validation and caching.

    Features:
    - Lazy property evaluation for expensive operations
    - Weak reference caching to prevent memory leaks
    - Cross-platform normalization of process data
    """
    name: str
    pid: int
    ppid: Optional[int] = None
    user: Optional[str] = None
    cpu_percent: Optional[float] = None
    memory_mb: Optional[float] = None
    command_line: Optional[str] = None
    status: Optional[str] = None
    start_time: Optional[str] = None
    _psutil_process: Optional[weakref.ref] = None  # Addition: Weak reference to psutil process

    def __post_init__(self):
        """Enhanced validation with better error messages and normalization."""
        if not self.name or not self.name.strip():
            raise ValueError(f"Process name cannot be empty (PID: {self.pid})")

        # EDIT: Allow PID 0 and 4 for known Windows system processes
        if self.pid <= 0 and self.name not in {"System Idle Process", "System"}:
            raise ValueError(f"Invalid PID: {self.pid} for process '{self.name}'")

        # Normalize process name with better path handling
        self.name = self._normalize_process_name(self.name)

        # Initialize psutil process reference for enhanced operations
        try:
            proc = psutil.Process(self.pid)
            self._psutil_process = weakref.ref(proc)
        except (psutil.NoSuchProcess, ImportError):
            self._psutil_process = None

    def _normalize_process_name(self, name: str) -> str:
        """
        Enhanced process name normalization with cross-platform support.

        Returns the executable name without path, handling both Unix and Windows paths.
        """
        # Handle both forward and backward slashes
        normalized = Path(name).name if ('/' in name or '\\' in name) else name
        return normalized.strip()

    @property
    def psutil_process(self) -> Optional[Any]:
        """
        Lazy access to psutil process object with automatic cleanup.

        Returns None if process no longer exists or psutil unavailable.
        """
        if self._psutil_process is None:
            return None

        proc = self._psutil_process()
        if proc is None:
            return None

        try:
            # Verify process still exists
            proc.is_running()
            return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    @property
    def memory_usage_detailed(self) -> Dict[str, float]:
        """
        NEW: Enhanced memory usage information using psutil.

        Returns detailed memory statistics if available.
        """
        proc = self.psutil_process
        if proc is None:
            return {}

        try:
            memory_info = proc.memory_info()
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'memory_percent': proc.memory_percent()
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {}

    @property
    def is_system_process(self) -> bool:
        """Enhanced system process detection with OS-specific patterns."""
        # NEW: More comprehensive system process detection
        system_processes = {
            'windows': {'System', 'Idle', 'csrss.exe', 'smss.exe', 'winlogon.exe', 'wininit.exe'},
            'linux': {'kthreadd', 'ksoftirqd', 'migration', 'systemd', 'init'},
            'darwin': {'kernel_task', 'launchd', 'kextd', 'UserEventAgent'}
        }

        os_type = get_current_os()
        os_specific = system_processes.get(os_type.value, set())

        return (self.name in os_specific or
                self.pid in {0, 4} or  # Common system PIDs
                self.name.startswith('[') and self.name.endswith(']'))  # Kernel threads


def _process_filtered_list(processes: List[ProcessInfo], os_type: OSType, filter_system: bool,
                          filter_name_list: List[str] = None) -> list:
    """
       Convert process list to set of processes with OS-specific filtering.

       Args:
           processes: List of ProcessInfo objects
           os_type: Operating system type
           filter_system: Whether to filter system processes
           filter_name_list: FileNames to filter out

       Returns:
           list: List of filtered processes
       """
    filtered_processes = processes

    if filter_system:
        filtered_processes = [p for p in processes if not p.is_system_process]
    if filter_name_list:
        filtered_processes = [p for p in processes if not p.name in filter_name_list]

    if os_type == OSType.WINDOWS:
        # For Windows, include only .exe files for consistency
        return [p for p in filtered_processes if p.name.lower().endswith('.exe')]
    else:
        # For Unix systems, use process name directly
        return [p for p in filtered_processes]


def _process_list_to_name_set(processes: List[ProcessInfo], os_type: OSType,
                              filter_system: bool) -> set:
    """
    Convert process list to name set with OS-specific filtering.

    Args:
        processes: List of ProcessInfo objects
        os_type: Operating system type
        filter_system: Whether to filter system processes

    Returns:
        set: Set of process names
    """
    filtered_processes = processes

    if filter_system:
        filtered_processes = [p for p in processes if not p.is_system_process]

    if os_type == OSType.WINDOWS:
        # For Windows, include only .exe files for consistency
        return {p.name for p in filtered_processes
                if p.name.lower().endswith('.exe')}
    else:
        # For Unix systems, use process name directly
        return {p.name for p in filtered_processes}


# Enhanced caching system with thread safety
class ProcessCache:
    """
    Thread-safe process cache with TTL and automatic cleanup.

    Features:
    - Thread-safe operations with RLock
    - Automatic cache expiration
    - Memory-efficient weak references
    - Statistics tracking for optimization
    """

    def __init__(self, ttl: int = 5):
        self._cache: Dict[str, Tuple[List[ProcessInfo], float]] = {}
        self._lock = RLock()
        self._ttl = ttl
        self._stats = {'hits': 0, 'misses': 0, 'evictions': 0}

    def get(self, key: str) -> Optional[List[ProcessInfo]]:
        """Thread-safe cache retrieval with automatic expiration."""
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None

            processes, timestamp = self._cache[key]
            if time.time() - timestamp > self._ttl:
                del self._cache[key]
                self._stats['evictions'] += 1
                self._stats['misses'] += 1
                return None

            self._stats['hits'] += 1
            return processes

    def put(self, key: str, processes: List[ProcessInfo]) -> None:
        """Thread-safe cache storage."""
        with self._lock:
            self._cache[key] = (processes, time.time())

    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self._stats = {'hits': 0, 'misses': 0, 'evictions': 0}


# Global cache instance for better reuse
_global_process_cache = ProcessCache()


# Enhanced ProcessCollector with better caching and error handling
class ProcessCollector(ABC):
    """
    Abstract base class with improved caching and error handling.

    Added Features:
    - Shared global cache for better memory efficiency
    - Enhanced retry logic with exponential backoff
    - Better error categorization and handling
    - Performance metrics collection
    """

    def __init__(self, remote_host: Optional[str] = None,
                 username: Optional[str] = None,
                 timeout: int = DEFAULT_TIMEOUT,
                 logger: Union[SafeLogger, str, None] = None):
        self.remote_host = remote_host
        self.username = username
        self.timeout = timeout
        self._cache_key = self._generate_cache_key()  # NEW: Unique cache key
        self._performance_metrics = {'collection_time': 0, 'process_count': 0}  # NEW: Metrics
        self._logger = None
        if isinstance(logger, SafeLogger):
            self._logger = logger
        elif isinstance(logger, str):
            self._logger = SafeLogger(name=logger, log_level=logging.DEBUG)

    def _generate_cache_key(self) -> str:
        """Generate unique cache key based on collector configuration."""
        components = [
            self.__class__.__name__,
            self.remote_host or 'local',
            self.username or 'default'
        ]
        return ':'.join(components)

    @abstractmethod
    def collect_processes(self) -> List[ProcessInfo]:
        """Abstract method to collect processes. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Abstract method to check if collector is available. Must be implemented by subclasses."""
        pass

    @retry_on_failure(max_attempts=3, delay=1.0)
    def collect_with_retry(self) -> List[ProcessInfo]:
        """
        ENHANCED: Collect processes with exponential backoff and metrics.

        NEW Features:
        - Performance timing
        - Better error categorization
        - Automatic cache management
        """
        start_time = time.time()

        try:
            # Check global cache first
            cached = _global_process_cache.get(self._cache_key)
            if cached:
                return cached

            # Collect fresh processes
            processes = self.collect_processes()

            # Update metrics
            self._performance_metrics['collection_time'] = time.time() - start_time
            self._performance_metrics['process_count'] = len(processes)

            # Cache results globally
            _global_process_cache.put(self._cache_key, processes)

            return processes

        except Exception as e:
            # NEW: Enhanced error handling with categorization
            error_type = type(e).__name__
            raise Exception(f"[{error_type}] {self.__class__.__name__}: {str(e)}")


def create_optimal_collector(os_type: OSType, remote_host: Optional[str] = None,
                             username: Optional[str] = None,
                             **kwargs) -> ProcessCollector:
    """
    Factory function to create the most optimal collector for given parameters.

    Added Features:
    - Intelligent collector selection based on system capabilities
    - Fallback mechanism with priority ordering
    - Enhanced configuration validation
    """
    # NEW: Capability-based collector selection
    if remote_host:
        # Remote collection - prefer SSH for Unix, WMI for Windows
        if os_type in [OSType.LINUX, OSType.DARWIN, OSType.UNIX]:
            return UnixPsCollector(remote_host, username, kwargs.get('ssh_key'))
        else:
            return WindowsWMICCollector(remote_host, username, kwargs.get('password'))

    # Local collection - prefer native tools
    if os_type == OSType.WINDOWS:
        # Try WMIC first (more detailed), fallback to tasklist
        # TODO: add Wevtutil if needed here
        if WindowsWMICCollector().is_available():
            return WindowsWMICCollector()
        return WindowsTasklistCollector()
    else:
        return UnixPsCollector()


def _get_collectors_for_os(os_type: OSType, remote_host: Optional[str],
                           username: Optional[str], password: Optional[str],
                           ssh_key: Optional[str]) -> List[ProcessCollector]:
    """
    Get appropriate collectors for the operating system.

    Args:
        os_type: Operating system type
        remote_host: Optional remote host
        username: Username for remote connection
        password: Password for remote connection
        ssh_key: SSH key for remote connection

    Returns:
        List[ProcessCollector]: Ordered list of collectors to try
    """
    collectors = []

    if os_type == OSType.WINDOWS:
        # Prefer tasklist for basic verification, WMIC for detailed info
        collectors.extend([
            WindowsTasklistCollector(remote_host, username, password),
            WindowsWMICCollector(remote_host, username, password)
        ])
    elif os_type in [OSType.LINUX, OSType.DARWIN, OSType.UNIX]:
        collectors.append(UnixPsCollector(remote_host, username, ssh_key))
    else:
        # Unknown OS - try all collectors
        collectors.extend([
            WindowsTasklistCollector(remote_host, username, password),
            WindowsWMICCollector(remote_host, username, password),
            UnixPsCollector(remote_host, username, ssh_key)
        ])

    return collectors


class WindowsTasklistCollector(ProcessCollector):
    """Windows tasklist command collector."""

    def __init__(self, remote_host: Optional[str] = None, username: Optional[str] = None,
                 password: Optional[str] = None):
        super().__init__(remote_host, username)
        self.password = password

    def is_available(self) -> bool:
        """Check if tasklist command is available."""
        try:
            result = subprocess.run(['tasklist', '/?'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def collect_processes(self) -> List[ProcessInfo]:
        """Collect processes using Windows tasklist command."""
        cmd = ['tasklist', '/FO', 'CSV']

        # Add remote connection parameters if specified
        if self.remote_host:
            cmd.extend(['/S', self.remote_host])
            if self.username:
                cmd.extend(['/U', self.username])
            if self.password:
                cmd.extend(['/P', self.password])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise Exception(f"Tasklist command failed: {result.stderr}")

            return self._parse_tasklist_csv(result.stdout)
        except subprocess.TimeoutExpired:
            raise Exception("Tasklist command timed out")
        except Exception as e:
            raise Exception(f"Error running tasklist: {e}")

    def _parse_tasklist_csv(self, csv_output: str) -> List[ProcessInfo]:
        """Parse tasklist CSV output into ProcessInfo objects."""
        processes = []
        try:
            reader = csv.DictReader(StringIO(csv_output))
            for row in reader:
                # Handle different column names in different Windows versions
                name = row.get('Image Name', row.get('ImageName', ''))
                pid_str = row.get('PID', '0')
                memory_str = row.get('Mem Usage', row.get('Memory Usage', '0 K'))
                status = row.get('Status', '')
                user = row.get('User Name', row.get('Username', ''))

                # Parse PID
                try:
                    pid = int(pid_str.replace(',', ''))
                except (ValueError, AttributeError):
                    pid = 0

                # Parse memory usage
                try:
                    memory_mb = float(memory_str.replace(',', '').replace(' K', '')) / 1024
                except (ValueError, AttributeError):
                    memory_mb = None

                processes.append(ProcessInfo(
                    name=name,
                    pid=pid,
                    user=row.get('User Name', row.get('Username', '')),
                    memory_mb=memory_mb,
                    status=row.get('Status', '')
                ))
        except Exception as e:
            raise Exception(f"Error parsing tasklist CSV: {e}")

        return processes


class WindowsWMICCollector(ProcessCollector):
    """Windows WMIC collector for more detailed process information."""

    def __init__(self, remote_host: Optional[str] = None, username: Optional[str] = None,
                 password: Optional[str] = None):
        super().__init__(remote_host, username)
        self.password = password

    def is_available(self) -> bool:
        """Check if WMIC command is available."""
        try:
            result = subprocess.run(['wmic', '/?'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def collect_processes(self) -> List[ProcessInfo]:
        """Collect processes using Windows WMIC command."""
        cmd = [
            'wmic', 'process', 'get',
            'Name,ProcessId,ParentProcessId,CommandLine,PageFileUsage,CreationDate,Status',
            '/format:csv'
        ]

        # Add remote connection parameters
        if self.remote_host:
            cmd.extend(['/node:' + self.remote_host])
            if self.username:
                cmd.extend(['/user:' + self.username])
            if self.password:
                cmd.extend(['/password:' + self.password])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            if result.returncode != 0:
                raise Exception(f"WMIC command failed: {result.stderr}")

            return self._parse_wmic_csv(result.stdout)
        except subprocess.TimeoutExpired:
            raise Exception("WMIC command timed out")
        except Exception as e:
            raise Exception(f"Error running WMIC: {e}")

    def _parse_wmic_csv(self, csv_output: str) -> List[ProcessInfo]:
        """Parse WMIC CSV output into ProcessInfo objects."""
        processes = []
        try:
            lines = csv_output.strip().split('\n')
            if len(lines) < 2:
                return processes

            reader = csv.DictReader(StringIO(csv_output))
            for row in reader:
                if not row.get('Name'):  # Skip empty rows
                    continue

                # Extract fields
                name = row.get('Name', '')
                command_line = row.get('CommandLine', '')

                # Parse numeric fields
                try:
                    pid = int(row.get('ProcessId', '0'))
                except (ValueError, TypeError):
                    pid = 0

                try:
                    ppid = int(row.get('ParentProcessId', '0'))
                except (ValueError, TypeError):
                    ppid = None

                try:
                    memory_mb = float(row.get('PageFileUsage', '0')) / 1024
                except (ValueError, TypeError):
                    memory_mb = None

                processes.append(ProcessInfo(
                    name=name,
                    pid=pid,
                    ppid=ppid,
                    memory_mb=memory_mb,
                    command_line=command_line,
                    status=row.get('Status', '')
                ))
        except Exception as e:
            raise Exception(f"Error parsing WMIC CSV: {e}")

        return processes


class WindowsWevtutilCollector(ProcessCollector):
    """Windows Event Log collector using wevtutil."""

    def __init__(self, log_name: Optional[str] = None, remote_host: Optional[str] = None,
                 username: Optional[str] = None, password: Optional[str] = None):
        super().__init__(remote_host, username)
        self.password = password
        self.log_name = log_name

    def is_available(self) -> bool:
        """Check if wevtutil command is available."""
        try:
            result = subprocess.run(['wevtutil', 'el'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def _list_logs(self) -> List[str]:
        """List all available event logs."""
        try:
            result = subprocess.run(['wevtutil', 'el'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise Exception(f"Failed to list logs: {result.stderr}")
            return result.stdout.strip().splitlines()
        except Exception as e:
            raise Exception(f"Error listing logs: {e}")

    def _parse_wevtutil_xml(self, xml_output: str) -> List[ProcessInfo]:
        """Parse wevtutil XML output into ProcessInfo-like objects."""
        events = []
        try:
            root = ET.fromstring(f"<Events>{xml_output}</Events>")
            for event in root.findall("Event"):
                system = event.find("System")
                event_data = event.find("EventData")

                name = system.find("Provider").attrib.get("Name", "")
                pid = int(system.findtext("EventID", default="0"))
                time_created = system.find("TimeCreated").attrib.get("SystemTime", "")
                status = system.findtext("Level", default="")

                message = self._extract_event_data(event_data)

                events.append(ProcessInfo(
                    name=name,
                    pid=pid,
                    ppid=None,
                    memory_mb=None,
                    command_line=message,
                    status=status,
                    start_time=time_created
                ))
        except ET.ParseError:
            pass
        return events

    def _extract_event_data(self, event_data_elem) -> str:
        """Extract and format event data."""
        if event_data_elem is None:
            return ""
        return " | ".join(
            f"{data.attrib.get('Name', '')}: {data.text}" for data in event_data_elem.findall("Data")
        )

    def collect_processes(self) -> List[ProcessInfo]:
        """Collect recent events from specified or all logs."""
        if self.log_name:
            return self._collect_from_log(self.log_name)
        else:
            logs = spinner_wrapper(self._list_logs)
            all_events = []
            for log in logs:
                try:
                    all_events.extend(self._collect_from_log(log, max_events=5))
                except Exception:
                    continue
            return all_events

    def _collect_from_log(self, log_name: str, max_events: int = 50) -> List[ProcessInfo]:
        """Collect recent events from a specific log."""
        cmd = ['wevtutil', 'qe', log_name, '/f:xml', f'/c:{max_events}', '/rd:true']

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise Exception(f"wevtutil failed for {log_name}: {result.stderr}")
            return self._parse_wevtutil_xml(result.stdout)
        except Exception as e:
            raise Exception(f"Error collecting from {log_name}: {e}")


class WindowsWevTutilSystemLogCollectorWrapper:
    """Wrapper for collecting events specifically from the System log."""

    def __init__(self, log_name: str = "System", events_to_process: int = 50, remote_host: Optional[str] = None,
                 username: Optional[str] = None, password: Optional[str] = None):
        self.collector = WindowsWevtutilCollector(
            log_name=log_name,
            remote_host=remote_host,
            username=username,
            password=password
        )
        self.events_to_process = events_to_process

    def get_system_events(self) -> List[ProcessInfo]:
        if not self.collector.is_available():
            raise EnvironmentError("wevtutil is not available on this system.")
        return self.collector._collect_from_log("System", max_events=self.events_to_process)


# TODO: Add new windows methods here


class UnixPsCollector(ProcessCollector):
    """Unix/Linux ps command collector."""

    def __init__(self, remote_host: Optional[str] = None, username: Optional[str] = None,
                 ssh_key: Optional[str] = None):
        self.remote_host = remote_host
        self.username = username
        self.ssh_key = ssh_key

    def is_available(self) -> bool:
        """Check if ps command is available."""
        try:
            result = subprocess.run(['ps', '--version'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0 or 'ps' in result.stderr.lower()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            # Try alternative check
            try:
                result = subprocess.run(['ps', '-o', 'pid'], capture_output=True, text=True, timeout=5)
                return result.returncode == 0
            except:
                return False

    def collect_processes(self) -> List[ProcessInfo]:
        """Collect processes using Unix ps command."""
        # Use comprehensive ps format
        ps_format = 'pid,ppid,user,pcpu,pmem,comm,args,stat,lstart'
        cmd = ['ps', '-eo', ps_format]

        # Handle remote execution via SSH
        if self.remote_host:
            ssh_cmd = ['ssh']
            if self.ssh_key:
                ssh_cmd.extend(['-i', self.ssh_key])

            if self.username:
                ssh_cmd.append(f'{self.username}@{self.remote_host}')
            else:
                ssh_cmd.append(self.remote_host)

            ssh_cmd.extend(cmd)
            cmd = ssh_cmd

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise Exception(f"ps command failed: {result.stderr}")

            return self._parse_ps_output(result.stdout)
        except subprocess.TimeoutExpired:
            raise Exception("ps command timed out")
        except Exception as e:
            raise Exception(f"Error running ps: {e}")

    def _parse_ps_output(self, ps_output: str) -> List[ProcessInfo]:
        """Parse ps output into ProcessInfo objects."""
        processes = []
        lines = ps_output.strip().split('\n')

        if len(lines) < 2:
            return processes

        # Skip header line
        for line in lines[1:]:
            parts = line.strip().split(None, 8)  # Split into max 9 parts
            if len(parts) < 6:
                continue

            try:
                pid = int(parts[0])
                ppid = int(parts[1]) if parts[1] != '0' else None
                user = parts[2]
                cpu_percent = float(parts[3])
                memory_percent = float(parts[4])
                name = parts[5]
                command_line = parts[6] if len(parts) > 6 else ''
                status = parts[7] if len(parts) > 7 else ''
                start_time = parts[8] if len(parts) > 8 else ''

                processes.append(ProcessInfo(
                    name=name,
                    pid=pid,
                    ppid=ppid,
                    user=user,
                    cpu_percent=cpu_percent,
                    memory_mb=None,  # ps gives percentage, not absolute
                    command_line=command_line,
                    status=status,
                    start_time=start_time
                ))
            except (ValueError, IndexError) as e:
                raise Exception(f"Error parsing ps line '{line}': {e}")
                continue

        return processes


def get_running_processes_for_verification(
        run_process_verification_logger: Optional[SafeLogger] = None,
        remote_host: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ssh_key: Optional[str] = None,
        use_cache: bool = True,
        filter_system: bool = True,
        max_workers: int = 2  # NEW: Parallel processing support
    ) -> list:
    """
    Get running processes optimized for verification with enhanced filtering.

    NEW Features:
    - Parallel collector execution for faster results
    - Better error aggregation and reporting
    - Enhanced logging with performance metrics
    - Intelligent fallback mechanism

    This function provides an optimized interface for the verify_apps_running
    function with intelligent caching, filtering, and cross-platform support.

    Args:
        run_process_verification_logger: SafeLogger instance to log output or None to suppress it.
        remote_host: Optional remote host for process collection
        username: Username for remote connection
        password: Password for remote connection (Windows)
        ssh_key: SSH key path for remote connection (Unix)
        use_cache: Whether to use cached results if available
        filter_system: Whether to filter out system processes

    Returns:
        set: Set of running process names optimized for verification

    Raises:
        Exception: If no process collectors are available or all fail to create or return
    """
    os_type = get_current_os()

    # Create multiple collectors for parallel execution
    collectors = []
    try:
        # Primary collector
        primary = create_optimal_collector(
            os_type, remote_host, username, password=password, ssh_key=ssh_key
        )
        collectors.append(primary)

        # Fallback collector (different type)
        if os_type == OSType.WINDOWS and isinstance(primary, WindowsWMICCollector):
            collectors.append(WindowsTasklistCollector(remote_host, username, password))

    except Exception as e:
        if run_process_verification_logger:
            run_process_verification_logger.logger.error(f"Collector creation failed: {e}")

        # Fallback to original method
        collectors = _get_collectors_for_os(os_type, remote_host, username, password, ssh_key)

    # Parallel execution with ThreadPoolExecutor
    processes = []
    verification_errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all available collectors
        future_to_collector = {
            executor.submit(collector.collect_with_retry): collector
            for collector in collectors if collector.is_available()
        }

        # Get first successful result
        for future in as_completed(future_to_collector, timeout=30):
            collector = future_to_collector[future]
            try:
                result = future.result()
                if result:
                    processes = result
                    if run_process_verification_logger:
                        metrics = collector._performance_metrics
                        run_process_verification_logger.logger.info(
                            f"Collected {len(processes)} processes in "
                            f"{metrics['collection_time']:.2f}s using {collector.__class__.__name__}"
                        )
                    break
            except Exception as e:
                verification_errors.append(f"{collector.__class__.__name__}: {e}")
                if run_process_verification_logger:
                    run_process_verification_logger.logger.debug(f"Collector failed: {e}")

    if not processes:
        error_msg = f"All collectors failed. Errors: {'; '.join(verification_errors)}"
        if run_process_verification_logger:
            run_process_verification_logger.logger.error(error_msg)
        raise Exception(error_msg)
    # If we need only names - replace with line below return statement
    # return _process_list_to_name_set(processes, os_type, filter_system)
    return _process_filtered_list(processes, os_type, filter_system)

def verify_apps_running(
        computer: Dict[str, Union[str, List[str], Dict[str, List[str]]]],
        list_of_keys_to_test: List[str],
        verify_apps_running_logger: Optional[SafeLogger] = None,
        remote_host: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ssh_key: Optional[str] = None,
        detailed_errors: bool = True,
        parallel_verification: bool = True  # NEW: Enable parallel verification
    ) -> Tuple[bool, List[str]]:
    """
    Enhanced verification of running applications with -
    improved error handling, dependency graph analysis and parallel processing.

    This function verifies that all applications under specified keys are running
    on the target computer, with support for dependency checking and detailed
    error reporting.

    Added Features:
    - Dependency graph validation
    - Parallel verification for better performance
    - Enhanced error categorization
    - Progress tracking for large app lists

    Args:
        computer: Dictionary containing app configurations
        list_of_keys_to_test: List of keys to verify
        verify_apps_running_logger: Logger instance for detailed logging
        remote_host: Optional remote host for process collection
        username: Username for remote connection
        password: Password for remote connection (Windows)
        ssh_key: SSH key path for remote connection (Unix)
        detailed_errors: Whether to include detailed error information
        parallel_verification: Whether to run verification in parallel

    Returns:
        Tuple[bool, List[str]]: Success status and list of errors

    Raises:
        ValueError: If invalid configuration is provided
    """

    # Input validation with better error messages
    validation_errors = []
    if not computer:
        validation_errors.append("Computer configuration cannot be empty")
    if not list_of_keys_to_test:
        validation_errors.append("Keys to test cannot be empty")

    # Validate key existence early
    missing_keys = [key for key in list_of_keys_to_test if key not in computer]
    if missing_keys:
        validation_errors.append(f"Missing keys in configuration: {missing_keys}")

    if validation_errors:
        raise ValueError("; ".join(validation_errors))

    # Enhanced app extraction with dependency mapping
    expected_apps = []
    dependency_graph = {}
    verify_apps_running_errors = []

    for key in list_of_keys_to_test:
        value = computer[key]
        if isinstance(value, list):
            expected_apps.extend(value)
        elif isinstance(value, str):
            expected_apps.append(value)
        else:
            verify_apps_running_errors.append(f"Invalid value type for key '{key}': {type(value)}")

    # Build dependency graph
    app_dependencies = computer.get('common_app_order', {})
    for app in expected_apps:
        app_name = Path(app).name
        dependency_graph[app_name] = app_dependencies.get(app_name, [])

    if not expected_apps:
        return True, []

    # Collect running processes (using enhanced function)
    try:
        running_apps = get_running_processes_for_verification(
            run_process_verification_logger=verify_apps_running_logger,
            remote_host=remote_host,
            username=username,
            password=password,
            ssh_key=ssh_key,
            use_cache=True,
            filter_system=True
        )
    except Exception as e:
        error_msg = f"Failed to collect running processes: {e}"
        if detailed_errors:
            error_msg += f" (Remote: {remote_host}, User: {username})"
        verify_apps_running_errors.append(error_msg)
        return False, verify_apps_running_errors

    # Parallel verification with dependency checking
    if parallel_verification and len(expected_apps) > 5:
        return _verify_apps_parallel(
            expected_apps, dependency_graph, running_apps,
            verify_apps_running_errors, detailed_errors, verify_apps_running_logger
        )
    else:
        return _verify_apps_sequential(
            expected_apps, dependency_graph, running_apps,
            verify_apps_running_errors, detailed_errors, verify_apps_running_logger
        )


# Parallel verification helper
def _verify_apps_parallel(expected_apps: List[str], dependency_graph: Dict[str, List[str]],
                          running_apps: set, errors: List[str], detailed_errors: bool,
                          logger: Optional[SafeLogger]) -> Tuple[bool, List[str]]:
    """
    Parallel application verification with dependency checking.

    Uses ThreadPoolExecutor to verify multiple apps simultaneously while
    respecting dependency relationships.
    """
    all_success = True
    max_name_length = max(len(Path(app).name) for app in expected_apps)

    with ThreadPoolExecutor(max_workers=min(len(expected_apps), 4)) as executor:
        future_to_app = {
            executor.submit(
                _verify_single_app_with_dependencies,
                Path(app).name,
                dependency_graph.get(Path(app).name, []),
                running_apps,
                max_name_length,
                detailed_errors
            ): app
            for app in expected_apps
        }

        for future in as_completed(future_to_app):
            app = future_to_app[future]
            try:
                result = future.result()
                if not result.success:
                    all_success = False
                    errors.extend(result.errors)
                    if logger:
                        for error in result.errors:
                            logger.logger.warning(error)
                elif logger:
                    logger.logger.debug(f"✓ {Path(app).name} verified successfully")
            except Exception as e:
                all_success = False
                error_msg = f"Verification failed for {app}: {e}"
                errors.append(error_msg)
                if logger:
                    logger.logger.error(error_msg)

    return all_success, errors


# Sequential verification helper (fallback if parallel failes for any reason)
def _verify_apps_sequential(expected_apps: List[str], dependency_graph: Dict[str, List[str]],
                            running_apps: set, inherited_errors: List[str], detailed_errors: bool,
                            logger: Optional[SafeLogger]) -> Tuple[bool, List[str]]:
    """Sequential verification fallback with progress tracking."""
    all_success = True
    max_name_length = max(len(Path(app).name) for app in expected_apps)

    for i, app in enumerate(expected_apps, 1):
        app_name = Path(app).name

        if logger:
            logger.logger.debug(f"Verifying {i}/{len(expected_apps)}: {app_name}")

        result = _verify_single_app_with_dependencies(
            app_name,
            dependency_graph.get(app_name, []),
            running_apps,
            max_name_length,
            detailed_errors
        )

        if not result.success:
            all_success = False
            inherited_errors.extend(result.errors)
            if logger:
                for error in result.errors:
                    logger.logger.warning(error)
        elif logger:
            logger.logger.debug(f"✓ {app_name} verified successfully")

    return all_success, inherited_errors


def _verify_single_app_with_dependencies(
        app_name: str,
        expected_dependencies: List[str],
        running_apps: set,
        max_name_length: int,
        detailed_errors: bool
    ) -> AppVerificationResult:
    """
    Verify a single application and its dependencies.

    Args:
        app_name: Name of the application to verify
        expected_dependencies: List of expected dependencies
        running_apps: Set of currently running applications
        max_name_length: Maximum length for formatting
        detailed_errors: Whether to include detailed error information

    Returns:
        AppVerificationResult: Verification result with errors
    """
    verify_single_errors = []
    missing_dependencies = []

    # Check main application
    if app_name not in running_apps:
        missing_dependencies.append(app_name)

    # Check dependencies
    for dep in expected_dependencies:
        if dep not in running_apps:
            missing_dependencies.append(dep)

    if missing_dependencies:
        if expected_dependencies:
            error_msg = (f"Application {app_name:{max_name_length}} and/or its "
                         f"dependencies {missing_dependencies} are not running")
        else:
            error_msg = f"Application {app_name:{max_name_length}} is not running"

        if detailed_errors:
            error_msg += f" (Expected: {app_name}, Dependencies: {expected_dependencies})"

        verify_single_errors.append(error_msg)
        return AppVerificationResult(False, verify_single_errors, app_name, missing_dependencies)

    return AppVerificationResult(True, [], app_name, [])


def kill_application_by_name(app_name: str, logger=Union[SafeLogger, None],
                             timeout: int = 15) -> Tuple[bool, List[str]]:
    """
    Kill all processes related to an application by name and verify termination.

    Args:
        app_name: Name of the application (e.g., 'chrome.exe')
        logger: Optional logger
        timeout: Time to wait for all processes to terminate

    Returns:
        Tuple of success status and list of errors
    """
    kill_app_by_name_errors = []
    try:
        all_procs = psutil.process_iter(['pid', 'ppid', 'name', 'create_time'])
        matching_procs = [p for p in all_procs if p.info['name'].lower() == app_name.lower()]
        if not matching_procs:
            return True, []

        # Identify main process (earliest start time or lowest PPID)
        main_proc = min(matching_procs, key=lambda p: (p.info.get('ppid', 0), p.info.get('create_time', 0)))

        if logger:
            logger.logger.info(f"Killing main process {main_proc.pid} for {app_name}")

        main_proc.kill()

        # Wait for all related processes to terminate
        start_time = time.time()
        while time.time() - start_time < timeout:
            still_alive = [p for p in psutil.process_iter(['name']) if p.info['name'].lower() == app_name.lower()]
            if not still_alive:
                return True, []
            time.sleep(1)

        kill_app_by_name_errors.append(f"Timeout: {app_name} processes still running after {timeout} seconds")
        return False, errors
    except Exception as e:
        kill_app_by_name_errors.append(f"Failed to kill {app_name}: {e}")
        return False, errors


def kill_with_verification(computer: Dict[str, Union[str, List[str], Dict[str, List[str]]]],
                           list_of_keys_to_test: List[str],
                           kill_with_verification_logger=None) -> Tuple[bool, List[str]]:
    kill_with_verification_errors = []
    try:
        running_processes = get_running_processes_for_verification(
            run_process_verification_logger=kill_with_verification_logger)
    except Exception as e:
        kill_with_verification_errors.append(f"Failed to collect running processes: {e}")
        return False, kill_with_verification_errors

    # Build list of expected apps with PIDs
    expected_apps = []
    for key in list_of_keys_to_test:
        if key in computer:
            val = computer[key]
            if isinstance(val, list):
                expected_apps.extend(val)
            elif isinstance(val, str):
                expected_apps.append(val)

    to_kill = []
    for proc in running_processes:
        if proc.name in expected_apps:
            to_kill.append((proc.name, proc.pid))

    grouped = defaultdict(list)
    for key, value in to_kill:
        grouped[key].append(value)
    to_kill = [(key, values[0] if len(values) == 1 else values) for key, values in grouped.items()]

    # Attempt to kill using kill_application_by_name first, if unsuccessfully, try by PID
    for name, pid in to_kill:
        kill_by_name_success, errs = kill_application_by_name(name, kill_with_verification_logger)
        if not kill_by_name_success:
            kill_with_verification_errors.extend(errs)
            try:
                if platform.system().lower() == "windows":
                    os.system(f"taskkill /PID {pid} /F")
                else:
                    os.kill(pid, signal.SIGKILL)
            except Exception as e:
                kill_with_verification_errors.append(f"Failed to kill {name} (PID {pid}): {e}")
    # Recheck
    try:
        remaining = get_running_processes_for_verification(
            run_process_verification_logger=kill_with_verification_logger)
    except Exception as e:
        kill_with_verification_errors.append(f"Failed to re-collect running processes: {e}")
        return False, kill_with_verification_errors

    remaining_names = {p.name for p in remaining}
    still_running = [name for name, _ in to_kill if name in remaining_names]

    if still_running:
        kill_with_verification_errors.extend([f"{name} still running after kill attempt" for name in still_running])
        return False, kill_with_verification_errors
    return True, []


def verify_apps_killed(computer: Dict[str, Union[str, List[str], Dict[str, List[str]]]],
                       list_of_keys_to_test: List[str],
                       pid_map: Dict[str, Union[int, List[int]]],
                       verify_apps_killed_logger=Union[SafeLogger, None]) -> Tuple[bool, List[str]]:
    verify_apps_killed_errors = []
    try:
        running_processes = get_running_processes_for_verification(
            run_process_verification_logger=verify_apps_killed_logger)
    except Exception as e:
        verify_apps_killed_errors.append(f"Failed to collect running processes: {e}")
        return False, verify_apps_killed_errors

    current = {p.name: p.pid for p in running_processes}

    expected_apps = []
    for key in list_of_keys_to_test:
        if key in computer:
            val = computer[key]
            if isinstance(val, list):
                expected_apps.extend(val)
            elif isinstance(val, str):
                expected_apps.append(val)

    verify_apps_killed_success = True
    for name, old_pid in pid_map.items():
        if name not in expected_apps:
            verify_apps_killed_errors.append(f"Warning: {name} in PID map but not in expected apps")
            continue
        if name not in current:
            if verify_apps_killed_logger:
                verify_apps_killed_logger.logger.info(f"{name} was killed")
        elif current[name] != old_pid:
            """
                Returns True if the current_pid is considered 'killed' based on expected_pid.
                - If both are ints: they must not be equal.
                - If current_pid is int and expected_pid is list: current_pid must not be in expected_pid.
                - If current_pid is list and expected_pid is int: expected_pid must not be in current_pid.
                - If both are lists: no overlap allowed.
            """
            if isinstance(current[name], int):
                if isinstance(old_pid, int) and current[name] == old_pid:
                    verify_apps_killed_success = False
                    verify_apps_killed_errors.append(
                        f"App {name} has new PID {current[name]} equal to old PID{old_pid}")
                elif isinstance(old_pid, list):
                    if current[name] in old_pid:
                        verify_apps_killed_success = False
                        verify_apps_killed_errors.append(
                            f"App {name} has new PID {current[name]} that is part of old PIDs list: {old_pid}")
            elif isinstance(current[name], list):
                if isinstance(old_pid, int):
                    if old_pid in current[name]:
                        verify_apps_killed_success = False
                        verify_apps_killed_errors.append(
                            f"App {name} has new PID list: {current[name]} that has old PID as part of it: {old_pid}")
                elif isinstance(old_pid, list):
                    if any(pid in old_pid for pid in current[name]):
                        verify_apps_killed_success = False
                        verify_apps_killed_errors.append(
                            f"App {name} has new PID list: {current[name]} intersecting with old PID list: {old_pid}")
            if verify_apps_killed_logger and verify_apps_killed_success:
                verify_apps_killed_logger.logger.info(f"{name} was restarted "
                                                      f"(PID changed from {old_pid} to {current[name]})")
        else:
            verify_apps_killed_errors.append(f"{name} still running with same PID {old_pid}")
            verify_apps_killed_success = False

    return verify_apps_killed_success, verify_apps_killed_errors


if __name__ == "__main__":

    # Tester modules
    computer_config = {
        'critical_apps': ['notepad.exe', 'calc.exe'],
        'apps': ['chrome.exe'],
        'common_app_order': {
            'notepad.exe': ['explorer.exe'],
            'calc.exe': ['dwm.exe']
        }
    }

    print("=== verify_apps_running ===")
    success, errors = verify_apps_running(
        computer=computer_config,
        list_of_keys_to_test=['critical_apps', 'apps'],
        verify_apps_running_logger=None
    )
    print(f"Success: {success}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")

    print("\n=== kill_with_verification ===")
    success, errors = kill_with_verification(
        computer=computer_config,
        list_of_keys_to_test=['critical_apps', 'apps'],
        kill_with_verification_logger=None
    )
    print(f"Killed successfully: {success}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")

    print("\n=== verify_apps_killed ===")
    simulated_pid_map = {
        'notepad.exe': 1234,
        'calc.exe': 5678,
        'chrome.exe': 9012
    }
    success, errors = verify_apps_killed(
        computer=computer_config,
        list_of_keys_to_test=['critical_apps', 'apps'],
        pid_map=simulated_pid_map,
        verify_apps_killed_logger=None
    )
    print(f"Verification result: {success}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")