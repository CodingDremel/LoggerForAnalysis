import os

class SafePathValidator:
    """Validates paths for safety"""
    # Paths to exclude from execution and/or logging attempts
    DANGEROUS_PATHS = [
        'C:\\Windows\\System32',
        'C:\\Program Files',
        'C:\\Program Files (x86)',
        '/bin', '/sbin', '/usr/bin', '/usr/sbin',
        '/etc', '/boot', '/sys', '/proc'
    ]

    @staticmethod
    def is_safe_path(path: str) -> tuple[bool, str]:
        """Check if a path is safe to operate on, i.e. not part of dangerous paths defined above"""
        try:
            abs_path = os.path.abspath(path)

            # Check vs dangerous paths
            if any(abs_path.startswith(dangerous) for dangerous in SafePathValidator.DANGEROUS_PATHS):
                return False, f"Path {abs_path} is in protected system directory"

            # Check if path supplied and converted to abs path exists and is accessible
            parent_dir = os.path.dirname(abs_path)
            if not os.path.exists(parent_dir):
                return False, f"Parent directory {parent_dir} does not exist"

            return True, "Path is safe"

        except Exception as e:
            return False, f"Path validation error: {str(e)}"

    @staticmethod
    def is_os_safe_path(path: str) -> tuple[bool, str]:
        """
        Validate that the given path is 'safe' for file operations.
        Returns (bool, str): (True, "Safe") or (False, "Reason")
        """
        if not isinstance(path, str):
            return False, "Not a string"
        if not path.strip():
            return False, "Path is empty"
        if not os.path.isabs(path):
            return False, "Path is not absolute"
        # Optional: Check for forbidden patterns or characters
        forbidden = ['..', '~', '//', '\\\\']
        if any(f in path for f in forbidden):
            return False, f"Path contains forbidden sequence: {forbidden}"
        # Existence and permissions checks
        if not os.path.exists(path):
            return False, "Path does not exist"
        if not os.access(path, os.R_OK):
            return False, "No read permission"
        # Add any additional checks as needed
        return True, "Safe"