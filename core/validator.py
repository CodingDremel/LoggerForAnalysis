from typing import List, Dict, Any, Optional, Tuple

def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration file"""
    errors = []

    if "computers" not in config:
        errors.append("Missing 'computers' section in config")
        return errors

    # Handle both list and dict formats
    computers = config["computers"]
    if isinstance(computers, dict):
        computers_items = computers.items()
    elif isinstance(computers, list):
        computers_items = enumerate(computers)
    else:
        errors.append("'computers' must be a list or dict")
        return errors

    for identifier, computer in computers_items:
        if isinstance(computers, dict):
            # Dict format - identifier is the computer name
            if "computer_name" not in computer and "name" not in computer:
                errors.append(f"Computer '{identifier}': Missing 'computer_name' or 'name'")
        else:
            # List format - identifier is index
            if "computer_name" not in computer:
                errors.append(f"Computer {identifier}: Missing 'computer_name'")

        if "ip" not in computer:
            errors.append(f"Computer {identifier}: Missing 'ip'")

        # Validate file extensions if present
        if "file_extensions" in config and not isinstance(config["file_extensions"], list):
            errors.append("'file_extensions' must be a list")

    return errors

