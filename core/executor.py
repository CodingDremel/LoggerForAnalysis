def run_operation_safely(operation: str, computer: Dict[str, Any], config: Dict[str, Any],
                         kill_mode: str = "SAFE", start_mode: str = "CRITICAL",
                         search_for_apps: bool = True, log_queue=None, log_level=logging.DEBUG,
                         simulation_mode: bool = False) -> OperationResult:
    """Safely run operation on a single computer"""
    # logger = safe_ops.safe_logger.get_process_logger(f"Worker-{computer['computer_name']}")
    logger = ProcessLogger(log_queue, process_name=f"Worker-{computer['name']}", min_level=log_level)
    safe_ops = SafeRemoteOperations(config, logger, simulation_mode)

    operations = {
        "delete_files": safe_ops.safe_delete_files,
        "copy_files": safe_ops.safe_copy_files,
        "start_applications": lambda comp: safe_ops.safe_start_applications(comp, start_mode=start_mode,
                                                                            search_for_apps=search_for_apps),
        "kill_applications": lambda comp: safe_ops.safe_kill_applications(comp, kill_mode=kill_mode),
        "start_apps": safe_ops.start_apps,
        "restart_application": safe_ops.safe_restart_application
    }

    if operation in operations:
        result = operations[operation](computer)
        if hasattr(logger, 'flush'):
            logger.flush()
        return result
    else:
        logger.error(f"Unknown operation: {operation}")
        return OperationResult(
            operation=operation,
            computer_name=computer['name'],
            success_count=0,
            failure_count=1,
            errors=[f"Unknown operation: {operation}"],
            duration=0.0,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )


def generate_report(results: List[OperationResult], output_file: str = "operation_report.json"):
    """Generate comprehensive operation report"""
    report = {
        "summary": {
            "total_operations": len(results),
            "total_successes": sum(r.success_count for r in results),
            "total_failures": sum(r.failure_count for r in results),
            "total_duration": sum(r.duration for r in results),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        },
        "operations": []
    }

    for result in results:
        report["operations"].append({
            "operation": result.operation,
            "computer_name": result.computer_name,
            "success_count": result.success_count,
            "failure_count": result.failure_count,
            "errors": result.errors,
            "duration": result.duration,
            "timestamp": result.timestamp
        })

    # Write report atomically
    with atomic_file_operation(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    return report


def load_config_file(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from file with fallback options"""
    # Determine config file path
    if config_path and os.path.isabs(config_path):
        # Absolute path provided
        final_config_path = config_path
    elif config_path:
        # Relative path provided
        script_dir = os.path.dirname(os.path.abspath(__file__))
        final_config_path = os.path.join(script_dir, config_path)
    else:
        # Default: look for config.json in script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        final_config_path = os.path.join(script_dir, 'config.json')

        # If default doesn't exist, try defaults.json
        if not os.path.exists(final_config_path):
            defaults_path = os.path.join(script_dir, 'defaults.json')
            if os.path.exists(defaults_path):
                final_config_path = defaults_path

    if not os.path.exists(final_config_path):
        raise FileNotFoundError(f"No configuration file found at {final_config_path}")

    # Parse using new ConfigParser
    parser = ConfigParser(final_config_path)
    return parser.load_config()


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


def main():
    parser = argparse.ArgumentParser(
        description="Safe Remote Operations Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
              python safe_remote_ops.py delete_files copy_files --simulation
              python safe_remote_ops.py start_applications --max-workers 4
              python safe_remote_ops.py copy_files --config /path/to/config.json
              python safe_remote_ops.py copy_files --report custom_report.json
        """
    )

    LOG_LEVELS = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }

    def log_level_type(level_str: str) -> str:
        level_upper = level_str.upper()
        if level_upper not in LOG_LEVELS:
            raise argparse.ArgumentTypeError(f"Invalid log level supplied: {level_str}...")
        return level_upper

    def kill_type(kill_str: str) -> str:
        kill_up = kill_str.upper()
        if kill_up not in ["ALL", "SAFE", "CRITICAL", "KILL"]:
            raise argparse.ArgumentTypeError(f"Invalid kill mode supplied: {kill_str}...")
        return kill_up

    parser.add_argument("operations", nargs="+",
                        help="Operations to perform: delete_files, copy_files, start_applications, "
                             "kill_applications, start_apps, restart_application")
    parser.add_argument("--config", default=None,
                        help="Configuration file path (default: config.json in script directory)")
    parser.add_argument("--simulation", action="store_true",
                        help="Run in simulation mode (no actual operations)")
    parser.add_argument("--backup", action="store_true",
                        help="Backup stuff before delete kicks in")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Maximum number of concurrent workers")
    parser.add_argument("--report", default="operation_report.json",
                        help="Output report file")
    parser.add_argument("--verbosity", type=log_level_type, default="DEBUG",
                        help="Enable log verbosity (case-insensitive): DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument("--kill-mode", type=kill_type, default="SAFE",
                        help="Specify the kill mode for kill_operations method, "
                             "case-insensitive: ALL, SAFE, CRITICAL, KILL")
    parser.add_argument("--start-mode", type=kill_type, default="CRITICAL",
                        help="Specify the kill mode for kill_operations method, "
                             "case-insensitive: ALL, SAFE, CRITICAL")
    parser.add_argument("--search-for-apps", action="store_true",
                        help="Search for applications if they are not defined as absolute paths in config")

    args = parser.parse_args()

    # Setup logging
    log_level = LOG_LEVELS[args.verbosity]

    safe_logger = SafeLogger(log_level=log_level)
    shared_log_queue = safe_logger._log_queue

    safe_logger.start_log_listener()

    # Load and validate configuration using new loader
    try:
        config = load_config_file(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading configuration: {e}")
        return 1

    config_errors = validate_config(config)
    if config_errors:
        print("Configuration errors:")
        for error in config_errors:
            print(f"  - {error}")
        return 1

    config['backup_before_delete'] = args.backup

    try:
        # Print configuration summary
        safe_logger.logger.info(f"Configuration loaded: {len(config['computers'])} computers")
        safe_logger.logger.info(f"Operations to perform: {', '.join(args.operations)}")
        safe_logger.logger.info(f"Simulation mode: {'ON' if args.simulation else 'OFF'}")
        safe_logger.logger.info(f"Max workers: {args.max_workers}")

        # Execute operations
        all_results = []

        for operation in args.operations:
            safe_logger.logger.info(f"Starting operation: {operation}")

            # Use ProcessPoolExecutor for controlled concurrency
            with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                # Submit all tasks - handle both dict and list formats
                computers = config["computers"]

                # Create list of computer configurations with names added
                computer_configs = []
                if isinstance(computers, dict):
                    for comp_name, comp_config in computers.items():
                        # Ensure computer_name is set
                        comp_config = comp_config.copy()  # Don't modify original
                        if 'computer_name' not in comp_config:
                            comp_config['computer_name'] = comp_name
                        computer_configs.append(comp_config)
                elif isinstance(computers, list):
                    computer_configs = computers
                else:
                    raise ValueError("'computers' must be a list or dict")

                operation_runner = partial(run_operation_safely, config=config, simulation_mode=args.simulation,
                                           kill_mode=args.kill_mode, start_mode=args.start_mode,
                                           search_for_apps=args.search_for_apps,
                                           log_queue=shared_log_queue, log_level=log_level)
                future_to_computer = {
                    executor.submit(operation_runner, operation, computer): computer
                    for computer in computer_configs
                }

                # Collect results as they complete
                for future in as_completed(future_to_computer):
                    computer = future_to_computer[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as e:
                        computer_name = computer.get('name', 'Unknown') if isinstance(computer, dict) else str(computer)
                        safe_logger.logger.error(f"Operation failed for [{computer_name}]: {str(e)}")
                        all_results.append(
                            OperationResult(
                                operation=operation,
                                computer_name=computer_name,
                                success_count=0,
                                failure_count=1,
                                errors=[str(e)],
                                duration=0.0,
                                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
                            )
                        )

        # Generate report
        report = generate_report(all_results, args.report)
        safe_logger.logger.info(f"Operation completed. Report saved to: {args.report}")
        safe_logger.logger.info(
            f"Summary: {report['summary']['total_successes']} successes, "
            f"{report['summary']['total_failures']} failures")
        # Stop logger before exiting
        safe_logger.stop_log_listener()
        return 0

    except KeyboardInterrupt:
        safe_logger.logger.warning("Operation interrupted by user")
        safe_logger.stop_log_listener()
        return 1
    except Exception as e:
        safe_logger.logger.critical(f"Unexpected error: {str(e)}")
        safe_logger.stop_log_listener()
        return 1


if __name__ == "__main__":
