import argparse
import logging
import time

from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import core.logger
import core.executor
import core.validator
from core.core_html_report import generate_html_report
from run_on_platform.base import OperationResult
from core.executor import run_operation_safely

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
    parser.add_argument("--html", action="store_true",
                        help="Generate HTML file about current run")

    args = parser.parse_args()

    # Setup logging
    log_level = LOG_LEVELS[args.verbosity]

    safe_logger = core.logger.SafeLogger(log_level=log_level)
    shared_log_queue = safe_logger._log_queue

    safe_logger.start_log_listener()

    # Load and validate configuration using new loader
    try:
        config = core.executor.load_config_file(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading configuration: {e}")
        return 1

    config_errors = core.validator.validate_config(config)
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
        report = core.executor.generate_report(all_results, args.report)
        safe_logger.logger.info(f"Operation completed. Report saved to: {args.report}")
        safe_logger.logger.info(
            f"Summary: {report['summary']['total_successes']} successes, "
            f"{report['summary']['total_failures']} failures")
        if args.html:
            # log_folder: path where logs/results are saved
            # report_path: path to operation_report.json
            # log_capture: contents of your log buffer
            log_capture_buffer = safe_logger.get_captured_logs()

            output_html = generate_html_report(
                # TODO: Change to correct folder name - just read is not enough
                log_folder=config.get("destination", "logs"),
                report_path=args.report,
                log_capture=log_capture_buffer.getvalue().splitlines() # or your log list
            )
            safe_logger.logger.info(f"HTML summary created at {output_html}")
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
    exit(main())
    
