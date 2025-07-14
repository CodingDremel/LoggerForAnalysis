# integration_examples.py
# Demonstrates end-to-end usage of the system

from config.parser import ConfigParser
from core.logger import SafeLogger
from core.operations import SafeRemoteOperations

def run_demo():
    config_path = "safe_remote_ops/data/config.json"
    parser = ConfigParser(config_path)
    config = parser.load_config()

    logger = SafeLogger()
    logger.start_log_listener()
    process_logger = logger.get_process_logger("Demo")

    ops = SafeRemoteOperations(config, logger=process_logger)

    for name, computer in config["computers"].items():
        print(f"Running safe_copy_files on {name}")
        result = ops.safe_copy_files(computer)
        print(result)

    logger.stop_log_listener()

if __name__ == "__main__":
    run_demo()
