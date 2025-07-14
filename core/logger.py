import logging
import logging.handlers
import multiprocessing
import threading
import queue
import colorlog

class FlushSignal:
    pass

class SafeLogger:
    def __init__(self, name='safe_remote_ops', log_level=logging.DEBUG):
        self.name = name
        self.log_level = log_level
        self.setup_logger()

        self._manager = multiprocessing.Manager()
        self._log_queue = self._manager.Queue()

        self._log_thread = None
        self._stop_logging = multiprocessing.Event()

    def setup_logger(self):
        formatter = colorlog.ColoredFormatter(
            '%(asctime)s | %(log_color)s%(levelname)-8s | %(name)-15s | %(message)s%(reset)s',
            datefmt='%H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )

        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(self.log_level)

        file_handler = logging.handlers.RotatingFileHandler(
            'safe_remote_ops.log',
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s'
        ))
        file_handler.setLevel(self.log_level)

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.log_level)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False

    def start_log_listener(self):
        self._log_thread = threading.Thread(target=self._log_listener, daemon=True)
        self._log_thread.start()

    def stop_log_listener(self):
        self._stop_logging.set()
        if self._log_thread:
            self._log_thread.join(timeout=5)
        if hasattr(self, "_manager"):
            self._manager.shutdown()

    def _log_listener(self):
        while not self._stop_logging.is_set():
            try:
                record = self._log_queue.get(timeout=1)
                if isinstance(record, FlushSignal):
                    continue
                if record is None:
                    break
                record.name = self.logger.name
                self.logger.handle(record)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Logging error: {e}")

    def get_process_logger(self, process_name):
        return ProcessLogger(self._log_queue, process_name, min_level=self.log_level)

class ProcessLogger:
    def __init__(self, log_queue, process_name, min_level=logging.DEBUG):
        self.log_queue = log_queue
        self.process_name = process_name
        self.min_level = min_level
        self.buffer = []

    def _log(self, level, message):
        if level < self.min_level:
            return
        record = logging.LogRecord(
            name=self.process_name,
            level=level,
            pathname='',
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        self.buffer.append(record)

    def flush(self):
        for record in self.buffer:
            try:
                self.log_queue.put(record)
            except Exception:
                pass
        self.buffer.clear()
        try:
            self.log_queue.put(FlushSignal())
        except Exception:
            pass

    def debug(self, message):
        self._log(logging.DEBUG, message)

    def info(self, message):
        self._log(logging.INFO, message)

    def warning(self, message):
        self._log(logging.WARNING, message)

    def error(self, message):
        self._log(logging.ERROR, message)

    def critical(self, message):
        self._log(logging.CRITICAL, message)
