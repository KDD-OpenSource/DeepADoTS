import logging
import os
import sys

import re
import time

# Use: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
LOG_LEVEL = logging.DEBUG
CONSOLE_LOG_LEVEL = logging.INFO


def init_logging(output_dir='reports/logs'):
    # Prepare directory and file path for storing the logs
    timestamp = time.strftime('%Y-%m-%d-%H%M%S')
    log_file_path = os.path.join(output_dir, '{}.log'.format(timestamp))
    os.makedirs(output_dir, exist_ok=True)

    # Actually initialize the logging module
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)
    # Removes previous handlers (required for running pipeline multiple times)
    root_logger.handlers = []

    # Store logs in a log file in reports/logs
    file_handler = logging.FileHandler(log_file_path)  # mode='w'
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # Also print logs in the standard output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(CONSOLE_LOG_LEVEL)
    console_handler.addFilter(DebugModuleFilter(['^src\.', '^root$']))
    root_logger.addHandler(console_handler)

    # Create logger instance for the config file
    logger = logging.getLogger(__name__)
    logger.debug('Logger initialized')


class DebugModuleFilter(logging.Filter):
    def __init__(self, pattern=[]):
        logging.Filter.__init__(self)
        self.module_pattern = [re.compile(x) for x in pattern]

    def filter(self, record):
        # This filter assumes that we want INFO logging from all
        # modules and DEBUG logging from only selected ones, but
        # easily could be adapted for other policies.
        if record.levelno == logging.DEBUG:
            # e.g. src.evaluator.evaluation
            return any([x.match(record.name) for x in self.module_pattern])
        return True
