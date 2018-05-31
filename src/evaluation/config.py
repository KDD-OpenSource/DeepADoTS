import time
import os
import sys
import logging

# Use: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
LOG_LEVEL = logging.DEBUG


def init_logging():
    # Prepare directory and file path for storing the logs
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_files_dir = os.path.join('reports', 'logs')
    log_file_path = os.path.join(log_files_dir, '{}.log'.format(timestamp))
    os.makedirs(log_files_dir, exist_ok=True)

    # Actually initialize the logging module
    logFormatter = logging.Formatter(fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                                     datefmt='%Y-%m-%d %H:%M:%S')
    rootLogger = logging.getLogger()
    rootLogger.setLevel(LOG_LEVEL)

    # Store logs in a log file in reports/logs
    fileHandler = logging.FileHandler(log_file_path)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # Also print logs in the standard output
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    # Create logger instance for the config file
    logger = logging.getLogger(__name__)
    logger.debug("Logger initialized")
