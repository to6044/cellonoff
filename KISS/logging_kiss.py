import logging
from pathlib import Path

def get_logger(logger_name, logfile_path, log_level=logging.INFO):
    """
    Create and return a logger object.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Set up file handler to write all logging messages to the same file
    log_file_path = Path(logfile_path).with_suffix(".log")

    # Create the log file if it doesn't exist, including any parent directories
    if log_file_path.exists() == False:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        log_file_path.touch()
    
    # Set up file handler to write logging messages to the log file
    handler = logging.FileHandler(log_file_path, mode="a")
    handler.setLevel(log_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Set up stream handler to write logging messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger