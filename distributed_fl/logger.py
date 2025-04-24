# logger.py
import logging


def get_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Prevent adding multiple handlers if the logger is retrieved multiple times
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
