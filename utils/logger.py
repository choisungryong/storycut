import logging
import sys
import os


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"storycut.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s [%(name)s] %(levelname)s %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
        level = os.getenv("LOG_LEVEL", "DEBUG").upper()
        logger.setLevel(getattr(logging, level, logging.DEBUG))
    return logger
