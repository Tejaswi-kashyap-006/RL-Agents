"""Shared logger setup. Fail loudly: everything at INFO and up, to stderr."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger; safe to call repeatedly."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
