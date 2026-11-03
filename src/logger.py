"""Journalisation traçable : chaque message indique la phase et l'étape du plan."""

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger configuré (sortie console, une seule fois par logger)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")  # accents corrects sous Windows
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_step(logger: logging.Logger, step: str, message: str) -> None:
    """Journalise un message préfixé par l'étape du plan (ex. '1.1')."""
    logger.info("[étape %s] %s", step, message)
