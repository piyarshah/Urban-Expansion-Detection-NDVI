"""
logger.py — Pipeline Run Logger

Sets up a file-based logger for the full pipeline run.
Call setup_logger() once at pipeline start; then use logging.info() anywhere.

Public API:
    setup_logger(log_path)           → logging.Logger
    log_pipeline_start(logger, cfg)  → None
    log_stage(logger, stage, msg)    → None
"""

import logging
from pathlib import Path
from datetime import datetime


def setup_logger(log_path: str = "outputs/logs/run.log") -> logging.Logger:
    """
    Configure and return the pipeline logger.

    Creates parent directories automatically. Each run appends to the same
    log file so history is preserved across runs. WARNING+ messages are also
    echoed to the console; DEBUG/INFO stay file-only to keep stdout clean.

    Parameters
    ----------
    log_path : destination log file path

    Returns
    -------
    logger : logging.Logger named "pipeline"
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers if called more than once in a session
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — captures DEBUG and above
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler — WARNING and above only
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info("=" * 60)
    logger.info("Pipeline run started  %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    return logger


def log_pipeline_start(logger: logging.Logger, config: dict) -> None:
    """
    Log configuration values at pipeline startup.

    Parameters
    ----------
    logger : logger from setup_logger
    config : dict of key configuration values, e.g.:
             {"confidence_threshold": 0.60, "n_estimators": 300, ...}
    """
    logger.info("--- Configuration ---")
    for key, value in config.items():
        logger.info("  %-30s %s", key, value)


def log_stage(logger: logging.Logger, stage: str, message: str, **kwargs) -> None:
    """
    Log a stage completion event with optional structured metadata.

    Parameters
    ----------
    logger  : logger from setup_logger
    stage   : stage identifier, e.g. "S13"
    message : human-readable summary line
    **kwargs: key-value pairs appended to the log line, e.g. pixels=12345
    """
    extras = "  ".join(f"{k}={v}" for k, v in kwargs.items())
    if extras:
        logger.info("[%s] %s  |  %s", stage, message, extras)
    else:
        logger.info("[%s] %s", stage, message)
