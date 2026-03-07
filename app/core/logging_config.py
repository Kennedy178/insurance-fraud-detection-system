# app/core/logging_config.py
"""
Logging configuration using loguru.
- Development: coloured human-readable output to console
- Production:  JSON structured logs to rotating files
All modules do: from loguru import logger  — no further setup needed.
"""

import sys
import os
from loguru import logger
from app.core.config import settings


def setup_logging() -> None:
    """
    Called once from app/main.py on startup.
    Removes loguru's default handler and replaces with our config.
    """
    # Remove default loguru handler
    logger.remove()

    # ── Console handler ────────────────────────────────────────
    if settings.ENVIRONMENT == "development":
        # Coloured, human-readable for local dev
        logger.add(
            sys.stdout,
            level=settings.LOG_LEVEL,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            colorize=True,
        )
    else:
        # JSON structured logs for production (Render, Docker)
        logger.add(
            sys.stdout,
            level=settings.LOG_LEVEL,
            serialize=True,  # outputs JSON
        )

    # ── File handler — always on ───────────────────────────────
    os.makedirs(settings.LOG_DIR, exist_ok=True)

    # General application log
    logger.add(
        os.path.join(settings.LOG_DIR, "app.log"),
        level=settings.LOG_LEVEL,
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        serialize=True,  # JSON in files regardless of environment
        enqueue=True,    # thread-safe async writing
    )

    # Separate error log — only ERROR and above
    logger.add(
        os.path.join(settings.LOG_DIR, "errors.log"),
        level="ERROR",
        rotation="100 MB",
        retention="90 days",
        serialize=True,
        enqueue=True,
    )

    # Prediction audit log — every inference logged here
    logger.add(
        os.path.join(settings.LOG_DIR, "predictions.log"),
        level="INFO",
        rotation="500 MB",
        retention="90 days",
        serialize=True,
        enqueue=True,
        filter=lambda record: "prediction" in record["extra"],
    )

    logger.info(
        f"Logging configured | env={settings.ENVIRONMENT} | "
        f"level={settings.LOG_LEVEL} | log_dir={settings.LOG_DIR}"
    )