#!/usr/bin/env python3
"""
Startup script for the Skin Analyzer API.

This script ensures proper logging configuration before starting uvicorn.
"""

import sys
import logging.config
import uvicorn

# Define logging configuration (same as in api.py but applied first)
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "uvicorn": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["default"],
    },
    "loggers": {
        "uvicorn": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.error": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "src": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

# Apply logging configuration BEFORE importing api module
logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting Skin Analyzer API Server")
    logger.info("=" * 60)
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        log_config=LOGGING_CONFIG,
        access_log=True,
    )

