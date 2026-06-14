"""Structured logging configuration.

A single ``configure_logging`` call sets up structlog with a console renderer.
All modules obtain a logger via ``get_logger(__name__)``.
"""

from __future__ import annotations

import logging

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Initialise structlog + stdlib logging once at program start."""
    logging.basicConfig(format="%(message)s", level=getattr(logging, level.upper()))
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)
