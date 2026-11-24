from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(level: str = "INFO") -> None:
    log_format = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    date_format = "%H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    log_dir = Path("data/logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "gateway.log", encoding="utf-8"))
    except OSError:
        pass

    logging.basicConfig(
        level=level.upper(),
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
