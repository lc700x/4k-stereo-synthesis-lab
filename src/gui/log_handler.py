"""Thread-safe logging handler for the Flet GUI log panel."""
from __future__ import annotations

import logging
import queue
from collections import deque


class GuiLogHandler(logging.Handler):
    """Queue structured log records for safe polling from the Flet UI loop."""

    def __init__(self, maxlen: int = 2000):
        super().__init__()
        self.queue: queue.Queue[tuple[int, str, str, str]] = queue.Queue()
        self.cache: deque[tuple[int, str, str, str]] = deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            formatter = self.formatter or logging.Formatter("%(message)s")
            asctime = formatter.formatTime(record, formatter.datefmt or "%H:%M:%S")
            formatted = self.format(record)
            item = (record.levelno, record.name, asctime, formatted)
            self.queue.put(item)
            self.cache.append(item)
        except Exception:
            self.handleError(record)
