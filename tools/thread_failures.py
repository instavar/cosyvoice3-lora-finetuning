from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

MAX_ERROR_CHARACTERS = 1_000


class BackgroundThreadFailureCapture:
    """Capture uncaught thread failures during one isolated generation scope."""

    def __init__(self) -> None:
        self.failures: list[dict[str, str]] = []
        self._previous: Callable[[Any], None] | None = None
        self._installed: Callable[[Any], None] | None = None

    def __enter__(self) -> BackgroundThreadFailureCapture:
        self._previous = threading.excepthook

        def capture(args: Any) -> None:
            try:
                message = str(args.exc_value)
            except Exception:
                message = "exception string conversion failed"
            self.failures.append(
                {
                    "thread_name": getattr(args.thread, "name", "unknown")[
                        :MAX_ERROR_CHARACTERS
                    ],
                    "error_type": type(args.exc_value).__name__[:MAX_ERROR_CHARACTERS],
                    "error": message[:MAX_ERROR_CHARACTERS],
                }
            )
            if self._previous is not None:
                self._previous(args)

        self._installed = capture
        threading.excepthook = capture
        return self

    def __exit__(self, *_args: object) -> None:
        if self._installed is not None and threading.excepthook is self._installed:
            if self._previous is None:
                raise RuntimeError("threading excepthook disappeared during capture")
            threading.excepthook = self._previous
