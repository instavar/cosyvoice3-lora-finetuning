from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from thread_failures import BackgroundThreadFailureCapture


class BackgroundThreadFailureCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_hook = threading.excepthook
        threading.excepthook = lambda _args: None

    def tearDown(self) -> None:
        threading.excepthook = self.original_hook

    def test_captures_uncaught_worker_failure_and_restores_hook(self) -> None:
        quiet_hook = threading.excepthook
        capture = BackgroundThreadFailureCapture()

        with capture:
            worker = threading.Thread(
                name="cosyvoice-llm-worker",
                target=lambda: (_ for _ in ()).throw(RuntimeError("decoder failed")),
            )
            worker.start()
            worker.join()

        self.assertIs(threading.excepthook, quiet_hook)
        self.assertEqual(
            capture.failures,
            [
                {
                    "thread_name": "cosyvoice-llm-worker",
                    "error_type": "RuntimeError",
                    "error": "decoder failed",
                }
            ],
        )

    def test_successful_worker_leaves_empty_failure_list(self) -> None:
        with BackgroundThreadFailureCapture() as capture:
            worker = threading.Thread(target=lambda: None)
            worker.start()
            worker.join()

        self.assertEqual(capture.failures, [])


if __name__ == "__main__":
    unittest.main()
