from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

import numpy as np

from coldcamera.core.operations import PreviewTile
from coldcamera.workers import QtTaskRunner


class QtTaskRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_backend_result_and_progress_are_delivered(self) -> None:
        runner = QtTaskRunner()
        loop = QEventLoop()
        started_values = []
        result_values = []
        progress_values = []
        runner.started.connect(started_values.append)
        runner.result.connect(lambda task_id, value: result_values.append((task_id, value)))
        runner.progress.connect(lambda task_id, current, total: progress_values.append((task_id, current, total)))
        runner.finished.connect(lambda _task_id: loop.quit())

        runner.submit("sample", lambda _token, progress: (progress(1, 1), "done")[1])
        QTimer.singleShot(5000, loop.quit)
        loop.exec()

        self.assertEqual(started_values, ["sample"])
        self.assertEqual(result_values, [("sample", "done")])
        self.assertEqual(progress_values, [("sample", 1, 1)])

    def test_preview_tiles_are_delivered_from_background_tasks(self) -> None:
        runner = QtTaskRunner()
        loop = QEventLoop()
        received = []
        runner.preview.connect(lambda task_id, tile: received.append((task_id, tile)))
        runner.finished.connect(lambda _task_id: loop.quit())
        tile = PreviewTile(0, 0, 2, 1, np.full((1, 2, 4), 90, dtype=np.uint8))

        runner.submit("preview", lambda _token, reporter: reporter.preview(tile))
        QTimer.singleShot(5000, loop.quit)
        loop.exec()

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0][0], "preview")
        np.testing.assert_array_equal(received[0][1].pixels, tile.pixels)


if __name__ == "__main__":
    unittest.main()
