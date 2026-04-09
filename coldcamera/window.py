"""
Main application window — thin UI shell.

All business logic (media loading, pipeline processing, exporting,
preset management) lives in :class:`Application`.  This module is
responsible only for:

* Building the Qt layout (menu, viewport, pipeline panel, status bar).
* Translating between NumPy arrays (the canonical processing format)
  and QImage (the display format required by the viewport).
* Showing file dialogs and forwarding user intent to :class:`Application`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QAction, QImage
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QMainWindow,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from coldcamera.config import APPLICATION_VERSION
from coldcamera.widgets.pipeline import PipelineWidget
from coldcamera.widgets.progress_dialog import ProgressDialog
from coldcamera.widgets.viewport import ViewportWidget
from coldcamera.workers import (
    FrameProcessWorker,
    GifExportWorker,
    VideoExportWorker,
)

if TYPE_CHECKING:
    from coldcamera.application import Application


# ======================================================================
# NumPy ↔ QImage helpers  (display-boundary only)
# ======================================================================


def _numpy_to_qimage(arr: np.ndarray) -> QImage:
    """
    Convert an RGBA / RGB NumPy array to a :class:`QImage`.

    :param arr: ``(H, W, 3|4)`` uint8 array.
    :return: A **copied** QImage (safe to use after the array is freed).
    :raises ValueError: If the channel count is unsupported.
    """

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    h, w = arr.shape[:2]
    ch = arr.shape[2] if arr.ndim == 3 else 1

    if ch == 4:
        fmt = QImage.Format.Format_RGBA8888
    elif ch == 3:
        fmt = QImage.Format.Format_RGB888
    else:
        raise ValueError(f"Unsupported channel count: {ch}")

    return QImage(arr.data, w, h, arr.strides[0], fmt).copy()


# ======================================================================
# MainWindow
# ======================================================================


class MainWindow(QMainWindow):
    """
    Thin Qt window for ColdCamera.

    Receives an :class:`Application` instance and delegates every
    non-UI action to it.  The **only** format conversion happening here
    is ``numpy → QImage`` when a processed frame is sent to the viewport.

    :param app: The application controller (Qt-agnostic).
    """

    def __init__(self, app: Application) -> None:
        super().__init__()
        self.app = app

        # --- Window chrome ---
        self.setWindowTitle(f"coldcamera v{APPLICATION_VERSION}")
        self.resize(1200, 800)

        self._setup_menu()
        self._setup_statusbar()
        self._setup_central_widget()

        # --- Signal wiring ---
        self.pipeline_widget.pipeline_changed.connect(self._on_pipeline_changed)
        self.viewport.frame_request.connect(self._on_frame_request)
        # frame_changed kept for legacy/compat — routes to the same handler
        self.viewport.frame_changed.connect(self._on_frame_request)

        # --- Threading setup ---
        self._process_thread: QThread | None = None
        self._process_worker: FrameProcessWorker | None = None
        self._export_thread: QThread | None = None
        self._export_worker: GifExportWorker | VideoExportWorker | None = None
        self._pending_frame_index: int | None = None

    # ==================================================================
    # UI construction
    # ==================================================================

    def _setup_menu(self) -> None:
        """Build the *File* menu."""

        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        # --- Open ---
        open_image_action = QAction("Open image...", self)
        open_image_action.triggered.connect(self._open_image)
        file_menu.addAction(open_image_action)

        open_gif_action = QAction("Open GIF... (experimental)", self)
        open_gif_action.triggered.connect(self._open_gif)
        file_menu.addAction(open_gif_action)

        open_video_action = QAction("Open video... (experimental)", self)
        open_video_action.triggered.connect(self._open_video)
        file_menu.addAction(open_video_action)

        file_menu.addSeparator()

        # --- Export ---
        export_image_action = QAction("Export image...", self)
        export_image_action.triggered.connect(self._export_image)
        file_menu.addAction(export_image_action)

        export_gif_action = QAction("Export GIF... (experimental)", self)
        export_gif_action.triggered.connect(self._export_gif)
        file_menu.addAction(export_gif_action)

        export_video_action = QAction("Export video... (experimental)", self)
        export_video_action.triggered.connect(self._export_video)
        file_menu.addAction(export_video_action)

        file_menu.addSeparator()

        # --- Presets ---
        save_preset_action = QAction("Save preset...", self)
        save_preset_action.triggered.connect(self._save_preset)
        file_menu.addAction(save_preset_action)

        load_preset_action = QAction("Load preset...", self)
        load_preset_action.triggered.connect(self._load_preset)
        file_menu.addAction(load_preset_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _setup_statusbar(self) -> None:
        statusbar = QStatusBar()
        statusbar.setStyleSheet("background-color: #191a1c;")
        self.setStatusBar(statusbar)
        self.statusBar().showMessage("Application is ready to work!")

    def _setup_central_widget(self) -> None:
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(0)

        # Pipeline panel — receives the Application's pipeline reference
        self.pipeline_widget = PipelineWidget(self.app.pipeline, self)
        pipeline_frame = QFrame()
        pipeline_frame.setFrameShape(QFrame.Shape.StyledPanel)
        pipeline_layout = QVBoxLayout(pipeline_frame)
        pipeline_layout.setContentsMargins(0, 0, 0, 0)
        pipeline_layout.addWidget(self.pipeline_widget)

        # Viewport panel
        viewport_frame = QFrame()
        viewport_frame.setFrameShape(QFrame.Shape.StyledPanel)
        viewport_layout = QVBoxLayout(viewport_frame)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        self.viewport = ViewportWidget(self)
        viewport_layout.addWidget(self.viewport)

        splitter.addWidget(pipeline_frame)
        splitter.addWidget(viewport_frame)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)

        container_layout.addWidget(splitter)
        self.setCentralWidget(container)

    # ==================================================================
    # Signal handlers  (pipeline / viewport → process & display)
    # ==================================================================

    def _on_pipeline_changed(self) -> None:
        """Re-process and display when any pipeline parameter changes."""
        self._process_and_display()

    def _on_frame_request(self, frame_index: int) -> None:
        """Re-process and display for the requested *frame_index*."""
        self._process_and_display(frame_index)

    # ==================================================================
    # Core display loop
    # ==================================================================

    def _process_and_display(self, frame_index: int | None = None) -> None:
        """
        Ask :class:`Application` to process the current (or given) frame
        in a background thread, then update the viewport when complete.
        """
        # If a thread is already running, queue this request
        if self._process_thread is not None and self._process_thread.isRunning():
            self._pending_frame_index = frame_index
            return

        if frame_index is not None:
            self.app.current_frame_index = frame_index

        original = self.app.get_original_frame(self.app.current_frame_index)
        if original is None:
            return

        # Create worker and thread
        self._process_worker = FrameProcessWorker()
        self._process_worker.set_data(
            self.app.pipeline,
            original,
            self.app.current_frame_index,
        )

        self._process_thread = QThread()
        self._process_worker.moveToThread(self._process_thread)

        # Connect signals
        self._process_thread.started.connect(self._process_worker.process)
        self._process_worker.finished.connect(self._on_frame_processed)
        self._process_worker.error.connect(self._on_process_error)
        self._process_worker.finished.connect(self._process_thread.quit)
        self._process_worker.error.connect(self._process_thread.quit)
        self._process_thread.finished.connect(self._cleanup_process_thread)

        # Start processing
        self._process_thread.start()

    def _on_frame_processed(self, original: np.ndarray, processed: np.ndarray, frame_index: int) -> None:
        """Handle completed frame processing."""
        orig_qimg = _numpy_to_qimage(original)
        proc_qimg = _numpy_to_qimage(processed)

        # Store both versions so the viewport's "View original" button works.
        if self.app.original_image is not None:
            # Single-image mode
            self.viewport.original_qimage = orig_qimg
        else:
            # GIF / video mode — per-frame original
            self.viewport.original_frame_qimg = orig_qimg

        self.viewport.processed_qimage = proc_qimg

        if self.viewport.showing_original:
            self.viewport.set_qimage(orig_qimg)
        else:
            self.viewport.set_qimage(proc_qimg)

    def _on_process_error(self, error_msg: str) -> None:
        """Handle processing error."""
        self.statusBar().showMessage(f"Processing error: {error_msg}")

    def _cleanup_process_thread(self) -> None:
        """Clean up the processing thread."""
        if self._process_worker:
            self._process_worker.deleteLater()
            self._process_worker = None
        if self._process_thread:
            self._process_thread.deleteLater()
            self._process_thread = None

        # Process any pending frame request
        if self._pending_frame_index is not None:
            pending = self._pending_frame_index
            self._pending_frame_index = None
            self._process_and_display(pending)

    # ==================================================================
    # Open actions
    # ==================================================================

    def _open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return

        arr = self.app.open_image(path)
        orig_qimg = _numpy_to_qimage(arr)
        self.viewport.original_qimage = orig_qimg

        # Show the (possibly pipeline-processed) result immediately
        self._process_and_display()
        self.statusBar().showMessage(f"Image opened: {path}")

    def _open_gif(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open GIF", "", "GIF (*.gif)")
        if not path:
            return

        frames, fps = self.app.open_gif(path)
        self.viewport.set_playback(len(frames), fps)
        self.statusBar().showMessage(f"GIF opened: {path}")

    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", "Videos (*.mp4 *.avi *.mov)")
        if not path:
            return

        frame_count, fps = self.app.open_video(path)
        self.viewport.set_playback(frame_count, fps)
        self.statusBar().showMessage(f"Video opened: {path}")

    # ==================================================================
    # Export actions
    # ==================================================================

    def _export_image(self) -> None:
        if self.app.original_image is None:
            self.statusBar().showMessage("No processed image to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save image",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)",
        )
        if not path:
            return

        try:
            self.app.export_image(path)
            self.statusBar().showMessage(f"Image exported: {path}")
        except Exception as exc:
            self.statusBar().showMessage(f"Export failed: {exc}")

    def _export_gif(self) -> None:
        if not self.app.original_frames:
            self.statusBar().showMessage("No GIF to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save GIF", "", "GIF (*.gif)")
        if not path:
            return

        # Create progress dialog
        progress_dialog = ProgressDialog("Exporting GIF", self)

        # Create worker and thread
        self._export_worker = GifExportWorker()
        self._export_worker.set_data(
            self.app.original_frames,
            self.app.pipeline,
            self.app.frames_fps,
            path,
        )

        self._export_thread = QThread()
        self._export_worker.moveToThread(self._export_thread)

        # Connect signals
        self._export_thread.started.connect(self._export_worker.export)
        self._export_worker.progress.connect(progress_dialog.set_progress)
        self._export_worker.finished.connect(lambda p: self._on_export_finished(progress_dialog, p, "GIF"))
        self._export_worker.error.connect(lambda e: self._on_export_error(progress_dialog, e))
        self._export_worker.finished.connect(self._export_thread.quit)
        self._export_worker.error.connect(self._export_thread.quit)
        self._export_thread.finished.connect(self._cleanup_export_thread)

        # Connect cancel button
        progress_dialog.cancelled.connect(self._export_worker.stop)

        # Start export
        self._export_thread.start()
        progress_dialog.exec()

    def _export_video(self) -> None:
        if not self.app.video_provider:
            self.statusBar().showMessage("No video to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save video", "", "MP4 (*.mp4);;AVI (*.avi)")
        if not path:
            return

        # Create progress dialog
        progress_dialog = ProgressDialog("Exporting Video", self)

        # Create worker and thread
        self._export_worker = VideoExportWorker()
        self._export_worker.set_data(
            self.app.video_provider,
            self.app.pipeline,
            path,
        )

        self._export_thread = QThread()
        self._export_worker.moveToThread(self._export_thread)

        # Connect signals
        self._export_thread.started.connect(self._export_worker.export)
        self._export_worker.progress.connect(progress_dialog.set_progress)
        self._export_worker.finished.connect(lambda p: self._on_export_finished(progress_dialog, p, "Video"))
        self._export_worker.error.connect(lambda e: self._on_export_error(progress_dialog, e))
        self._export_worker.finished.connect(self._export_thread.quit)
        self._export_worker.error.connect(self._export_thread.quit)
        self._export_thread.finished.connect(self._cleanup_export_thread)

        # Connect cancel button
        progress_dialog.cancelled.connect(self._export_worker.stop)

        # Start export
        self._export_thread.start()
        progress_dialog.exec()

    def _on_export_finished(self, dialog: ProgressDialog, path: str, media_type: str) -> None:
        """Handle successful export completion."""
        dialog.mark_complete()
        self.statusBar().showMessage(f"{media_type} exported: {path}")

    def _on_export_error(self, dialog: ProgressDialog, error_msg: str) -> None:
        """Handle export error."""
        dialog.mark_error(error_msg)
        self.statusBar().showMessage(f"Export failed: {error_msg}")

    def _cleanup_export_thread(self) -> None:
        """Clean up the export thread."""
        if self._export_worker:
            self._export_worker.deleteLater()
            self._export_worker = None
        if self._export_thread:
            self._export_thread.deleteLater()
            self._export_thread = None

    # ==================================================================
    # Preset actions
    # ==================================================================

    def _save_preset(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save preset", "", "JSON (*.json)")
        if not path:
            return

        try:
            self.app.save_preset(path)
            self.statusBar().showMessage(f"Preset saved: {path}")
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to save preset: {exc}")

    def _load_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load preset", "", "JSON (*.json)")
        if not path:
            return

        try:
            pipeline = self.app.load_preset(path)
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to load preset: {exc}")
            return

        # Rebuild the pipeline widget UI to reflect the new effects
        self.pipeline_widget.load_pipeline(pipeline)
        self._process_and_display()
        self.statusBar().showMessage(f"Preset loaded: {path}")

    # ==================================================================
    # Cleanup
    # ==================================================================

    def closeEvent(self, event) -> None:
        """Clean up threads before closing."""
        # Stop any running processing
        if self._process_worker:
            self._process_worker.stop()
        if self._process_thread and self._process_thread.isRunning():
            self._process_thread.quit()
            self._process_thread.wait()

        # Stop any running export
        if self._export_worker:
            if isinstance(self._export_worker, GifExportWorker):
                self._export_worker.stop()
            elif isinstance(self._export_worker, VideoExportWorker):
                self._export_worker.stop()
        if self._export_thread and self._export_thread.isRunning():
            self._export_thread.quit()
            self._export_thread.wait()

        event.accept()
