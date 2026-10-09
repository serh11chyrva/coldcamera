from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QVBoxLayout

from coldcamera.core.processing_settings import ProcessingBackend


class ProcessingSettingsDialog(QDialog):
    """Machine-local CPU/GPU preference dialog."""

    def __init__(self, backend: ProcessingBackend, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Processing settings")
        self.setModal(True)

        self.backend_combo = QComboBox(self)
        self.backend_combo.addItem("Auto", ProcessingBackend.AUTO.value)
        self.backend_combo.addItem("CPU", ProcessingBackend.CPU.value)
        self.backend_combo.addItem("GPU preferred", ProcessingBackend.GPU.value)
        index = self.backend_combo.findData(backend.value)
        self.backend_combo.setCurrentIndex(max(0, index))

        description = QLabel(
            "Auto uses CPU below roughly 0.26 megapixels and prefers implemented GPU effects above that size. "
            "GPU preferred falls back to CPU for unsupported effects or if GL initialization fails.\n"
            "Intermediate image cache: 512 MiB. This preference is stored on this computer and is not included in presets."
        )
        description.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Processing backend", self.backend_combo)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(description)
        layout.addWidget(buttons)

    @property
    def backend(self) -> ProcessingBackend:
        return ProcessingBackend(self.backend_combo.currentData())
