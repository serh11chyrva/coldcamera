import sys
from pathlib import Path


def resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, works for PyInstaller.

    :param relative_path: Relative file path
    :return: Absolute path to resource
    """

    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    return str(base_path / relative_path.lstrip("/\\"))
