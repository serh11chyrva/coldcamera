from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs
import os
import sys

pedalboard_hidden = collect_submodules("pedalboard")
pedalboard_bins = collect_dynamic_libs("pedalboard")

block_cipher = None
debug_console = os.environ.get("COLDCAMERA_DEBUG", "").lower() in {"1", "true", "yes"}
# Keep a console on Unix so launch-time failures are visible when started from a terminal.
# Windows remains windowed for normal releases; set COLDCAMERA_DEBUG=1 for diagnostics.
windowed = sys.platform == "win32" and not debug_console

a = Analysis(
    ['coldcamera/launcher.py'],
    pathex=[],
    binaries=pedalboard_bins,
    datas=[
        ('coldcamera/resources/zoom.png', 'coldcamera/resources'),
    ],
    hiddenimports=[
        "PySide6",
        "PIL",
        "numpy",
        "loguru",
        "qdarktheme",
        "moderngl",
        "cv2",
        "blend_modes",
        "pedalboard",
        "pedalboard._pedalboard",
    ] + pedalboard_hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='coldcamera',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=not windowed,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='coldcamera',
)
