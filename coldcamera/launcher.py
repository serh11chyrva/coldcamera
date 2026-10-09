"""Minimal frozen-app entry point that records failures before app imports."""

from __future__ import annotations

import logging

from coldcamera.utils.local_path import get_user_local_directory


def main() -> None:
    log_directory = get_user_local_directory() / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_directory / "startup.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        encoding="utf-8",
        force=True,
    )
    logging.info("Starting Coldcamera")

    try:
        from coldcamera.application import run_gui

        run_gui()
    except Exception:
        logging.exception("Application startup failed")
        raise


if __name__ == "__main__":
    main()
