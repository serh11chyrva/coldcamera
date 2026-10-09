import os
from pathlib import Path

DEFAULT_PATHS: dict[str, dict[str, str]] = {
    "nt": {"application": "coldcamera"},
    "posix": {"application": "coldcamera"},
}


def get_user_local_directory(paths: dict[str, dict[str, str]] = DEFAULT_PATHS) -> Path:
    """
    Get the user's local directory for application data.

    This function determines the user's local
    application directory based on the operating system
    and returns a pathlib.Path object pointing to that directory.

    :return: Path to the user's local application directory.
    """

    path: Path

    # Determine the user's local application directory
    if os.name == "posix":
        path = Path.home() / f".{paths[os.name]['application']}"
    elif os.name == "nt":
        # LOCALAPPDATA is available in packaged app sessions and does not depend on
        # os.getlogin(), which can fail when launched from a service or CI account.
        local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        path = local_app_data / paths[os.name]["application"]
    else:
        path = Path("./")

    # Create the log directory if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)

    return path
