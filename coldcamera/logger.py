from loguru import logger as _logger

from coldcamera.utils.local_path import get_user_local_directory

logger = _logger


DEFAULT_LOG_DIRECTORY = get_user_local_directory() / "logs"
DEFAULT_LOG_PATH = str(DEFAULT_LOG_DIRECTORY / "log_{time}.log")
DEFAULT_LOG_FORMAT = "{time:HH:mm:ss.SS} ({file}) [{level}] {message} {exception}"
_file_sink_id = None


def initialize_logger():
    """Initialize the logger with the default log path and format."""

    global logger, _file_sink_id

    if _file_sink_id is None:
        DEFAULT_LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
        _file_sink_id = logger.add(
            DEFAULT_LOG_PATH,
            format=DEFAULT_LOG_FORMAT,
            colorize=False,
            catch=True,
            backtrace=True,
            diagnose=False,
        )
