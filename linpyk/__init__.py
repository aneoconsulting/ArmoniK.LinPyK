import logging
import os


def set_logger_handlers(level: int) -> None:
    """
    Set the logging level and activates the display of logs in the console.

    Parameters
    ----------
    level : int
        Logging level.

    """
    global logger
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(level)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    level = os.environ["LINPYK_LOGGING"]
    match level:
        case "DEBUG":
            set_logger_handlers(logging.DEBUG)
        case "INFO":
            set_logger_handlers(logging.INFO)
        case "WARNING":
            set_logger_handlers(logging.WARNING)
        case "ERROR":
            set_logger_handlers(logging.ERROR)
        case "CRITICAL":
            set_logger_handlers(logging.CRITICAL)
        case _:
            try:
                level = int(level)
                if level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
                    set_logger_handlers(level)
            except ValueError:
                pass
            set_logger_handlers("INFO")
except KeyError:
    pass

__version__ = "0.1.0"
