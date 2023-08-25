import logging
import os

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    level = int(os.environ["LINPYK_LOGGING"])
    if level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(level)
except KeyError:
    pass

__version__ = "0.1.0"
