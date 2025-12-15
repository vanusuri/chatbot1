import logging
from config.settings import settings

LOGGER_NAME = "banking_support_ai"

logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
