import logging
import colorlog

def configure_default_logger():
    LOG_FORMAT = (
        "%(log_color)s%(levelname)-8s%(reset)s "
        "%(name)s   %(message)s"
    )

    LOG_COLORS = {
        'DEBUG': 'green',
        'INFO': 'bold_white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }

    formatter = colorlog.ColoredFormatter(LOG_FORMAT, log_colors=LOG_COLORS)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger('workspace')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger
