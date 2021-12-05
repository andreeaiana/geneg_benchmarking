import logging
import sys


# define setup rules
def setup_logging(name: str = __name__, log_level: str = None) -> logging.Logger:
    """Setup basic logging

    Args:
        name: string variable
        log_level (str): minimum loglevel for emitting messages

    Returns:
        logger object

    """
    # define logger
    logger = logging.getLogger(name)

    # determine log_level from string input
    if log_level in ["info", "debug"]:

        if log_level == "info":

            log_level_int = logging.INFO

        else:

            log_level_int = logging.DEBUG

    else:

        log_level_int = logging.INFO

    logformat = "[%(asctime)s]\t%(levelname)s:\t%(name)s:\t%(message)s"
    logging.basicConfig(
        level=log_level_int,
        stream=sys.stdout,
        format=logformat,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    return logger
