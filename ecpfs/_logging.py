from loguru import Logger
import sys

logger = Logger()

def ecp_enable_logging(
    sink=sys.stderr,
    *,
    level: str = "INFO",
    **kwargs
):
    """
    Turn on logging for my_package.

    :param sink: where to write logs (file path, stream, etc).
    :param level: minimum level to emit ("DEBUG", "INFO", …).
    :param kwargs: passed through to logger.add (format, rotation, etc).
    """
    # Avoid duplicate sinks if called twice
    # (you could store the returned sink ID if you want to allow removal later)
    logger.add(sink, level=level, **kwargs)