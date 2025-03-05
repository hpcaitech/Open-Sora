import logging
import os

import torch.distributed as dist


def is_distributed() -> bool:
    """
    Check if the code is running in a distributed setting.

    Returns:
        bool: True if running in a distributed setting, False otherwise
    """
    return os.environ.get("WORLD_SIZE", None) is not None


def is_main_process() -> bool:
    """
    Check if the current process is the main process.

    Returns:
        bool: True if the current process is the main process, False otherwise.
    """
    return not is_distributed() or dist.get_rank() == 0


def get_world_size() -> int:
    """
    Get the number of processes in the distributed setting.

    Returns:
        int: The number of processes.
    """
    if is_distributed():
        return dist.get_world_size()
    else:
        return 1


def create_logger(logging_dir: str = None) -> logging.Logger:
    """
    Create a logger that writes to a log file and stdout. Only the main process logs.

    Args:
        logging_dir (str): The directory to save the log file.

    Returns:
        logging.Logger: The logger.
    """
    if is_main_process():
        additional_args = dict()
        if logging_dir is not None:
            additional_args["handlers"] = [
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ]
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            **additional_args,
        )
        logger = logging.getLogger(__name__)
        if logging_dir is not None:
            logger.info("Experiment directory created at %s", logging_dir)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def log_message(*args, level: str = "info"):
    """
    Log a message to the logger.

    Args:
        *args: The message to log.
        level (str): The logging level.
    """
    logger = logging.getLogger(__name__)
    if level == "info":
        logger.info(*args)
    elif level == "warning":
        logger.warning(*args)
    elif level == "error":
        logger.error(*args)
    elif level == "print":
        print(*args)
    else:
        raise ValueError(f"Invalid logging level: {level}")
