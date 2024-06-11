import logging
from sheen import ColoredHandler


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(log_level if rank == 0 else 'ERROR')

    console = ColoredHandler()
    formatter = logging.Formatter("[%(asctime)s %(name)s]: %(message)s", datefmt="%m/%d %H:%M:%S")
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
