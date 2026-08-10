# traitly/utils/logs.py
import logging

def setup_logging(
    level=logging.INFO,
    log_file="traitly.log",
    to_console=False
):
    # With mode = w, rewrite the file if it exist
    handlers = [logging.FileHandler(log_file, mode="w")]

    if to_console:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

    logging.getLogger(__name__).info("Logging saved in: %s", log_file)
