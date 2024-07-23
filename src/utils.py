import json
import logging
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path


def setup_logger(log_file: Path):
    """Setup logger to write logs to a specified file."""
    logger = logging.getLogger("action_label_processor")
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


def log_to_file(log_file: Path):
    """Decorator to log function calls to a specified log file."""
    logger = setup_logger(log_file)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
            try:
                result = func(*args, **kwargs)
            except Exception:
                logger.exception(f"Error in {func.__name__}")
                raise
            else:
                logger.debug(f"{func.__name__} returned {result}")
                return result

        return wrapper

    return decorator


def save_json_with_timestamp(data: dict, output_dir: Path, base_filename: str) -> None:
    """Save the given data to a JSON file with a timestamp in the filename.

    Args:
        data (dict): The data to save in the JSON file.
        output_dir (Path): The directory where the JSON file will be saved.
        base_filename (str): The base name for the JSON file.
    """
    jst = timezone(timedelta(hours=9))
    timestamp = datetime.now(tz=jst).strftime("%Y%m%d_%H%M%S")
    output_filename = f"{base_filename}_{timestamp}.json"
    output_path = output_dir / output_filename
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Processed labels saved to {output_path}")
    except OSError as e:
        print(f"Error saving processed labels: {e}")
