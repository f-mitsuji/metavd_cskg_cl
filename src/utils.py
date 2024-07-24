import datetime
import json
import logging
from functools import wraps
from pathlib import Path


def setup_logger(logger_name: str, log_file: Path, level=logging.DEBUG) -> logging.Logger:
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    return logger


def log_to_file(logger: logging.Logger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
            try:
                result = func(*args, **kwargs)
            except Exception:
                logger.exception(f"Exception in function {func.__name__}")
                raise
            else:
                logger.debug(f"{func.__name__} returned {result}")
                return result

        return wrapper

    return decorator


def save_json_with_timestamp(data: dict, output_dir: Path, base_filename: str) -> None:
    timestamp = get_current_jst_timestamp()
    output_filename = f"{base_filename}_{timestamp}.json"
    output_path = output_dir / output_filename
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"saved to {output_path}")
    except OSError as e:
        print(f"Error saving JSON to {output_path}: {e}")
        raise


def get_current_jst_timestamp() -> str:
    now_utc = datetime.datetime.now(tz=datetime.UTC)
    jst = datetime.timezone(datetime.timedelta(hours=9))
    now_jst = now_utc.astimezone(jst)
    return now_jst.strftime("%Y%m%d_%H%M%S")
