import datetime
import json
import logging
from functools import wraps
from pathlib import Path

JST = datetime.timezone(datetime.timedelta(hours=9))


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


def load_json(file_path):
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except OSError as e:
        print(f"Error loading JSON from {file_path}: {e}")
        raise


def get_current_jst_timestamp() -> str:
    now_utc = datetime.datetime.now(tz=datetime.UTC)
    now_jst = now_utc.astimezone(JST)
    return now_jst.strftime("%Y%m%d_%H%M%S")


def get_latest_file_path(directory: Path, prefix: str) -> Path:
    files = [f for f in directory.iterdir() if f.is_file() and f.name.startswith(prefix) and f.name.endswith(".json")]
    if not files:
        msg = f"No files found in {directory} with prefix {prefix}"
        raise FileNotFoundError(msg)

    files_with_timestamps = [
        (f, datetime.datetime.strptime(f.name[len(prefix) : -5], "%Y%m%d_%H%M%S").replace(tzinfo=JST)) for f in files
    ]

    return max(files_with_timestamps, key=lambda x: x[1])[0]
