import argparse
import re
from pathlib import Path

import pandas as pd
import spacy

from src.settings import LOGS_DIR, METAVD_DIR, RESULTS_DIR, STOP_WORDS
from src.utils import get_current_jst_timestamp, log_to_file, save_json_with_timestamp, setup_logger

timestamp = get_current_jst_timestamp()
LOGS_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOGS_DIR / f"lemmatize_labels_{timestamp}.log"
logger = setup_logger("lemmatize labels", log_file)

nlp = spacy.load("en_core_web_lg")

CUSTOM_LEMMAS = {"opening": "open", "closing": "close", "fixing": "fix", "welding": "weld", "diving": "dive"}


def get_dataset_info(dataset: str) -> dict:
    METAVD_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = METAVD_DIR / f"{dataset}_classes.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_dir = RESULTS_DIR / dataset
    result_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        msg = f"CSV file for dataset '{dataset}' does not exist at {csv_path}"
        raise ValueError(msg)

    return {"csv_path": csv_path, "result_dir": result_dir}


def load_labels(file_path: Path) -> list:
    classes_df = pd.read_csv(file_path)
    return classes_df.iloc[:, 1].tolist()


@log_to_file(logger)
def normalize_label(label: str) -> str:
    if label.isupper():
        return label.lower()

    label = re.sub(r"\([^)]*\)", "", label)
    label = re.sub(r"(?<!^)(?=[A-Z])", " ", label)
    label = re.sub(r"[_\-/]", " ", label)
    return " ".join(label.lower().split())


@log_to_file(logger)
def lemmatize_and_filter_token(token_text: str) -> str:
    if token_text in CUSTOM_LEMMAS:
        return CUSTOM_LEMMAS[token_text]

    doc = nlp(token_text)
    if len(doc) == 0:
        return ""
    token = doc[0]
    if token.lemma_ not in STOP_WORDS and not token.is_punct:
        return token.lemma_
    return ""


@log_to_file(logger)
def lemmatize_labels(labels: list) -> dict:
    lemmatized_data = {}
    for raw_label in labels:
        normalized_label = normalize_label(raw_label)
        lemmatized_tokens = set()
        for token in normalized_label.split():
            lemmatized_token = lemmatize_and_filter_token(token)
            if lemmatized_token:
                lemmatized_tokens.add(lemmatized_token)
        lemmatized_data[raw_label] = list(lemmatized_tokens)
    return lemmatized_data


def main(dataset: str) -> None:
    try:
        logger.info(f"Started processing dataset: {dataset}")
        dataset_info = get_dataset_info(dataset)
        labels = load_labels(dataset_info["csv_path"])
        lemmatized_labels = lemmatize_labels(labels)
        save_json_with_timestamp(lemmatized_labels, dataset_info["result_dir"], f"{dataset}_lemmatized_labels")
        logger.info(f"Finished processing dataset: {dataset}")
    except Exception:
        logger.exception("An error occurred")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process action labels for a specified dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="*",
        choices=["activitynet", "charades", "hmdb51", "kinetics700", "stair_actions", "ucf101"],
        help="The name(s) of the dataset(s) to process. If not specified, all datasets will be processed.",
    )
    args = parser.parse_args()

    datasets = (
        args.dataset
        if args.dataset
        else ["activitynet", "charades", "hmdb51", "kinetics700", "stair_actions", "ucf101"]
    )

    for dataset in datasets:
        main(dataset)
