import argparse
from pathlib import Path

import pandas as pd
import spacy
from settings import LOGS_DIR, METAVD_DIR, RESULTS_DIR
from utils import get_current_jst_timestamp, log_to_file, save_json_with_timestamp, setup_logger

timestamp = get_current_jst_timestamp()
log_file = LOGS_DIR / f"lemmatize_labels_{timestamp}.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logger = setup_logger("lemmatize labels", log_file)


nlp = spacy.load("en_core_web_md")
stop_words = nlp.Defaults.stop_words


def get_dataset_info(dataset: str) -> dict:
    csv_path = METAVD_DIR / f"{dataset}_classes.csv"
    result_dir = RESULTS_DIR / dataset

    if not csv_path.exists():
        msg = f"CSV file for dataset '{dataset}' does not exist at {csv_path}"
        raise ValueError(msg)

    return {"csv_path": csv_path, "result_dir": result_dir}


def load_labels(file_path: Path) -> list:
    classes_df = pd.read_csv(file_path)
    return classes_df.iloc[:, 1].tolist()


@log_to_file(logger)
def preprocess_label(label: str) -> str:
    return label.lower().replace("_", " ").replace("-", " ")


@log_to_file(logger)
def lemmatize_and_remove_stopwords(label: str) -> list:
    doc = nlp(label)
    return [token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct]


@log_to_file(logger)
def process_labels(labels: list) -> dict:
    processed_data = {}
    for label in labels:
        preprocessed_label = preprocess_label(label)
        lemmatized_tokens = lemmatize_and_remove_stopwords(preprocessed_label, nlp, stop_words)
        processed_data[label] = lemmatized_tokens
    return processed_data


@log_to_file(logger)
def main(dataset: str) -> None:
    try:
        logger.info(f"Started processing dataset: {dataset}")
        dataset_info = get_dataset_info(dataset)
        labels = load_labels(dataset_info["csv_path"])
        processed_labels = process_labels(labels)
        save_json_with_timestamp(processed_labels, dataset_info["result_dir"], f"{dataset}_lemmatized_labels")
        logger.info(f"Finished processing dataset: {dataset}")
    except Exception:
        logger.exception("An error occurred")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process action labels for a specified dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["activitynet", "charades", "hmdb51", "kinetics700", "stair_actions", "ucf101"],
        help="The name of the dataset to process.",
    )
    args = parser.parse_args()

    main(args.dataset)
