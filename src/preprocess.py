import argparse
from pathlib import Path

import pandas as pd
import spacy
from settings import METAVD_DIR, RESULTS_DIR
from utils import log_to_file, save_json_with_timestamp

DATASETS = {
    "activitynet": "activitynet_classes.csv",
    "kinetics700": "kinetics700_classes.csv",
    "hmdb51": "hmdb51_classes.csv",
    "stair_actions": "stair_actions_classes.csv",
    "ucf101": "ucf101_classes.csv",
    "charades": "charades_classes.csv",
}


@log_to_file(RESULTS_DIR / "process.log")
def get_dataset_info(dataset: str) -> dict:
    """Get the file path and result directory for the given dataset.

    Args:
        dataset (str): The name of the dataset.

    Returns:
        dict: Dictionary with 'csv_path' and 'result_dir'.
    """
    if dataset not in DATASETS:
        msg = f"Dataset '{dataset}' is not supported. Supported datasets: {list(DATASETS.keys())}"
        raise ValueError(msg)

    csv_path = METAVD_DIR / DATASETS[dataset]
    result_dir = RESULTS_DIR / dataset
    return {"csv_path": csv_path, "result_dir": result_dir}


@log_to_file(RESULTS_DIR / "process.log")
def load_labels(file_path: Path) -> list:
    """Load action labels from a CSV file.

    Args:
        file_path (Path): The path to the CSV file containing the labels.

    Returns:
        list: A list of action labels.
    """
    classes_df = pd.read_csv(file_path)
    return classes_df.iloc[:, 1].tolist()


@log_to_file(RESULTS_DIR / "process.log")
def preprocess_label(label: str) -> str:
    """Preprocess the label by making it lowercase and replacing underscores and hyphens with spaces.

    Args:
        label (str): The original label string.

    Returns:
        str: The preprocessed label string.
    """
    return label.lower().replace("_", " ").replace("-", " ")


@log_to_file(RESULTS_DIR / "process.log")
def lemmatize_and_filter(label: str, nlp, stop_words) -> list:
    """Lemmatize the label and filter out stop words and punctuation.

    Args:
        label (str): The preprocessed label string.
        nlp (Language): The SpaCy language model.
        stop_words (set): A set of stop words to be excluded.

    Returns:
        list: A list of lemmatized tokens from the label.
    """
    doc = nlp(label)
    return [token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct]


@log_to_file(RESULTS_DIR / "process.log")
def process_labels(labels: list, nlp, stop_words) -> dict:
    """Process a list of labels and return a dictionary of original labels to lemmatized token lists.

    Args:
        labels (list): A list of original labels.
        nlp (Language): The SpaCy language model.
        stop_words (set): A set of stop words to be excluded.

    Returns:
        dict: A dictionary mapping original labels to their lemmatized token lists.
    """
    processed_data = {}
    for label in labels:
        preprocessed_label = preprocess_label(label)
        lemmatized_tokens = lemmatize_and_filter(preprocessed_label, nlp, stop_words)
        processed_data[label] = lemmatized_tokens
    return processed_data


@log_to_file(RESULTS_DIR / "process.log")
def main(dataset: str) -> None:
    """Main function to load, process, and save action labels for the specified dataset.

    Args:
        dataset (str): The name of the dataset to process.
    """
    nlp = spacy.load("en_core_web_md")
    stop_words = nlp.Defaults.stop_words

    dataset_info = get_dataset_info(dataset)
    labels = load_labels(dataset_info["csv_path"])
    processed_labels = process_labels(labels, nlp, stop_words)
    save_json_with_timestamp(processed_labels, dataset_info["result_dir"], f"{dataset}_lemmatized_labels")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process action labels for a specified dataset.")
    parser.add_argument(
        "--dataset", type=str, help="The name of the dataset to process (e.g., 'activitynet', 'kinetics700')."
    )
    args = parser.parse_args()

    main(args.dataset)
