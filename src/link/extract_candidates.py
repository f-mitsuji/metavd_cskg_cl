import argparse
import sqlite3

from src.settings import CN_DICT2_DB, LOGS_DIR, RESULTS_DIR
from src.utils import (
    get_current_jst_timestamp,
    get_latest_file_path,
    load_json,
    log_to_file,
    save_json_with_timestamp,
    setup_logger,
)

timestamp = get_current_jst_timestamp()
LOGS_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOGS_DIR / f"extract_candidates_{timestamp}.log"
logger = setup_logger("extract candidates", log_file)


def get_dataset_info(dataset: str) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_dir = RESULTS_DIR / dataset
    lemmatized_labels_path = get_latest_file_path(result_dir, f"{dataset}_lemmatized_labels_")

    if not lemmatized_labels_path.exists():
        msg = f"JSON file for dataset '{dataset}' does not exist at {lemmatized_labels_path}"
        raise ValueError(msg)

    return {"lemmatized_labels_path": lemmatized_labels_path, "result_dir": result_dir}


@log_to_file(logger)
def extract_candidates(lemmatized_labels):
    extracted_candidates = {}
    unique_lemmas = {lemma for lemmas in lemmatized_labels.values() for lemma in lemmas}

    with sqlite3.connect(CN_DICT2_DB) as conn:
        cursor = conn.cursor()
        placeholders = ",".join("?" for _ in unique_lemmas)
        cursor.execute(f"SELECT lemma, concepts FROM cn_dict WHERE lemma IN ({placeholders})", list(unique_lemmas))  # noqa: S608
        lemma_to_concepts = {row[0]: set(row[1].split(",")) for row in cursor.fetchall()}

    for label, lemmas in lemmatized_labels.items():
        common_concepts = set.intersection(*(lemma_to_concepts.get(lemma, set()) for lemma in lemmas))
        lemma_concepts = {lemma: list(lemma_to_concepts.get(lemma, set())) for lemma in lemmas}
        extracted_candidates[label] = {"common_concepts": list(common_concepts), "lemma_concepts": lemma_concepts}

    return extracted_candidates


def main(dataset: str) -> None:
    try:
        logger.info(f"Started processing dataset: {dataset}")
        dataset_info = get_dataset_info(dataset)
        lemmatized_labels = load_json(dataset_info["lemmatized_labels_path"])
        extracted_candidates = extract_candidates(lemmatized_labels)
        save_json_with_timestamp(extracted_candidates, dataset_info["result_dir"], f"{dataset}_extracted_candidates")
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
