import argparse
import sqlite3

from settings import CN_DICT2_DB as DB_PATH
from settings import LOGS_DIR, RESULTS_DIR
from utils import (
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
def fetch_concepts_for_lemma(lemma):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT concepts FROM cn_dict WHERE lemma=?", (lemma,))
            result = cursor.fetchone()
            if result:
                concepts = result[0].split(",")
                return set(concepts)
            return set()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return set()
    except Exception as e:  # noqa: BLE001
        print(f"Unexpected error: {e}")
        return set()


@log_to_file(logger)
def fetch_common_concepts_for_lemmas(lemmas):
    if not lemmas:
        return set()

    common_concepts = fetch_concepts_for_lemma(lemmas[0])
    for lemma in lemmas[1:]:
        concepts = fetch_concepts_for_lemma(lemma)
        common_concepts &= concepts

        if not common_concepts:
            return set()

    return common_concepts


def extract_candidates(lemmatized_labels):
    extracted_candidates = {}
    for label, lemmas in lemmatized_labels.items():
        common_concepts = fetch_common_concepts_for_lemmas(lemmas)
        lemma_concepts = {lemma: list(fetch_concepts_for_lemma(lemma)) for lemma in lemmas}
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
