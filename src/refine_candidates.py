import argparse

import numpy as np
import spacy
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

from src.settings import GOOGLENEWS_PATH, LOGS_DIR, NUMBERBATCH_PATH, RESULTS_DIR
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
log_file = LOGS_DIR / f"refined_candidates_{timestamp}.log"
logger = setup_logger("refine candidates", log_file)

nlp = spacy.load("en_core_web_md")
stops = nlp.Defaults.stop_words


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process action labels for a specified dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["activitynet", "charades", "hmdb51", "kinetics700", "stair_actions", "ucf101"],
        help="The name of the dataset to process.",
    )
    parser.add_argument(
        "--vectorization",
        type=str,
        default="numberbatch",
        choices=["numberbatch", "googlenews", "sentence", "all"],
        help="The vectorization method to use. Default is 'numberbatch'.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.83,
        help="The similarity threshold for filtering candidates. Default is 0.83.",
    )
    return parser.parse_args()


def get_dataset_info(dataset: str) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_dir = RESULTS_DIR / dataset
    extracted_candidates_path = get_latest_file_path(result_dir, f"{dataset}_extracted_candidates_")

    if not extracted_candidates_path.exists():
        msg = f"JSON file for dataset '{dataset}' does not exist at {extracted_candidates_path}"
        raise ValueError(msg)

    return {"extracted_candidates_path": extracted_candidates_path, "result_dir": result_dir}


def tokenize_phrase(label: str) -> list[str]:
    label = label.lower().replace(" ", "_").replace("-", "_")
    words = label.split("_")
    return [word for word in words if word not in {"a", "an", "the"}]


def compute_vector(words: list[str], model, stops: set[str], method: str) -> np.ndarray | None:
    if method == "sentence":
        return model.encode([" ".join(words)])[0]

    phrases = ["_".join(words), "_".join([word for word in words if word not in stops])]
    for phrase in phrases:
        if phrase in model:
            return model[phrase]

    vectors = [model[word] for word in words if word in model and word not in stops]
    return np.mean(vectors, axis=0) if vectors else None


def calculate_cosine_similarity(
    source_tokens: list[str], target_tokens: list[str], model, stops: set[str], method: str
) -> float:
    source_vector = compute_vector(source_tokens, model, stops, method)
    target_vector = compute_vector(target_tokens, model, stops, method)

    if source_vector is None or target_vector is None:
        return 0.0

    return 1 - cosine(source_vector, target_vector)


@log_to_file(logger)
def refine_candidates(processed_labels: dict, model, method: str, threshold: float):
    result = {}
    for label, data in processed_labels.items():
        label_words = tokenize_phrase(label)
        common_phrases = data["common_concepts"]
        filtered_phrases = []
        for phrase in common_phrases:
            phrase_words = tokenize_phrase(phrase)
            similarity = calculate_cosine_similarity(label_words, phrase_words, model, stops, method)
            if similarity >= threshold:
                filtered_phrases.append(phrase)
        result[label] = filtered_phrases
    return result


@log_to_file(logger)
def refine_candidates_multi_model(
    processed_labels: dict, numberbatch_model, googlenews_model, sentence_model, threshold: float
):
    result = {}
    for label, data in processed_labels.items():
        label_words = tokenize_phrase(label)
        common_phrases = data["common_concepts"]
        filtered_phrases = []
        for phrase in common_phrases:
            phrase_words = tokenize_phrase(phrase)
            similarities = [
                calculate_cosine_similarity(label_words, phrase_words, numberbatch_model, stops, "numberbatch"),
                calculate_cosine_similarity(label_words, phrase_words, googlenews_model, stops, "googlenews"),
                calculate_cosine_similarity(label_words, phrase_words, sentence_model, stops, "sentence"),
            ]
            if any(sim >= threshold for sim in similarities):
                filtered_phrases.append(phrase)
        result[label] = filtered_phrases
    return result


def main() -> None:
    args = parse_arguments()

    models = {}
    if args.vectorization in ["numberbatch", "all"]:
        models["numberbatch"] = KeyedVectors.load_word2vec_format(NUMBERBATCH_PATH, binary=False)
    if args.vectorization in ["googlenews", "all"]:
        models["googlenews"] = KeyedVectors.load_word2vec_format(GOOGLENEWS_PATH, binary=True)
    if args.vectorization in ["sentence", "all"]:
        models["sentence"] = SentenceTransformer("all-MiniLM-L6-v2")

    try:
        logger.info(f"Started processing dataset: {args.dataset} with vectorization method: {args.vectorization}")
        dataset_info = get_dataset_info(args.dataset)
        extracted_candidates = load_json(dataset_info["extracted_candidates_path"])

        if args.vectorization == "all":
            refined_candidates = refine_candidates_multi_model(
                extracted_candidates, models["numberbatch"], models["googlenews"], models["sentence"], args.threshold
            )
        else:
            refined_candidates = refine_candidates(
                extracted_candidates, models[args.vectorization], args.vectorization, args.threshold
            )

        save_json_with_timestamp(refined_candidates, dataset_info["result_dir"], f"{args.dataset}_refined_candidates")
        logger.info(f"Finished processing dataset: {args.dataset}")
    except Exception:
        logger.exception("An error occurred")
        raise


if __name__ == "__main__":
    main()
