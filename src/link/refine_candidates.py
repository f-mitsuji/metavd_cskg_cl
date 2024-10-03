import argparse
import re

import numpy as np
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.settings import GOOGLENEWS_PATH, LOGS_DIR, NUMBERBATCH_PATH, RESULTS_DIR, STOP_WORDS
from src.utils import (
    get_current_jst_timestamp,
    get_latest_file_path,
    load_json,
    save_json_with_timestamp,
    setup_logger,
)

timestamp = get_current_jst_timestamp()
LOGS_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOGS_DIR / f"refined_candidates_{timestamp}.log"
logger = setup_logger("refine candidates", log_file)

BATCH_SIZE = 4096


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process action labels for specified datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="*",
        choices=["activitynet", "charades", "hmdb51", "kinetics700", "stair_actions", "ucf101"],
        help="The name(s) of the dataset(s) to process. If not specified, all datasets will be processed.",
    )
    parser.add_argument(
        "--vectorization",
        type=str,
        default="sentence",
        choices=["numberbatch", "googlenews", "sentence"],
        help="The vectorization method to use. Default is 'sentence'.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="The similarity threshold for filtering candidates. Default is 0.8.",
    )
    return parser.parse_args()


def get_dataset_info(dataset: str) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_dir = RESULTS_DIR / dataset
    extracted_candidates_path = get_latest_file_path(result_dir, f"{dataset}_extracted_candidates_")

    if not extracted_candidates_path.exists():
        msg = f"JSON file for dataset '{dataset}' does not exist at {extracted_candidates_path}"
        raise FileNotFoundError(msg)

    return {"extracted_candidates_path": extracted_candidates_path, "result_dir": result_dir}


def tokenize_phrase(label: str) -> list[str]:
    words = re.split(r"[ \-_]", label.lower())
    return [word for word in words if word not in {"a", "an", "the"}]


def compute_vectors(phrases: list[str], model, method: str, stop_words: set[str]) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    if method == "sentence":
        for i in tqdm(range(0, len(phrases), BATCH_SIZE), desc="Computing vectors"):
            batch = phrases[i : i + BATCH_SIZE]
            tokenized_batch = [" ".join(tokenize_phrase(phrase)) for phrase in batch]
            batch_vectors = model.encode(tokenized_batch, show_progress_bar=False)
            vectors.update(zip(batch, batch_vectors, strict=True))
    else:
        for phrase in tqdm(phrases, desc="Computing vectors"):
            words = tokenize_phrase(phrase)
            phrases_to_check = ["_".join(words), "_".join([word for word in words if word not in stop_words])]

            for p in phrases_to_check:
                if p in model:
                    vectors[phrase] = model[p]
                    break
            else:
                word_vectors = [model[word] for word in words if word in model and word not in stop_words]
                if word_vectors:
                    vectors[phrase] = np.mean(word_vectors, axis=0)
    return vectors


def refine_candidates(processed_labels: dict, model, method: str, threshold: float):
    result: dict[str, list] = {}

    unique_phrases = set()
    for label, concepts in processed_labels.items():
        unique_phrases.add(label)
        unique_phrases.update(concepts)

    print(f"Computing vectors for {len(unique_phrases)} unique phrases...")
    phrase_vectors = compute_vectors(list(unique_phrases), model, method, STOP_WORDS)

    total_labels = len(processed_labels)
    for label, concepts in tqdm(processed_labels.items(), total=total_labels, desc="Refining candidates"):
        label_vector = phrase_vectors.get(label)
        if label_vector is None:
            result[label] = []
            continue

        filtered_concepts = []
        for concept in concepts:
            concept_vector = phrase_vectors.get(concept)
            if concept_vector is not None:
                similarity = 1 - cosine(label_vector, concept_vector)
                if similarity >= threshold:
                    filtered_concepts.append({"concept": concept, "similarity": similarity})
        result[label] = filtered_concepts
    return result


def main(dataset: str, vectorization: str, threshold: float, models: dict) -> None:
    try:
        logger.info(f"Started processing dataset: {dataset} with vectorization method: {vectorization}")
        print(f"Started processing dataset: {dataset} with vectorization method: {vectorization}")
        dataset_info = get_dataset_info(dataset)
        extracted_candidates = load_json(dataset_info["extracted_candidates_path"])
        refined_candidates = refine_candidates(extracted_candidates, models[vectorization], vectorization, threshold)
        save_json_with_timestamp(refined_candidates, dataset_info["result_dir"], f"{dataset}_refined_candidates")
        logger.info(f"Finished processing dataset: {dataset}")
    except Exception:
        logger.exception(f"An error occurred while processing dataset: {dataset}")


if __name__ == "__main__":
    args = parse_arguments()
    datasets = (
        args.dataset
        if args.dataset
        else ["activitynet", "charades", "hmdb51", "kinetics700", "stair_actions", "ucf101"]
    )
    vectorization = args.vectorization
    threshold = args.threshold

    print("Loading models...")
    models = {}
    if vectorization == "numberbatch":
        models["numberbatch"] = KeyedVectors.load_word2vec_format(NUMBERBATCH_PATH, binary=False)
    if vectorization == "googlenews":
        models["googlenews"] = KeyedVectors.load_word2vec_format(GOOGLENEWS_PATH, binary=True)
    if vectorization == "sentence":
        models["sentence"] = SentenceTransformer("all-mpnet-base-v2")
    print("Models loaded successfully")

    for dataset in datasets:
        main(dataset, vectorization, threshold, models)
