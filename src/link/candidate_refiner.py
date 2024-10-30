import argparse
import re

import numpy as np
import spacy
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sent2vec
from src.settings import GOOGLENEWS_PATH, LOGS_DIR, NUMBERBATCH_PATH, RESULTS_DIR, SENT2VEC_MODEL_PATH, STOP_WORDS
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

nlp = spacy.load("en_core_web_lg")

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
        default="mpnet",
        choices=["numberbatch", "word2vec", "sent2vec", "mpnet"],
        help="The vectorization method to use. Default is 'sentence'.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="The similarity threshold for filtering candidates. Default is 0.8.",
    )
    return parser.parse_args()


def get_dataset_info(dataset: str, vectorization: str) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_dir = RESULTS_DIR / dataset
    result_dir.mkdir(parents=True, exist_ok=True)
    output_dir = RESULTS_DIR / dataset / vectorization
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_candidates_path = get_latest_file_path(result_dir, f"{dataset}_extracted_candidates_")

    if not extracted_candidates_path.exists():
        msg = f"JSON file for dataset '{dataset}' does not exist at {extracted_candidates_path}"
        raise FileNotFoundError(msg)

    return {"extracted_candidates_path": extracted_candidates_path, "output_dir": output_dir}


def tokenize_label(label: str, method: str) -> list[str]:
    if label.isupper():
        label = label.lower()

    label = re.sub(r"\([^)]*\)", "", label)
    label = re.sub(r"(?<!^)(?=[A-Z])", " ", label)
    label = re.sub(r"[_\-/]", " ", label).lower()
    if method == "numberbatch":
        return [token for token in label.split() if token not in {"a", "an", "the"}]
    if method == "sent2vec":
        doc = nlp(label)
        return [token.text for token in doc]
    return label.split()


def tokenize_concept(concept: str, method: str) -> list[str]:
    if method == "sent2vec":
        doc = nlp(concept)
        return [token.text for token in doc]
    return concept.split()


def create_embedding(
    phrases: list[str], model, method: str, stop_words: set[str], *, is_label: bool
) -> dict[str, np.ndarray]:
    phrase_embeddings: dict[str, np.ndarray] = {}
    tokenize_func = tokenize_label if is_label else tokenize_concept

    if method == "sent2vec":
        for phrase in tqdm(phrases, desc="Creating embeddings"):
            tokenize_words = " ".join(tokenize_func(phrase, method))
            embedding = model.embed_sentence(tokenize_words)
            phrase_embeddings[phrase] = embedding.flatten()
            if phrase_embeddings[phrase].shape != (600,):
                logger.warning(f"Unexpected shape for '{phrase}': {phrase_embeddings[phrase].shape}")
    elif method == "mpnet":
        for i in tqdm(range(0, len(phrases), BATCH_SIZE), desc="Creating embeddings"):
            batch = phrases[i : i + BATCH_SIZE]
            tokenized_batch = [" ".join(tokenize_func(phrase, method)) for phrase in batch]
            batch_embeddings = model.encode(tokenized_batch, show_progress_bar=False)
            phrase_embeddings.update(zip(batch, batch_embeddings, strict=True))
    else:
        for phrase in tqdm(phrases, desc="Creating embeddings"):
            tokenized_words = tokenize_func(phrase, method)
            compound_phrases = [
                "_".join(tokenized_words),
                "_".join([word for word in tokenized_words if word not in stop_words]),
            ]

            for compound_phrase in compound_phrases:
                if compound_phrase in model:
                    phrase_embeddings[phrase] = model[compound_phrase]
                    break
            else:
                word_embeddings = [model[word] for word in tokenized_words if word in model and word not in stop_words]
                if word_embeddings:
                    phrase_embeddings[phrase] = np.mean(word_embeddings, axis=0)
    return phrase_embeddings


def refine_candidates(processed_labels: dict, model, method: str, threshold: float):
    result: dict[str, list] = {}

    unique_labels = set(processed_labels.keys())
    unique_concepts = set()
    for concepts in processed_labels.values():
        unique_concepts.update(concepts)

    print(f"Computing vectors for {len(unique_labels)} unique labels...")
    label_vectors = create_embedding(list(unique_labels), model, method, STOP_WORDS, is_label=True)

    print(f"Computing vectors for {len(unique_concepts)} unique concepts...")
    concept_vectors = create_embedding(list(unique_concepts), model, method, STOP_WORDS, is_label=False)

    total_labels = len(processed_labels)
    for label, concepts in tqdm(processed_labels.items(), total=total_labels, desc="Refining candidates"):
        label_vector = label_vectors.get(label)
        if label_vector is None:
            result[label] = []
            continue

        filtered_concepts = []
        for concept in concepts:
            concept_vector = concept_vectors.get(concept)
            if concept_vector is not None:
                similarity = 1 - cosine(label_vector, concept_vector)
                if similarity >= threshold:
                    filtered_concepts.append({"concept": concept, "similarity": similarity})
        result[label] = filtered_concepts
    return result


def main(dataset: str, vectorization: str, threshold: float, model: dict) -> None:
    try:
        logger.info(f"Started processing dataset: {dataset} with vectorization method: {vectorization}")
        print(f"Started processing dataset: {dataset} with vectorization method: {vectorization}")
        dataset_info = get_dataset_info(dataset, vectorization)
        extracted_candidates = load_json(dataset_info["extracted_candidates_path"])
        refined_candidates = refine_candidates(extracted_candidates, model, vectorization, threshold)
        save_json_with_timestamp(
            refined_candidates, dataset_info["output_dir"], f"{dataset}_refined_candidates_{vectorization}"
        )
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
    if vectorization == "numberbatch":
        model = KeyedVectors.load_word2vec_format(NUMBERBATCH_PATH, binary=False)
    if vectorization == "word2vec":
        model = KeyedVectors.load_word2vec_format(GOOGLENEWS_PATH, binary=True)
    if vectorization == "mpnet":
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    if vectorization == "sent2vec":
        model = sent2vec.Sent2vecModel()
        model.load_model(str(SENT2VEC_MODEL_PATH))
    print("Models loaded successfully")

    for dataset in datasets:
        main(dataset, vectorization, threshold, model)
