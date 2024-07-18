import json
import time

import numpy as np
import spacy
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
from settings import ACTIVITYNET_COCOEX3_JSON, ACTIVITYNET_COCOEX_FILTERED_JSON, NUMBERBATCH_PATH


def log_message(message):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{current_time}] {message}")


THRESHOLD = 0.9
log_message("Loading ConceptNet Numberbatch vectors...")
start_time = time.time()
model = KeyedVectors.load_word2vec_format(NUMBERBATCH_PATH, binary=False)
log_message(
    f"Loaded {len(model.key_to_index)} vectors from {NUMBERBATCH_PATH} in {time.time() - start_time:.2f} seconds"
)
nlp = spacy.load("en_core_web_md")
stops = nlp.Defaults.stop_words


def cos_similarity(source_words, target_words, model, stops):
    vectors_to_compare = []

    for words in [source_words, target_words]:
        if len(words) > 1:
            joined_with_dashes = "_".join(words)
            if joined_with_dashes in model:
                vector = model[joined_with_dashes]
            else:
                words_without_stops = [word for word in words if word not in stops]
                joined_with_dashes_nostops = "_".join(words_without_stops)
                if joined_with_dashes_nostops in model:
                    vector = model[joined_with_dashes_nostops]
                else:
                    vectors = [model[word] for word in words_without_stops if word in model]
                    if vectors:
                        vector = np.mean(vectors, axis=0)
                    else:
                        return 0.0
        else:
            try:
                vector = model[words[0]]
            except KeyError:
                return 0.0

        vectors_to_compare.append(vector)

    return 1 - cosine(vectors_to_compare[0], vectors_to_compare[1])


def transform_label(label):
    """動作ラベルをNumberbatchの形式に変換し、冠詞を削除する"""
    label = label.lower().replace(" ", "_").replace("-", "_")
    words = label.split("_")
    filtered_words = [word for word in words if word not in {"a", "an", "the"}]
    return "_".join(filtered_words)


def filter_by_cosine_similarity(processed_labels, model, stops, threshold):
    result = {}
    total_labels = len(processed_labels)
    start_time = time.time()
    for i, (label, phrases) in enumerate(processed_labels.items()):
        if i % 10 == 0:
            log_message(f"Processing label {i+1}/{total_labels} ({(i+1)/total_labels*100:.2f}%)")
        transformed_label = transform_label(label)
        label_words = transformed_label.split("_")
        filtered_phrases = []
        for phrase in phrases:
            transformed_phrase = transform_label(phrase)
            phrase_words = transformed_phrase.split("_")
            similarity = cos_similarity(label_words, phrase_words, model, stops)
            if similarity >= threshold:
                filtered_phrases.append(phrase)
        result[label] = filtered_phrases
    log_message(f"Filtering completed in {time.time() - start_time:.2f} seconds")
    return result


with ACTIVITYNET_COCOEX3_JSON.open("r") as f:
    processed_labels = json.load(f)

filtered_labels = filter_by_cosine_similarity(processed_labels, model, stops, THRESHOLD)


with ACTIVITYNET_COCOEX_FILTERED_JSON.open("w") as f:
    json.dump(filtered_labels, f, indent=2, ensure_ascii=False)

print("Filtered labels saved to", ACTIVITYNET_COCOEX_FILTERED_JSON)
