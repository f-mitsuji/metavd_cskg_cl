import json

import pandas as pd
import spacy
from settings import ACTIVITYNET_CLASSES_CSV, ACTIVITYNET_COCOEX_JSON

activitynet_classes_df = pd.read_csv(ACTIVITYNET_CLASSES_CSV)

activitynet_action_label_list = activitynet_classes_df.iloc[:, 1].tolist()

nlp = spacy.load("en_core_web_md")
stop_words = nlp.Defaults.stop_words


def preprocess_label(label):
    return label.lower().replace("_", " ").replace("-", " ")


def lemmatize_and_filter(label):
    doc = nlp(label)
    return [token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct]


def process_labels(labels):
    processed_data = {}
    for label in labels:
        preprocessed_label = preprocess_label(label)
        lemmatized_tokens = lemmatize_and_filter(preprocessed_label)
        processed_data[label] = lemmatized_tokens
    return processed_data


processed_labels = process_labels(activitynet_action_label_list)

with ACTIVITYNET_COCOEX_JSON.open("w") as f:
    json.dump(processed_labels, f, indent=2, ensure_ascii=False)

print("Processed labels saved to", ACTIVITYNET_COCOEX_JSON)
