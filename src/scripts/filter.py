import json

import pandas as pd
import spacy
from settings import CHARADES_CLASSES_CSV, CHARADES_FILTERED2_JSON, CHARADES_FILTERED_JSON
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_md")

charades_classes_df = pd.read_csv(CHARADES_CLASSES_CSV)

charades_action_label_list = charades_classes_df.iloc[:, 1].tolist()


def get_vector(text):
    return nlp(text).vector


def read_json_file(charades_result_json=CHARADES_FILTERED_JSON):
    with charades_result_json.open(encoding="UTF-8") as input_file:
        return json.load(input_file)


def filter_labels(data, threshold=0.5):
    filtered_data = {}
    total_uris = 0
    remaining_uris = 0
    for item in data:
        for label, uris in item.items():
            total_uris += len(uris)
            label_vector = get_vector(label.replace("_", " "))
            for uri in uris:
                # URIのベクトル表現を取得
                uri_vector = get_vector(uri.split("/")[-1].replace("_", " "))
                # コサイン類似度を計算
                similarity = cosine_similarity([label_vector], [uri_vector])[0][0]
                # 類似度が閾値以上の場合は保持
                if similarity >= threshold:
                    if label not in filtered_data:
                        filtered_data[label] = []
                    filtered_data[label].append(uri)
                    remaining_uris += 1

    with CHARADES_FILTERED2_JSON.open("w", encoding="UTF-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    precision = (remaining_uris / total_uris) * 100
    print(f"フィルタリングの精度: {precision:.2f}%")


if __name__ == "__main__":
    charades_result = read_json_file()
    filter_labels(charades_result, threshold=0.43)

    phrase1 = "playing with a phone camera"
    phrase2 = "technology"
    vector1 = get_vector(phrase1)
    vector2 = get_vector(phrase2)

    similarity = cosine_similarity([vector1], [vector2])[0][0]
    print(f"Similarity between '{phrase1}' and '{phrase2}': {similarity:.4f}")

    # technology 0.3913866
