import json
from time import sleep

import pandas as pd
import requests
from settings import CHARADES_CLASSES_CSV, CHARADES_FILTERED_JSON, CHARADES_RESULT_JSON

charades_classes_df = pd.read_csv(CHARADES_CLASSES_CSV)

charades_action_label_list = charades_classes_df.iloc[:, 1].tolist()


def read_json_file(charades_result_json=CHARADES_RESULT_JSON):
    with charades_result_json.open(encoding="UTF-8") as input_file:
        return json.load(input_file)


def is_uri():
    charades_result = read_json_file()
    base_url = "http://api.conceptnet.io"
    counts = []
    filtered_results = []
    valid_uri_counts = []
    total_uri_counts = []
    for action_label, entry in zip(charades_action_label_list, charades_result, strict=True):
        generated_uris = entry.values()
        actions = entry.keys()
        valid_uris = []
        filtered_entry = {}
        for action, generated_uri in zip(actions, generated_uris, strict=True):
            count = 0
            for uri in generated_uri:
                print(uri)
                r = requests.get(f"{base_url}{uri}").json()
                if "error" not in r:
                    valid_uris.append(uri)
                    count = count + 1
                    print("valid.")
                elif r["error"].get("status") == 404:
                    print("invalid")
                sleep(1)
            counts.append(count)
            filtered_entry[action] = valid_uris
            valid_uri_counts.append(len(valid_uris))
            total_uri_counts.append(len(generated_uri))
            # print(action_label, count, valid_uri_counts, total_uri_counts)
            # print(f"action: {action}, valid_uris: {valid_uris}")

        filtered_results.append(filtered_entry)
        # print(filtered_results)

    with CHARADES_FILTERED_JSON.open("w", encoding="UTF-8") as f:
        json.dump(filtered_results, f, ensure_ascii=False, indent=2)

    total_valid_uris = sum(valid_uri_counts)
    total_uris = sum(total_uri_counts)
    precision = total_valid_uris / total_uris if total_uris > 0 else 0
    print(f"平均: {sum(counts) / len(counts)}")
    print(f"フィルタリングの精度: {precision:.2%}")


if __name__ == "__main__":
    is_uri()
