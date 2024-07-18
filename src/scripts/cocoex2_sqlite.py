import json
import sqlite3

from settings import ACTIVITYNET_COCOEX2_JSON, ACTIVITYNET_COCOEX_JSON

# ファイルパスの設定
# DB_PATH = "conceptnet_lemmas.db"
DB_PATH = "cn_dict2.db"

# JSONファイルの読み込み
with ACTIVITYNET_COCOEX_JSON.open("r") as f:
    processed_labels = json.load(f)


def get_uris_for_lemma(lemma):
    """指定されたレンマのURIを取得する"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT uris FROM lemmas WHERE lemma=?", (lemma,))
    result = c.fetchone()
    conn.close()
    return set(result[0].split(",")) if result else set()


def get_common_uris(lemmas):
    """指定されたレンマリストの共通のURIを取得する"""
    if not lemmas:
        return set()

    common_uris = None
    for lemma in lemmas:
        uris = get_uris_for_lemma(lemma)
        if common_uris is None:
            common_uris = uris
        else:
            common_uris &= uris  # 共通のURIのみを保持
    return common_uris if common_uris is not None else set()


def process_labels_with_conceptnet(processed_labels):
    result = {}
    for label, lemmas in processed_labels.items():
        # print(f"Processing label: {label}")
        # print(f"Lemmas: {lemmas}")
        common_uris = get_common_uris(lemmas)
        # print(f"Common URIs for {label}: {common_uris}")
        result[label] = list(common_uris) if common_uris else []  # 共通URIがない場合、空リスト
    return result


# ConceptNetのレンマで動作ラベルを処理
processed_labels_with_conceptnet = process_labels_with_conceptnet(processed_labels)

# 結果をJSON形式で保存
with ACTIVITYNET_COCOEX2_JSON.open("w") as f:
    json.dump(processed_labels_with_conceptnet, f, indent=2, ensure_ascii=False)

print("Processed labels saved to", ACTIVITYNET_COCOEX2_JSON)
