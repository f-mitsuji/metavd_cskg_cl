import json
import pickle

from settings import ACTIVITYNET_COCOEX2_JSON, ACTIVITYNET_COCOEX_JSON, CN_EN_LEMMAS_P

with ACTIVITYNET_COCOEX_JSON.open("r") as f:
    processed_labels = json.load(f)

# with CN_EN_LEMMAS_P.open("rb") as f:
#     conceptnet_lemmas = pickle.load(f)
with CN_EN_LEMMAS_P.open("rb") as f:
    conceptnet_lemmas = pickle.load(f)


# def get_common_uris(lemmas):
#     if not lemmas:
#         return set()

#     common_uris = None
#     for lemma in lemmas:
#         if lemma in conceptnet_lemmas:
#             uris = conceptnet_lemmas[lemma]
#             if common_uris is None:
#                 common_uris = uris
#             else:
#                 common_uris &= uris  # 共通のURIのみを保持
#         else:
#             return set()  # レンマが見つからない場合、共通部分は空集合

#     return common_uris if common_uris is not None else set()


def get_uris_for_lemma(lemma):
    """指定されたレンマのURIを取得する"""
    return conceptnet_lemmas.get(lemma, set())


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
        common_uris = get_common_uris(lemmas)
        result[label] = list(common_uris) if common_uris else []  # 共通URIがない場合、空リスト
    return result


processed_labels_with_conceptnet = process_labels_with_conceptnet(processed_labels)


with ACTIVITYNET_COCOEX2_JSON.open("w") as f:
    json.dump(processed_labels_with_conceptnet, f, indent=2, ensure_ascii=False)

print("Processed labels saved to", ACTIVITYNET_COCOEX2_JSON)
