from pathlib import Path

RESULTS_DIR = Path("results")

DATA_DIR = Path("data")
METAVD_DIR = DATA_DIR / "metavd"

EMBEDDINGS_DIR = DATA_DIR / "embeddings"
NUMBERBATCH_PATH = EMBEDDINGS_DIR / "numberbatch-en-19.08.txt"
GOOGLE_NEWS_PATH = EMBEDDINGS_DIR / "GoogleNews-vectors-negative300.bin"

CN_DICT_DIR = DATA_DIR / "conceptnet"
CN_EN_LEMMAS_P = CN_DICT_DIR / "concepts_en_lemmas.p"
CN_DICT_P = CN_DICT_DIR / "cn_dict2.p"
