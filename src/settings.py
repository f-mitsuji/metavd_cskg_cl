from pathlib import Path

RESULTS_DIR = Path("results")
LOGS_DIR = RESULTS_DIR / "logs"

DATA_DIR = Path("data")
METAVD_DIR = DATA_DIR / "metavd"

EMBEDDINGS_DIR = DATA_DIR / "embeddings"
NUMBERBATCH_PATH = EMBEDDINGS_DIR / "numberbatch-en-19.08.txt"
GOOGLENEWS_PATH = EMBEDDINGS_DIR / "GoogleNews-vectors-negative300.bin"

CN_DICT_DIR = DATA_DIR / "conceptnet"
CN_EN_LEMMAS_P = CN_DICT_DIR / "concepts_en_lemmas.p"
CN_DICT2_P = CN_DICT_DIR / "cn_dict2.p"
CN_DICT2_DB = CN_DICT_DIR / "cn_dict2.db"

STOP_WORDS_FILE_PATH = DATA_DIR / "stop_words.txt"


def load_stop_words(file_path: Path) -> set[str]:
    with file_path.open(encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip() and not line.startswith("#")}


STOP_WORDS = load_stop_words(STOP_WORDS_FILE_PATH)
