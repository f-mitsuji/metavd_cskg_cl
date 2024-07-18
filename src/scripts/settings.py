from pathlib import Path

METAVD_DIR = Path("metavd")
CHARADES_CLASSES_CSV = METAVD_DIR / "charades_classes.csv"
KINETICS700_CLASSES_CSV = METAVD_DIR / "kinetics700_classes.csv"
ACTIVITYNET_CLASSES_CSV = METAVD_DIR / "activitynet_classes.csv"

CHARADES_DIR = Path("charades")
CHARADES_TEST_CSV = CHARADES_DIR / "charades_test.csv"

RESULTS_DIR = Path("results")
CHARADES_RESULT_DIR = RESULTS_DIR / "charades"
# CHARADES_RESULT_JSON = CHARADES_RESULT_DIR / "charades_result.json"
CHARADES_RESULT_JSON = CHARADES_RESULT_DIR / "charades_result_test.json"
CHARADES_FILTERED_JSON = CHARADES_RESULT_DIR / "charades_result_filtered.json"
CHARADES_FILTERED2_JSON = CHARADES_RESULT_DIR / "charades_result_filtered3.json"
CHARADES_COCOEX_JSON = CHARADES_RESULT_DIR / "charades_cocoex.json"
CHARADES_COCOEX2_JSON = CHARADES_RESULT_DIR / "charades_cocoex2.json"

ACTIVITYNET_RESULT_DIR = RESULTS_DIR / "activitynet"
ACTIVITYNET_COCOEX_JSON = ACTIVITYNET_RESULT_DIR / "activitynet_cocoex.json"
ACTIVITYNET_COCOEX2_JSON = ACTIVITYNET_RESULT_DIR / "activitynet_cocoex2.json"
ACTIVITYNET_COCOEX3_JSON = ACTIVITYNET_RESULT_DIR / "activitynet_cocoex3.json"
ACTIVITYNET_COCOEX_FILTERED_JSON = ACTIVITYNET_RESULT_DIR / "activitynet_cocoex_filtered0.9.json"

KINETICS700_RESULT_DIR = RESULTS_DIR / "kinetics700"
KINETICS700_RESULT_JSON = KINETICS700_RESULT_DIR / "kinetics700_result.json"

CN_EN_LEMMAS_P = Path("concepts_en_lemmas.p")
CN_DICT_P = Path("cn_dict2.p")

NUMBERBATCH_PATH = "numberbatch-en-19.08.txt"
