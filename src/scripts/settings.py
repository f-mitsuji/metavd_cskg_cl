from pathlib import Path

METAVD_DIR = Path("metavd")
CHARADES_CLASSES_CSV = METAVD_DIR / "charades_classes.csv"
KINETICS700_CLASSES_CSV = METAVD_DIR / "kinetics700_classes.csv"

CHARADES_DIR = Path("charades")
CHARADES_TEST_CSV = CHARADES_DIR / "charades_test.csv"

RESULTS_DIR = Path("results")
CHARADES_RESULT_DIR = RESULTS_DIR / "charades"
# CHARADES_RESULT_JSON = CHARADES_RESULT_DIR / "charades_result.json"
CHARADES_RESULT_JSON = CHARADES_RESULT_DIR / "charades_result_test.json"
CHARADES_FILTERED_JSON = CHARADES_RESULT_DIR / "charades_result_filtered.json"
CHARADES_FILTERED2_JSON = CHARADES_RESULT_DIR / "charades_result_filtered3.json"

KINETICS700_RESULT_DIR = RESULTS_DIR / "kinetics700"
KINETICS700_RESULT_JSON = KINETICS700_RESULT_DIR / "kinetics700_result.json"
