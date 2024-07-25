import pytest

from src.lemmatize_labels import lemmatize_and_remove_stopwords, preprocess_label, process_labels

# テストデータを定義
test_labels = ["running_fast", "jumping_high"]
expected_processed_labels = {"running_fast": ["run", "fast"], "jumping_high": ["jump", "high"]}


def test_preprocess_label():
    """Test the preprocess_label function."""
    assert preprocess_label("Running_Fast") == "running fast"
    assert preprocess_label("Jumping-High") == "jumping high"


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("running fast", ["run", "fast"]),
        ("jumping high", ["jump", "high"]),
    ],
)
def test_lemmatize_and_remove_stopwords(label, expected):
    """Test the lemmatize_and_remove_stopwords function."""
    assert lemmatize_and_remove_stopwords(label) == expected


def test_process_labels():
    """Test the process_labels function."""
    assert process_labels(test_labels) == expected_processed_labels
