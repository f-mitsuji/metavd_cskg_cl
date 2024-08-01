import pytest

from src.lemmatize_labels import lemmatize_and_filter_token, lemmatize_labels, normalize_label

test_labels = ["running_fast", "jumping_high"]
expected_lemmatized_labels = {"running_fast": ["run", "fast"], "jumping_high": ["jump", "high"]}


def test_normalize_label():
    assert normalize_label("Running_Fast") == "running fast"
    assert normalize_label("Jumping-High") == "jumping high"


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("running", "run"),
        ("jumping", "jump"),
    ],
)
def test_lemmatize_and_filter_token(label, expected):
    assert lemmatize_and_filter_token(label) == expected


def test_lemmatize_labels():
    assert lemmatize_labels(test_labels) == expected_lemmatized_labels
