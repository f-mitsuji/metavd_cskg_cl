import numpy as np
import pytest
from gensim.models import KeyedVectors

from src.refine_candidates import (
    calculate_cosine_similarity,
    compute_vector,
    refine_candidates,
    refine_candidates_multi_model,
    tokenize_phrase,
)


@pytest.fixture()
def mock_data():
    return {
        "running": {"common_concepts": ["run", "jog", "sprint"]},
        "eating": {"common_concepts": ["eat", "consume", "dine"]},
    }


@pytest.fixture()
def mock_numberbatch_model():
    model = KeyedVectors(vector_size=5)
    model.add_vector("running", np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    model.add_vector("run", np.array([0.11, 0.21, 0.31, 0.41, 0.51]))
    model.add_vector("jog", np.array([0.15, 0.25, 0.35, 0.45, 0.55]))
    model.add_vector("sprint", np.array([0.2, 0.3, 0.4, 0.5, 0.6]))
    model.add_vector("eating", np.array([0.6, 0.7, 0.8, 0.9, 1.0]))
    model.add_vector("eat", np.array([0.61, 0.71, 0.81, 0.91, 1.01]))
    model.add_vector("consume", np.array([0.65, 0.75, 0.85, 0.95, 1.05]))
    model.add_vector("dine", np.array([0.7, 0.8, 0.9, 1.0, 1.1]))
    model.add_vector("fast", np.array([0.12, 0.22, 0.32, 0.42, 0.52]))
    return model


@pytest.fixture()
def mock_sentence_model():
    class MockSentenceTransformer:
        def encode(self, sentences):
            return np.array([[0.1, 0.2, 0.3, 0.4, 0.5]] * len(sentences))

    return MockSentenceTransformer()


def test_tokenize_phrase():
    assert tokenize_phrase("running fast") == ["running", "fast"]
    assert tokenize_phrase("eating_a_burger") == ["eating", "burger"]
    assert tokenize_phrase("jumping-over-fence") == ["jumping", "over", "fence"]


def test_compute_vector(mock_numberbatch_model, mock_sentence_model):
    stops = {"a", "an", "the"}

    vector = compute_vector(["run"], mock_numberbatch_model, stops, "numberbatch")
    assert np.allclose(vector, np.array([0.11, 0.21, 0.31, 0.41, 0.51]), atol=1e-6)

    vector = compute_vector(["run"], mock_sentence_model, stops, "sentence")
    assert np.allclose(vector, np.array([0.1, 0.2, 0.3, 0.4, 0.5]), atol=1e-6)

    vector = compute_vector(["unknown"], mock_numberbatch_model, stops, "numberbatch")
    assert vector is None

    vector = compute_vector(["run", "fast"], mock_numberbatch_model, stops, "numberbatch")
    expected = (np.array([0.11, 0.21, 0.31, 0.41, 0.51]) + mock_numberbatch_model["fast"]) / 2
    assert np.allclose(vector, expected, atol=1e-6)

    vector = compute_vector(["the", "unknown", "run", "fast"], mock_numberbatch_model, stops, "numberbatch")
    expected = (np.array([0.11, 0.21, 0.31, 0.41, 0.51]) + mock_numberbatch_model["fast"]) / 2
    assert np.allclose(vector, expected, atol=1e-6)


def test_calculate_cosine_similarity(mock_numberbatch_model):
    stops = {"a", "an", "the"}

    similarity = calculate_cosine_similarity(["running"], ["run"], mock_numberbatch_model, stops, "numberbatch")
    assert 0.99 < similarity <= 1.0

    similarity = calculate_cosine_similarity(["running"], ["sprint"], mock_numberbatch_model, stops, "numberbatch")
    assert 0.99 < similarity < 1.0

    similarity = calculate_cosine_similarity(["running"], ["eating"], mock_numberbatch_model, stops, "numberbatch")
    assert similarity < 0.97


def test_refine_candidates(mock_data, mock_numberbatch_model):
    refined = refine_candidates(mock_data, mock_numberbatch_model, "numberbatch", 0.8)
    assert "running" in refined
    assert "eating" in refined
    assert len(refined["running"]) > 0
    assert len(refined["eating"]) > 0


def test_refine_candidates_multi_model(mock_data, mock_numberbatch_model, mock_sentence_model):
    refined = refine_candidates_multi_model(
        mock_data,
        mock_numberbatch_model,
        mock_numberbatch_model,
        mock_sentence_model,
        0.8,
    )
    assert "running" in refined
    assert "eating" in refined
    assert len(refined["running"]) > 0
    assert len(refined["eating"]) > 0


def test_compute_vector_unknown_word(mock_numberbatch_model):
    stops = set()
    vector = compute_vector(["unknown"], mock_numberbatch_model, stops, "numberbatch")
    assert vector is None


def test_compute_vector_empty_input(mock_numberbatch_model):
    stops = set()
    vector = compute_vector([], mock_numberbatch_model, stops, "numberbatch")
    assert vector is None


def test_threshold_behavior(mock_data, mock_numberbatch_model):
    refined_high_threshold = refine_candidates(mock_data, mock_numberbatch_model, "numberbatch", 0.995)
    refined_low_threshold = refine_candidates(mock_data, mock_numberbatch_model, "numberbatch", 0.95)
    high_count = sum(len(v) for v in refined_high_threshold.values())
    low_count = sum(len(v) for v in refined_low_threshold.values())
    assert high_count < low_count
