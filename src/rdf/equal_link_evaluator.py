import json
import logging
import textwrap
from pathlib import Path

from SPARQLWrapper import JSON, SPARQLWrapper
from SPARQLWrapper.SPARQLExceptions import SPARQLWrapperException
from utils import get_latest_file_path, load_json

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

# VECTORIZATION = "mpnet"
# VECTORIZATION = "sent2vec"
VECTORIZATION = "numberbatch"
# VECTORIZATION = "word2vec"
# THRESHOLD = 0.8
THRESHOLD = 0.85
# THRESHOLD = 0.9

FUSEKI_ENDPOINT = f"http://localhost:3030/{VECTORIZATION}_{THRESHOLD}/query"


PREFIXES = """
PREFIX act: <http://example.org/action/>
PREFIX ds: <http://example.org/dataset/>
PREFIX cn: <http://conceptnet.io/c/en/>
PREFIX cn_rel: <http://conceptnet.io/r/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def execute_sparql_query(query: str) -> list[dict]:
    full_query = PREFIXES + query
    sparql = SPARQLWrapper(FUSEKI_ENDPOINT)
    sparql.setQuery(full_query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        return results["results"]["bindings"]
    except SPARQLWrapperException as e:
        logging.exception(f"SPARQLWrapper specific error: {type(e).__name__}")
    except Exception as e:
        logging.exception(f"Unexpected error: {type(e).__name__}")

    return []


def load_refined_candidates(dataset: str) -> dict[str, list[str]]:
    result_dir = RESULTS_DIR / dataset / VECTORIZATION
    refined_candidates_path = get_latest_file_path(result_dir, f"{dataset}_refined_candidates_{VECTORIZATION}_")
    return load_json(refined_candidates_path)


def get_graph_stats() -> dict[str, int]:
    queries = [
        ("total_triples", "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }"),
        (
            "total_actions",
            "SELECT (COUNT(DISTINCT ?s) as ?count) WHERE { ?s a act:Action }",
        ),
        (
            "total_datasets",
            "SELECT (COUNT(DISTINCT ?s) as ?count) WHERE { ?s a ds:Dataset }",
        ),
        (
            "unique_concepts",
            "SELECT (COUNT(DISTINCT ?o) as ?count) WHERE { ?s act:relatedToConcept ?o }",
        ),
    ]
    stats = {}
    for label, query in queries:
        results = execute_sparql_query(query)
        if results:
            stats[label] = int(results[0]["count"]["value"])
        else:
            logging.warning(f"No results for {label} query")
    return stats


def print_graph_stats(stats: dict[str, int]):
    print("\nGraph Statistics:")
    for label, count in stats.items():
        print(f"{label.replace('_', ' ').title()}: {count}")


def query_actions_by_concept(concept: str) -> list[tuple[str, str]]:
    query = textwrap.dedent(f"""
    SELECT DISTINCT ?action_label ?dataset_label
    WHERE {{
        ?input_concept rdfs:label "{concept}" .

        ?action act:relatedToConcept ?input_concept ;
            rdfs:label ?action_label ;
            ds:belongsTo ?dataset .
        ?dataset rdfs:label ?dataset_label .
    }}
    ORDER BY ?action_label ?dataset_label
    """)
    results = execute_sparql_query(query)
    return [(result["action_label"]["value"], result["dataset_label"]["value"]) for result in results]


def evaluate_results(query_results: list[tuple[str, str]], ground_truth: list[tuple[str, str]]) -> dict:
    query_set = set(query_results)
    ground_truth_set = set(ground_truth)

    true_positives = query_set & ground_truth_set
    false_positives = query_set - ground_truth_set
    false_negatives = ground_truth_set - query_set

    precision = len(true_positives) / len(query_set) if query_set else 0
    recall = len(true_positives) / len(ground_truth_set) if ground_truth_set else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "true_positives": list(true_positives),
        "false_positives": list(false_positives),
        "false_negatives": list(false_negatives),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_evaluation():
    equal_sets = load_json(DATA_DIR / "action_equal_sets.json")
    equal_set_results = []

    for equal_set in equal_sets:
        set_evaluation_results = []

        # Evaluate each action in the equal set
        for action_info in equal_set["actions"]:
            dataset, action = action_info["dataset"], action_info["action"]

            refined_candidates = load_refined_candidates(dataset)
            concepts_with_sim = refined_candidates.get(action, [])

            filtered_concepts = [entry for entry in concepts_with_sim if float(entry["similarity"]) >= THRESHOLD]

            all_query_results = set()
            if filtered_concepts:
                for concept_entry in filtered_concepts:
                    concept = concept_entry["concept"]
                    query_results = query_actions_by_concept(concept)
                    all_query_results.update(query_results)

                ground_truth = [(action_info["action"], action_info["dataset"]) for action_info in equal_set["actions"]]

                result = evaluate_results(list(all_query_results), ground_truth)
                set_evaluation_results.append(
                    {
                        "action": action,
                        "dataset": dataset,
                        "concepts": [entry["concept"] for entry in filtered_concepts],
                        **result,
                    }
                )

        # Calculate average metrics for each equal set
        if set_evaluation_results:
            avg_precision = sum(r["precision"] for r in set_evaluation_results) / len(set_evaluation_results)
            avg_recall = sum(r["recall"] for r in set_evaluation_results) / len(set_evaluation_results)
            avg_f1 = sum(r["f1"] for r in set_evaluation_results) / len(set_evaluation_results)

            equal_set_results.append(
                {
                    "set_id": equal_set["set_id"],
                    "action_results": set_evaluation_results,
                    "metrics": {
                        "precision": avg_precision,
                        "recall": avg_recall,
                        "f1": avg_f1,
                    },
                }
            )

    # Average of the average metrics for each equal set
    overall_metrics = {
        "precision": sum(r["metrics"]["precision"] for r in equal_set_results) / len(equal_set_results),
        "recall": sum(r["metrics"]["recall"] for r in equal_set_results) / len(equal_set_results),
        "f1": sum(r["metrics"]["f1"] for r in equal_set_results) / len(equal_set_results),
    }

    evaluation_result = {
        "equal_set_results": equal_set_results,
        "overall_metrics": overall_metrics,
        "parameters": {"vectorization": VECTORIZATION, "threshold": THRESHOLD},
    }

    output_path = RESULTS_DIR / f"evaluation_results_{VECTORIZATION}_{THRESHOLD}.json"
    with output_path.open("w") as f:
        json.dump(evaluation_result, f, indent=2)

    return overall_metrics


def main():
    stats = get_graph_stats()
    print_graph_stats(stats)

    metrics = run_evaluation()
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.3f}")


if __name__ == "__main__":
    main()
