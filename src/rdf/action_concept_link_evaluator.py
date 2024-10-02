import logging
import random
import textwrap

from SPARQLWrapper import JSON, SPARQLWrapper
from SPARQLWrapper.SPARQLExceptions import SPARQLWrapperException

from src.settings import DATA_DIR, RESULTS_DIR
from src.utils import get_latest_file_path, load_json, save_json_with_timestamp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FUSEKI_ENDPOINT = "http://192.168.0.21:3030/action_concept_graph_with_equivalence/query"

PREFIXES = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX act: <http://example.org/action/>
PREFIX ds: <http://example.org/dataset/>
PREFIX cn: <http://conceptnet.io/c/en/>
PREFIX cn_rel: <http://conceptnet.io/r/>
"""


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
    query = textwrap.dedent("""
    SELECT DISTINCT ?action_label ?dataset_label
    WHERE {{
        ?input_concept rdfs:label "{concept}" .
        ?action act:relatedToConcept ?related_concept ;
                rdfs:label ?action_label ;
                ds:belongsTo ?dataset .
        ?dataset rdfs:label ?dataset_label .
        VALUES ?related_concept {{ ?input_concept ?exact_match }}
        OPTIONAL {{
            ?input_concept skos:exactMatch ?exact_match .
        }}
    }}
    """)
    # query = textwrap.dedent(f"""
    # SELECT DISTINCT ?action_label ?dataset_label
    # WHERE {{
    #     ?action a act:Action ;
    #         rdfs:label ?action_label ;
    #         ds:belongsTo ?dataset ;
    #         act:relatedToConcept ?related_concept .
    #     ?dataset rdfs:label ?dataset_label .
    #     ?related_concept owl:equivalentClass ?input_concept .
    #     ?input_concept rdfs:label "{concept}" .
    # }}
    # ORDER BY ?dataset_label ?action_label
    # """)

    results = execute_sparql_query(query)
    return [(result["action_label"]["value"], result["dataset_label"]["value"]) for result in results]


def load_refined_candidates(dataset: str) -> dict[str, list[str]]:
    result_dir = RESULTS_DIR / dataset
    refined_candidates_path = get_latest_file_path(result_dir, f"{dataset}_refined_candidates_")
    return load_json(refined_candidates_path)


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
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": list(true_positives),
        "false_positives": list(false_positives),
        "false_negatives": list(false_negatives),
    }


def main():
    stats = get_graph_stats()
    print_graph_stats(stats)

    equal_sets = load_json(DATA_DIR / "action_equal_sets.json")
    evaluation_results = []

    for equal_set in equal_sets:
        random_action = random.choice(equal_set["actions"])
        dataset, action = random_action["dataset"], random_action["action"]

        refined_candidates = load_refined_candidates(dataset)
        concepts = refined_candidates.get(action, [])

        if concepts:
            random_concept = random.choice(concepts)
            query_results = query_actions_by_concept(random_concept)
            ground_truth = [(action["action"], action["dataset"]) for action in equal_set["actions"]]

            result = evaluate_results(query_results, ground_truth)
            result.update(
                {
                    "action": action,
                    "dataset": dataset,
                    "concept": random_concept,
                    "set_id": equal_set["set_id"],
                }
            )
            evaluation_results.append(result)

            if (
                "false_positives" in result
                and isinstance(result["false_positives"], list)
                and result["false_positives"]
            ):
                print(f"set_id: {equal_set['set_id']}, action: {action}, FP: {result['false_positives']}")

    overall_metrics = {
        "precision": sum(r["precision"] for r in evaluation_results) / len(evaluation_results),
        "recall": sum(r["recall"] for r in evaluation_results) / len(evaluation_results),
        "f1": sum(r["f1"] for r in evaluation_results) / len(evaluation_results),
    }

    print("\nOverall Evaluation:")
    for metric, value in overall_metrics.items():
        print(f"Average {metric.capitalize()}: {value:.4f}")

    output = {
        "overall_metrics": overall_metrics,
        "evaluations": evaluation_results,
    }

    save_json_with_timestamp(output, RESULTS_DIR, "evaluation_results", msg="Evaluation results ")


if __name__ == "__main__":
    main()
