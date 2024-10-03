import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import TypeAlias

from src.settings import DATA_DIR, METAVD_DIR

ActionPair: TypeAlias = tuple[str, str]
RelationData: TypeAlias = dict[str, str]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def read_relation_data(file_path: Path) -> list[RelationData]:
    try:
        with file_path.open(encoding="utf-8") as file:
            return list(csv.DictReader(file))
    except FileNotFoundError:
        logging.exception(f"File not found: {file_path}")
        raise
    except csv.Error:
        logging.exception("Error occurred while reading CSV file")
        raise


def build_equal_relation_graph(relations: list[RelationData]) -> dict[ActionPair, set[ActionPair]]:
    graph = defaultdict(set)
    for relation in relations:
        if relation["relation"] == "equal":
            from_action = (relation["from_dataset"], relation["from_action_name"])
            to_action = (relation["to_dataset"], relation["to_action_name"])
            graph[from_action].add(to_action)
            graph[to_action].add(from_action)
    return graph


def generate_equal_action_sets(graph: dict[ActionPair, set[ActionPair]]) -> list[set[ActionPair]]:
    def dfs(node: ActionPair, component: set[ActionPair]):
        component.add(node)
        for neighbor in graph[node]:
            if neighbor not in component:
                dfs(neighbor, component)

    equal_sets = []
    visited = set()

    for action in graph:
        if action not in visited:
            component: set[ActionPair] = set()
            dfs(action, component)
            if len(component) > 1:
                equal_sets.append(component)
            visited.update(component)

    return equal_sets


def format_equal_sets(equal_sets: list[set[ActionPair]]) -> list[dict]:
    return [
        {"set_id": i, "actions": [{"dataset": dataset, "action": action} for dataset, action in sorted(action_set)]}
        for i, action_set in enumerate(equal_sets, 1)
    ]


def save_to_json(data: list[dict], output_file: Path):
    try:
        with output_file.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
    except OSError:
        logging.exception("Error occurred while writing JSON file")
        raise


def generate_action_equal_sets(input_file: Path, output_file: Path) -> None:
    try:
        relations = read_relation_data(input_file)
        graph = build_equal_relation_graph(relations)
        equal_sets = generate_equal_action_sets(graph)
        formatted_data = format_equal_sets(equal_sets)
        save_to_json(formatted_data, output_file)
        logging.info(f"Generated equal sets have been saved to {output_file}")
    except Exception:
        logging.exception("An error occurred during processing")
        raise


if __name__ == "__main__":
    input_csv = METAVD_DIR / "metavd_v1.csv"
    output_json = DATA_DIR / "action_equal_sets.json"
    generate_action_equal_sets(input_csv, output_json)
