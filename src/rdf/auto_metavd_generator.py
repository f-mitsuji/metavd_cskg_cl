from collections import defaultdict
from pathlib import Path

import pandas as pd
import rdflib

from src.settings import AUTO_METAVD_DIR, METAVD_DIR, RESULTS_DIR

# VECTORIZATION = "numberbatch"
# VECTORIZATION = "sent2vec"
# VECTORIZATION = "word2vec"
VECTORIZATION = "mpnet"
# THRESHOLD = 0.8
# THRESHOLD = 0.81
# THRESHOLD = 0.82
# THRESHOLD = 0.83
# THRESHOLD = 0.84
# THRESHOLD = 0.85
# THRESHOLD = 0.86
# THRESHOLD = 0.87
# THRESHOLD = 0.88
THRESHOLD = 0.89
# THRESHOLD = 0.9

SPARQL_QUERY = """
PREFIX act: <http://example.org/action/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ds: <http://example.org/dataset/>

SELECT ?action ?action_label ?concept ?concept_label ?dataset ?dataset_label WHERE {
    ?action act:relatedToConcept ?concept .
    ?action rdfs:label ?action_label .
    ?action ds:belongsTo ?dataset .
    ?dataset rdfs:label ?dataset_label .
    OPTIONAL { ?concept rdfs:label ?concept_label }
}
"""


def parse_ttl_and_extract_relations(ttl_file_path, dataset_csv_folder, output_folder):
    """TTLファイルを読み込んで、同じConceptNetノードにリンクしている動作間のequal関係を抽出し、CSVファイルに出力する."""
    g = rdflib.Graph()
    g.parse(ttl_file_path, format="turtle")

    concept_to_actions = defaultdict(list)

    results = g.query(SPARQL_QUERY)

    print(f"Found {len(results)} action-concept relations")

    # 結果を処理
    for row in results:
        action_uri = str(row.action)
        action_label = str(row.action_label)
        concept_uri = str(row.concept)
        concept_label = str(row.concept_label) if row.concept_label else concept_uri.split("/")[-1]
        dataset_uri = str(row.dataset)
        dataset_label = str(row.dataset_label)

        concept_to_actions[concept_uri].append(
            {
                "action_name": action_label,
                "action_uri": action_uri,
                "dataset": dataset_label,
                "concept_label": concept_label,
            }
        )

    print(f"Found {len(concept_to_actions)} unique concepts")

    dataset_mappings = {}
    csv_folder = Path(dataset_csv_folder)

    available_datasets = set()
    for concept_uri, actions in concept_to_actions.items():
        for action in actions:
            available_datasets.add(action["dataset"])

    print(f"Datasets found in TTL: {sorted(available_datasets)}")

    for dataset_name in available_datasets:
        csv_file = csv_folder / f"{dataset_name}_classes.csv"
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                if "idx" in df.columns and "name" in df.columns:
                    mapping = dict(zip(df["name"], df["idx"], strict=False))
                    dataset_mappings[dataset_name] = mapping
                    print(f"Loaded {len(mapping)} actions from {dataset_name}")
                else:
                    print(f"Warning: {csv_file} doesn't have 'idx' and 'name' columns")
                    print(f"Available columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
        else:
            print(f"Warning: CSV file not found for dataset {dataset_name}: {csv_file}")

    relations = []
    relation_id = 0
    seen_pairs = set()

    concepts_with_multiple_actions = 0
    for concept_uri, actions in concept_to_actions.items():
        if len(actions) > 1:
            concepts_with_multiple_actions += 1
            concept_label = actions[0]["concept_label"]
            print(f"\nConcept: {concept_label} ({concept_uri.split('/')[-1]})")
            print(f"Actions: {[(a['action_name'], a['dataset']) for a in actions]}")

            for i, action1 in enumerate(actions):
                for j, action2 in enumerate(actions):
                    if i != j:
                        dataset1 = action1["dataset"]
                        dataset2 = action2["dataset"]
                        action_name1 = action1["action_name"]
                        action_name2 = action2["action_name"]

                        pair_key = tuple(sorted([(dataset1, action_name1), (dataset2, action_name2)]))

                        if pair_key in seen_pairs:
                            continue

                        idx1 = dataset_mappings.get(dataset1, {}).get(action_name1)
                        idx2 = dataset_mappings.get(dataset2, {}).get(action_name2)

                        if idx1 is not None and idx2 is not None:
                            relations.append(
                                {
                                    "": relation_id,
                                    "from_dataset": dataset1,
                                    "from_action_idx": idx1,
                                    "from_action_name": action_name1,
                                    "to_dataset": dataset2,
                                    "to_action_idx": idx2,
                                    "to_action_name": action_name2,
                                    "relation": "equal",
                                }
                            )
                            relation_id += 1

                            relations.append(
                                {
                                    "": relation_id,
                                    "from_dataset": dataset2,
                                    "from_action_idx": idx2,
                                    "from_action_name": action_name2,
                                    "to_dataset": dataset1,
                                    "to_action_idx": idx1,
                                    "to_action_name": action_name1,
                                    "relation": "equal",
                                }
                            )
                            relation_id += 1

                            # 処理済みとしてマーク
                            seen_pairs.add(pair_key)
                        else:
                            if idx1 is None:
                                print(f"Warning: Could not find idx for '{action_name1}' in {dataset1}")
                            if idx2 is None:
                                print(f"Warning: Could not find idx for '{action_name2}' in {dataset2}")

    print(f"\nFound {concepts_with_multiple_actions} concepts with multiple actions")

    if relations:
        df_relations = pd.DataFrame(relations)
        output_path = Path(output_folder) / f"auto_metavd_{VECTORIZATION}_{THRESHOLD}.csv"
        df_relations.to_csv(output_path, index=False)
        print(f"\nExtracted {len(relations)} relations")
        print(f"Results saved to {output_path}")

        print("\nStatistics:")
        print(f"Total relations: {len(relations)}")
        datasets = set()
        for rel in relations:
            datasets.add(rel["from_dataset"])
            datasets.add(rel["to_dataset"])
        print(f"Datasets involved: {sorted(datasets)}")

        dataset_pairs = defaultdict(int)
        for rel in relations:
            ds_pair = tuple(sorted([rel["from_dataset"], rel["to_dataset"]]))
            dataset_pairs[ds_pair] += 1

        print("\nRelations by dataset pair:")
        for (ds1, ds2), count in sorted(dataset_pairs.items()):
            if ds1 == ds2:
                print(f"  Within {ds1}: {count}")
            else:
                print(f"  {ds1} <-> {ds2}: {count}")

        return output_path
    else:
        print("\nNo relations found")
        return None


def main():
    ttl_file_path = RESULTS_DIR / "ttl" / f"action_concept_graph_{VECTORIZATION}_{THRESHOLD}.ttl"
    dataset_csv_folder = METAVD_DIR
    output_folder = AUTO_METAVD_DIR

    print("Extracting relations from TTL file...")
    output_path = parse_ttl_and_extract_relations(ttl_file_path, dataset_csv_folder, output_folder)

    if output_path:
        print("\nProcess completed successfully!")
        print(f"Output file: {output_path}")
    else:
        print("Process completed but no relations were found.")


if __name__ == "__main__":
    main()
