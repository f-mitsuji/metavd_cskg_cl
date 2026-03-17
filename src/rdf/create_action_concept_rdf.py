from pathlib import Path

from rdflib import Graph, Literal, Namespace
from rdflib.namespace import RDF, RDFS
from tqdm import tqdm

from src.settings import RESULTS_DIR
from src.utils import get_latest_file_path, load_json

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
# THRESHOLD = 0.89
THRESHOLD = 0.9

ACT = Namespace("http://example.org/action/")
DS = Namespace("http://example.org/dataset/")
CN = Namespace("http://conceptnet.io/c/en/")
CN_REL = Namespace("http://conceptnet.io/r/")

API_TIMEOUT = 10
SLEEP_TIME = 1
NOT_FOUND = 404


def bind_namespaces(g: Graph) -> None:
    g.bind("act", ACT)
    g.bind("ds", DS)
    g.bind("cn", CN)
    g.bind("cn_rel", CN_REL)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)


def create_action_concept_graph(data: dict[str, list[dict[str, str]]], dataset_name: str) -> Graph:
    g = Graph()
    bind_namespaces(g)

    dataset_uri = DS[dataset_name]
    g.add((dataset_uri, RDF.type, DS.Dataset))
    g.add((dataset_uri, RDFS.label, Literal(dataset_name)))

    total_actions = len(data)
    with tqdm(total=total_actions, desc=f"Processing {dataset_name}", unit="action") as pbar:
        for action, concepts_with_sim in data.items():
            action_uri = ACT[action.replace(" ", "_")]
            g.add((action_uri, RDF.type, ACT.Action))
            g.add((action_uri, RDFS.label, Literal(action)))
            g.add((action_uri, DS.belongsTo, dataset_uri))

            filtered_concepts = [entry for entry in concepts_with_sim if float(entry["similarity"]) >= THRESHOLD]

            for concept_data in filtered_concepts:
                concept = concept_data["concept"]
                concept_uri = CN[concept.replace(" ", "_").lower()]
                g.add((action_uri, ACT.relatedToConcept, concept_uri))
                g.add((concept_uri, RDF.type, CN.Concept))
                g.add((concept_uri, RDFS.label, Literal(concept)))

            pbar.update(1)

    return g


def combine_datasets_to_rdf(datasets: list[str], output_dir: Path) -> Graph:
    combined_graph = Graph()
    bind_namespaces(combined_graph)

    total_datasets = len(datasets)
    for i, dataset in enumerate(datasets, start=1):
        result_dir = RESULTS_DIR / dataset / VECTORIZATION
        refined_candidates_path = get_latest_file_path(result_dir, f"{dataset}_refined_candidates_{VECTORIZATION}_")

        data = load_json(refined_candidates_path)
        graph = create_action_concept_graph(data, dataset)
        combined_graph += graph

        print(f"Processed dataset: {dataset} ({i}/{total_datasets})")

    output_path = output_dir / f"action_concept_graph_{VECTORIZATION}_{THRESHOLD}.ttl"
    combined_graph.serialize(destination=output_path, format="turtle")
    print(f"Saved combined RDF to {output_path}")

    return combined_graph


if __name__ == "__main__":
    datasets = ["activitynet", "charades", "hmdb51", "kinetics700", "stair_actions", "ucf101"]
    output_dir = RESULTS_DIR / "ttl"
    output_dir.mkdir(parents=True, exist_ok=True)
    combine_datasets_to_rdf(datasets, output_dir)
