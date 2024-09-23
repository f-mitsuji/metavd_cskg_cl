from itertools import combinations
from pathlib import Path

from rdflib import Graph, Literal, Namespace
from rdflib.namespace import OWL, RDF, RDFS

from src.settings import RESULTS_DIR
from src.utils import get_latest_file_path, load_json

ACT = Namespace("http://example.org/action/")
DS = Namespace("http://example.org/dataset/")
CN = Namespace("http://conceptnet.io/c/en/")


def create_action_concept_graph(data: dict[str, list[str]], dataset_name: str) -> Graph:
    g = Graph()
    g.bind("act", ACT)
    g.bind("ds", DS)
    g.bind("cn", CN)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)

    dataset_uri = DS[dataset_name]
    g.add((dataset_uri, RDF.type, DS.Dataset))
    g.add((dataset_uri, RDFS.label, Literal(dataset_name)))

    for action, concepts in data.items():
        action_uri = ACT[action.replace(" ", "_")]
        g.add((action_uri, RDF.type, ACT.Action))
        g.add((action_uri, RDFS.label, Literal(action)))
        g.add((action_uri, DS.belongsTo, dataset_uri))

        # for i in range(len(concepts) - 1):
        #     concept_uri1 = CN[concepts[i].replace(" ", "_").lower()]
        #     concept_uri2 = CN[concepts[i + 1].replace(" ", "_").lower()]
        #     g.add((concept_uri1, OWL.equivalentClass, concept_uri2))

        for c1, c2 in combinations(concepts, 2):
            concept_uri1 = CN[c1.replace(" ", "_").lower()]
            concept_uri2 = CN[c2.replace(" ", "_").lower()]
            g.add((concept_uri1, OWL.equivalentClass, concept_uri2))

        for concept in concepts:
            concept_uri = CN[concept.replace(" ", "_").lower()]
            g.add((action_uri, ACT.relatedToConcept, concept_uri))
            g.add((concept_uri, RDF.type, CN.Concept))
            g.add((concept_uri, RDFS.label, Literal(concept)))

    return g


def combine_datasets_to_rdf(datasets: list[str], output_dir: Path) -> Graph:
    combined_graph = Graph()
    combined_graph.bind("act", ACT)
    combined_graph.bind("ds", DS)
    combined_graph.bind("cn", CN)

    for dataset in datasets:
        result_dir = RESULTS_DIR / dataset
        refined_candidates_path = get_latest_file_path(result_dir, f"{dataset}_refined_candidates_")

        data = load_json(refined_candidates_path)
        graph = create_action_concept_graph(data, dataset)
        combined_graph += graph

        print(f"Processed dataset: {dataset}")

    output_path = output_dir / "action_concept_graph_with_equivalence.ttl"
    # output_path = output_dir / "action_concept_graph_without_equivalence.ttl"
    combined_graph.serialize(destination=output_path, format="turtle")
    print(f"Saved combined RDF to {output_path}")

    return combined_graph


if __name__ == "__main__":
    datasets = ["activitynet", "charades", "hmdb51", "kinetics700", "stair_actions", "ucf101"]
    output_dir = RESULTS_DIR / "ttl"
    output_dir.mkdir(parents=True, exist_ok=True)
    combine_datasets_to_rdf(datasets, output_dir)
