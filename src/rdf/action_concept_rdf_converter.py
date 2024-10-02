from itertools import combinations
from pathlib import Path
from time import sleep

import requests
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, SKOS
from tqdm import tqdm

from src.settings import RESULTS_DIR
from src.utils import get_latest_file_path, load_json

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
    g.bind("skos", SKOS)


def get_conceptnet_synonyms(concept: str) -> list[str]:
    url = f"http://api.conceptnet.io/query?node=/c/en/{concept.replace(' ', '_')}&other=/c/en&rel=/r/Synonym"

    try:
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            if data["error"].get("status") == NOT_FOUND:
                print(f"Concept '{concept}' does not exist in ConceptNet.")
            else:
                print(f"Error querying ConceptNet for '{concept}': {data['error']}")
            return []

        return list(
            {
                edge["start"]["label"] if edge["start"]["label"].lower() != concept.lower() else edge["end"]["label"]
                for edge in data.get("edges", [])
            }
        )
    except requests.RequestException as e:
        print(f"Error making request to ConceptNet for '{concept}': {e}")
        return []


def add_conceptnet_synonyms(g: Graph, concept_uri: URIRef, concept: str):
    synonyms = get_conceptnet_synonyms(concept)
    for synonym in synonyms:
        synonym_uri = CN[synonym.replace(" ", "_")]

        g.add((synonym_uri, RDF.type, CN.Concept))
        g.add((synonym_uri, RDFS.label, Literal(synonym)))
        g.add((concept_uri, CN_REL.Synonym, synonym_uri))
        g.add((synonym_uri, CN_REL.Synonym, concept_uri))


def create_action_concept_graph(data: dict[str, list[str]], dataset_name: str) -> Graph:
    g = Graph()
    bind_namespaces(g)

    dataset_uri = DS[dataset_name]
    g.add((dataset_uri, RDF.type, DS.Dataset))
    g.add((dataset_uri, RDFS.label, Literal(dataset_name)))

    total_actions = len(data)
    with tqdm(total=total_actions, desc=f"Processing {dataset_name}", unit="action") as pbar:
        for action, concepts in data.items():
            action_uri = ACT[action.replace(" ", "_")]
            g.add((action_uri, RDF.type, ACT.Action))
            g.add((action_uri, RDFS.label, Literal(action)))
            g.add((action_uri, DS.belongsTo, dataset_uri))

            for c1, c2 in combinations(concepts, 2):
                concept_uri1 = CN[c1.replace(" ", "_").lower()]
                concept_uri2 = CN[c2.replace(" ", "_").lower()]
                g.add((concept_uri1, SKOS.exactMatch, concept_uri2))
                g.add((concept_uri2, SKOS.exactMatch, concept_uri1))

            for concept in concepts:
                concept_uri = CN[concept.replace(" ", "_").lower()]
                g.add((action_uri, ACT.relatedToConcept, concept_uri))
                g.add((concept_uri, RDF.type, CN.Concept))
                g.add((concept_uri, RDFS.label, Literal(concept)))

                add_conceptnet_synonyms(g, concept_uri, concept)
                sleep(SLEEP_TIME)

            pbar.update(1)

    return g


def combine_datasets_to_rdf(datasets: list[str], output_dir: Path) -> Graph:
    combined_graph = Graph()
    bind_namespaces(combined_graph)

    total_datasets = len(datasets)
    for i, dataset in enumerate(datasets, start=1):
        result_dir = RESULTS_DIR / dataset
        refined_candidates_path = get_latest_file_path(result_dir, f"{dataset}_refined_candidates_")

        data = load_json(refined_candidates_path)
        graph = create_action_concept_graph(data, dataset)
        combined_graph += graph

        print(f"Processed dataset: {dataset} ({i}/{total_datasets})")

    output_path = output_dir / "action_concept_graph_with_equivalence_synonym.ttl"
    combined_graph.serialize(destination=output_path, format="turtle")
    print(f"Saved combined RDF to {output_path}")

    return combined_graph


if __name__ == "__main__":
    datasets = ["activitynet", "charades", "hmdb51", "kinetics700", "stair_actions", "ucf101"]
    output_dir = RESULTS_DIR / "ttl"
    output_dir.mkdir(parents=True, exist_ok=True)
    combine_datasets_to_rdf(datasets, output_dir)
