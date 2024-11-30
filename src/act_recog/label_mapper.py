from pathlib import Path

import pandas as pd


class ActionLabelMapper:
    def __init__(self, mapping_file: Path):
        self.mapping_file = mapping_file
        # (target_dataset, target_action) -> list[(source_dataset, source_action)]
        self.mapping: dict[tuple[str, str], list[tuple[str, str]]] = {}
        self.reverse_mapping: dict[tuple[str, str], str] = {}  # (source_dataset, source_action) -> target_action

    def load_mapping(self, target_dataset: str) -> None:
        mapping_df = pd.read_csv(self.mapping_file)
        self.mapping.clear()
        self.reverse_mapping.clear()

        target_actions = mapping_df[mapping_df["from_dataset"] == target_dataset]

        for _, row in target_actions.iterrows():
            if row["relation"] == "equal":
                target_key = (row["from_dataset"], row["from_action_name"])
                source_pair = (row["to_dataset"], row["to_action_name"])

                if target_key not in self.mapping:
                    self.mapping[target_key] = []
                self.mapping[target_key].append(source_pair)

                self.reverse_mapping[source_pair] = row["from_action_name"]

    def get_target_label(self, source_dataset: str, source_action: str) -> str | None:
        return self.reverse_mapping.get((source_dataset, source_action))
