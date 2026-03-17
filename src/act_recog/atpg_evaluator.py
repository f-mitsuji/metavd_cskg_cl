import logging
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.video import r2plus1d_18

from src.act_recog.config import ExperimentConfig
from src.act_recog.datasets import ActionRecognitionDataset
from src.act_recog.label_mapper import ActionLabelMapper
from src.act_recog.trainer import get_dataset_class

# from src.settings import AUTO_METAVD_DIR, LOGS_DIR, TRAINED_MODELS_DIR
from src.settings import LOGS_DIR, METAVD_DIR, TRAINED_MODELS_DIR
from src.utils import get_current_jst_timestamp, setup_logger


class ATPGEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        label_mapper: ActionLabelMapper,
        target_dataset: str,
        target_label_to_idx: dict[str, int],
        logger: logging.Logger | None = None,
    ):
        self.model = model
        self.device = device
        self.label_mapper = label_mapper
        self.target_dataset = target_dataset
        self.target_label_to_idx = target_label_to_idx
        self.logger = logger
        self.model.eval()

    def calculate_target_accuracy(
        self,
        target_dataset: ActionRecognitionDataset,
        target_loader: DataLoader,
    ) -> float:
        correct = 0
        total = 0

        with torch.no_grad():
            for videos, labels in target_loader:
                videos = videos.to(self.device)
                outputs = self.model(videos)
                pred_indices = outputs.argmax(dim=1)

                correct += (pred_indices == labels.to(self.device)).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        if self.logger:
            self.logger.info(f"Target dataset accuracy: {accuracy:.3f}")

        return accuracy

    def calculate_precision(
        self,
        source_dataset: ActionRecognitionDataset,
        dataloader: DataLoader,
    ) -> dict[str, dict[str, float]]:
        if self.logger:
            self.logger.info("Target dataset label mapping:")
            for label, idx in sorted(self.target_label_to_idx.items()):
                self.logger.info(f"  {label}: {idx}")

        predictions_per_label = defaultdict(lambda: defaultdict(list))

        if self.logger:
            label_counts = defaultdict(int)
            for video in source_dataset.videos:
                target_label = self.label_mapper.get_target_label(source_dataset.dataset_name, video.original_label)
                if target_label:
                    label_counts[target_label] += 1

            self.logger.info("\nSource dataset statistics:")
            for label, count in sorted(label_counts.items()):
                self.logger.info(f"  {label} <- {count} videos")

        with torch.no_grad():
            total_processed = 0

            for videos, _ in dataloader:
                videos = videos.to(self.device)
                outputs = self.model(videos)
                pred_indices = outputs.argmax(dim=1)

                for i in range(len(videos)):
                    global_idx = total_processed + i
                    if global_idx >= len(source_dataset.videos):
                        break

                    video_info = source_dataset.videos[global_idx]
                    source_label = video_info.original_label
                    source_dataset_name = source_dataset.dataset_name

                    target_label = self.label_mapper.get_target_label(source_dataset_name, source_label)
                    if target_label is None:
                        continue

                    if (
                        self.logger
                        and len(predictions_per_label[target_label][(source_dataset_name, source_label)]) == 0
                    ):
                        self.logger.info(f"Processing first {target_label} video:")
                        self.logger.info(f"  Video index: {global_idx}")
                        self.logger.info(f"  Original label: {source_label}")
                        self.logger.info(f"  Predicted index: {pred_indices[i].item()}")
                        self.logger.info(f"  Expected index: {self.target_label_to_idx[target_label]}")

                    is_correct = pred_indices[i].item() == self.target_label_to_idx[target_label]
                    predictions_per_label[target_label][(source_dataset_name, source_label)].append(is_correct)

                total_processed += len(videos)

        if self.logger:
            self.logger.info("\nCollected predictions summary:")
            for target_label in sorted(predictions_per_label.keys()):
                for (source_dataset_name, source_label), predictions in predictions_per_label[target_label].items():
                    self.logger.info(
                        f"{target_label} <- {source_dataset_name}/{source_label}: "
                        f"{len(predictions)} predictions, "
                        f"{sum(predictions)} correct"
                    )

        precision_scores = {}
        for target_label, source_predictions in predictions_per_label.items():
            precision_scores[target_label] = {}
            for (source_dataset_name, source_label), predictions in source_predictions.items():
                if predictions:
                    precision = sum(predictions) / len(predictions)
                    precision_scores[target_label][(source_dataset_name, source_label)] = precision

                    if self.logger:
                        self.logger.info(
                            f"Precision for {target_label} <- {source_dataset_name}/{source_label}: "
                            f"{precision:.3f} ({sum(predictions)}/{len(predictions)})"
                        )

        return precision_scores

    def calculate_atpg(
        self,
        source_dataset: ActionRecognitionDataset,
        target_dataset: ActionRecognitionDataset,
        tau_ratio: float = 0.8,
    ) -> dict[str, float]:
        target_loader = DataLoader(target_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        target_accuracy = self.calculate_target_accuracy(target_dataset, target_loader)

        # τを設定（精度の80%）
        tau = target_accuracy * tau_ratio
        if self.logger:
            self.logger.info(f"Setting tau to {tau:.3f} (target accuracy: {target_accuracy:.3f} × {tau_ratio})")

        source_loader = DataLoader(source_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        precision_scores = self.calculate_precision(source_dataset, source_loader)

        atpg_scores = {}
        for target_label, source_precisions in precision_scores.items():
            gains = []
            for _, precision in source_precisions.items():
                gain = max(precision - tau, 0)
                gains.append(gain)

            if gains:
                atpg_score = sum(gains) / len(gains)
                atpg_scores[target_label] = atpg_score

                if self.logger:
                    self.logger.info(f"ATPG for {target_label}: {atpg_score:.3f}")

        return atpg_scores


def evaluate_model(
    model_path: Path, config: ExperimentConfig, logger: logging.Logger | None = None
) -> dict[str, dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if logger:
        logger.info(f"Using device: {device}")

    if logger:
        logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    num_classes = checkpoint["model"]["fc.weight"].size(0)
    model = r2plus1d_18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)

    # label_mapper = ActionLabelMapper(AUTO_METAVD_DIR / "auto_metavd.csv")
    label_mapper = ActionLabelMapper(METAVD_DIR / "metavd_v1.csv")
    label_mapper.load_mapping(config.target_dataset)
    print(label_mapper.mapping)
    # sys.exit(0)

    target_dataset_class = get_dataset_class(config.target_dataset)
    target_dataset = target_dataset_class(
        split="test",
        sampling_frames=config.data.sampling_frames,
        label_mapper=None,
        is_target=True,
        logger=logger,
    )

    evaluator = ATPGEvaluator(
        model=model,
        device=device,
        label_mapper=label_mapper,
        target_dataset=config.target_dataset,
        target_label_to_idx=target_dataset.label_to_idx,
        logger=logger,
    )

    results = {}
    for source_name in config.source_datasets:
        if logger:
            logger.info(f"\nEvaluating ATPG for source dataset: {source_name}")

        source_dataset_class = get_dataset_class(source_name)
        source_dataset = source_dataset_class(
            split="test",
            sampling_frames=config.data.sampling_frames,
            label_mapper=label_mapper,
            is_target=False,
            logger=logger,
        )

        atpg_scores = evaluator.calculate_atpg(source_dataset=source_dataset, target_dataset=target_dataset)

        if logger:
            logger.info(f"\nATPG Scores for {source_name}:")
            avg_score = sum(atpg_scores.values()) / len(atpg_scores) if atpg_scores else 0
            logger.info(f"Average ATPG: {avg_score:.3f}")

        results[source_name] = atpg_scores

    return results


def main():
    timestamp = get_current_jst_timestamp()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"atpg_evaluation_{timestamp}.log"
    logger = setup_logger("atpg_eval", log_file)
    logger.info("Starting ATPG evaluation")

    config = ExperimentConfig(
        target_dataset="hmdb51",
        # target_dataset="ucf101",
        # target_dataset="stair_actions",
        # target_dataset="activitynet",
        # target_dataset="charades",
        # source_datasets=["ucf101", "kinetics700", "charades", "stair_actions", "activitynet"],  # hmdb51
        # source_datasets=["hmdb51", "kinetics700", "charades", "stair_actions", "activitynet"],  # ucf101
        # source_datasets=["hmdb51", "ucf101", "kinetics700", "charades", "activitynet"],  # stair_actions
        # source_datasets=["hmdb51", "ucf101", "kinetics700", "charades", "stair_actions"],  # activitynet
        # source_datasets=["hmdb51", "ucf101", "kinetics700", "activitynet", "stair_actions"], # charades
        source_datasets=["ucf101,"],
    )

    model_path = (
        TRAINED_MODELS_DIR / "hmdb51_from_no_source_e45_lr1_0000000000000002e-06_acc67_65_20241130_201818_final.pth"
        # TRAINED_MODELS_DIR / "ucf101_from_no_source_e45_lr1_0000000000000002e-06_acc91_88_20241201_194619_final.pth"
        # TRAINED_MODELS_DIR
        # / "stair_actions_from_no_source_e45_lr1_0000000000000002e-06_acc83_62_20241211_082208_final.pth"
        # TRAINED_MODELS_DIR
        # / "activitynet_from_no_source_e45_lr1_0000000000000002e-06_acc66_78_20250131_221626_final.pth"
        # TRAINED_MODELS_DIR / "charades_from_no_source_e45_lr1_0000000000000002e-06_acc9_81_20241111_002737_final.pth"
    )
    results = evaluate_model(model_path=model_path, config=config, logger=logger)


if __name__ == "__main__":
    main()
