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
from src.settings import LOGS_DIR, METAVD_DIR, TRAINED_MODELS_DIR
from src.utils import get_current_jst_timestamp, setup_logger


class ATPGEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        label_mapper: ActionLabelMapper,
        target_dataset: str,
        logger: logging.Logger | None = None,
    ):
        self.model = model
        self.device = device
        self.label_mapper = label_mapper
        self.target_dataset = target_dataset
        self.logger = logger
        self.model.eval()

    def calculate_precision(
        self,
        source_dataset: ActionRecognitionDataset,
        dataloader: DataLoader,
    ) -> dict[str, dict[str, float]]:
        """AXに対応するBYラベルの動画について、AXと予測される割合を計算"""
        # AXごとの予測結果を格納
        predictions_per_label = defaultdict(lambda: defaultdict(list))

        with torch.no_grad():
            for videos, _ in dataloader:
                videos = videos.to(self.device)
                outputs = self.model(videos)
                pred_indices = outputs.argmax(dim=1)

                # 各バッチの動画について
                for i, pred_idx in enumerate(pred_indices):
                    video_info = source_dataset.videos[i]
                    source_label = video_info.original_label
                    source_dataset_name = source_dataset.dataset_name

                    # この動画の元のラベル（BY）に対応するターゲットラベル（AX）を取得
                    target_label = self.label_mapper.get_target_label(source_dataset_name, source_label)
                    if target_label is None:
                        continue

                    # target_labelと予測が一致するか確認
                    is_correct = pred_idx.item() == source_dataset.label_to_idx[target_label]
                    predictions_per_label[target_label][(source_dataset_name, source_label)].append(is_correct)

        # Precisionの計算
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

    def calculate_atpg(self, source_dataset: ActionRecognitionDataset, tau: float = 0.1) -> dict[str, float]:
        """Average Transferred Precision Gainの計算"""
        source_loader = DataLoader(source_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        # 各ラベルペアのPrecisionを計算
        precision_scores = self.calculate_precision(source_dataset, source_loader)

        # ATPGの計算
        atpg_scores = {}
        for target_label, source_precisions in precision_scores.items():
            gains = []
            for _, precision in source_precisions.items():
                gain = max(precision - tau, 0)
                gains.append(gain)

            if gains:  # S(AX)が空でない場合
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

    label_mapper = ActionLabelMapper(METAVD_DIR / "metavd_v1.csv")
    label_mapper.load_mapping(config.target_dataset)

    evaluator = ATPGEvaluator(
        model=model, device=device, label_mapper=label_mapper, target_dataset=config.target_dataset, logger=logger
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

        atpg_scores = evaluator.calculate_atpg(source_dataset=source_dataset)

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
        source_datasets=["ucf101", "kinetics700", "charades"],
    )

    model_path = (
        TRAINED_MODELS_DIR / "hmdb51_from_no_source_e45_lr1_0000000000000002e-06_acc67_65_20241130_201818_final.pth"
        # TRAINED_MODELS_DIR / "ucf101_from_no_source_e45_lr1_0000000000000002e-06_acc91_88_20241201_194619_final.pth"
    )
    results = evaluate_model(model_path=model_path, config=config, logger=logger)


if __name__ == "__main__":
    main()
