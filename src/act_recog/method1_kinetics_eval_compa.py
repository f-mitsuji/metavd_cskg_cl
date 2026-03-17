"""第3のデータセットに対する汎化性能評価実験.

拡張元データセット（少ない方）の全量に対して、
拡張用データセット（多い方）の追加量を割合で変化させ、
第3のデータセット（Kinetics700）での汎化性能を評価する。
"""

import csv
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src.act_recog.config import DataConfig, TrainingConfig
from src.act_recog.datasets import (
    ActionRecognitionDataset,
    ActivityNetDataset,
    CharadesDataset,
    HMDB51Dataset,
    Kinetics700Dataset,
    STAIRActionsDataset,
    UCF101Dataset,
)
from src.act_recog.label_mapper import ActionLabelMapper
from src.act_recog.models.r2plus1d import R2Plus1DModel
from src.settings import (
    AUTO_METAVD_DIR,
    LOGS_DIR,
    TRAINED_MODELS_DIR,
)
from src.utils import get_current_jst_timestamp, setup_logger


@dataclass(frozen=True)
class CrossDatasetConfig:
    """クロスデータセット訓練の設定"""

    primary_dataset: str
    auxiliary_dataset: str
    auxiliary_percent: float
    evaluation_dataset: str = "kinetics700"
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)


@dataclass
class ExperimentResult:
    primary_dataset: str
    auxiliary_dataset: str
    auxiliary_percent: float
    actual_percent: float
    num_primary: int
    num_auxiliary: int
    num_total: int
    num_classes: int
    train_acc: float
    val_acc: float
    kinetics_acc: float
    kinetics_loss: float
    model_path: str
    timestamp: str


class CommonLabelExtractor:
    def __init__(self, mapping_file: Path, logger: logging.Logger | None = None):
        self.mapping_file = mapping_file
        self.logger = logger

    def extract_common_labels(self, dataset_a: str, dataset_b: str) -> dict[str, str]:
        mapping_df = pd.read_csv(self.mapping_file)

        # dataset_a -> dataset_b の equal 関係を抽出
        common_labels_ab = {}
        forward_mapping = mapping_df[
            (mapping_df["from_dataset"] == dataset_a)
            & (mapping_df["to_dataset"] == dataset_b)
            & (mapping_df["relation"] == "equal")
        ]

        for _, row in forward_mapping.iterrows():
            common_labels_ab[row["from_action_name"]] = row["to_action_name"]

        common_labels_ba = {}
        backward_mapping = mapping_df[
            (mapping_df["from_dataset"] == dataset_b)
            & (mapping_df["to_dataset"] == dataset_a)
            & (mapping_df["relation"] == "equal")
        ]

        for _, row in backward_mapping.iterrows():
            common_labels_ba[row["to_action_name"]] = row["from_action_name"]

        all_common_labels = {}
        for a_label, b_label in common_labels_ab.items():
            all_common_labels[a_label] = b_label

        for a_label, b_label in common_labels_ba.items():
            if a_label not in all_common_labels:
                all_common_labels[a_label] = b_label

        if self.logger:
            self.logger.info(f"\n=== Common Labels between {dataset_a} and {dataset_b} ===")
            self.logger.info(f"Total common labels: {len(all_common_labels)}")
            for a_label, b_label in sorted(all_common_labels.items()):
                self.logger.info(f"  {dataset_a}:{a_label} <-> {dataset_b}:{b_label}")

        return all_common_labels

    def extract_triple_common_labels(self, dataset_a: str, dataset_b: str, dataset_c: str) -> dict[str, dict[str, str]]:
        """3つのデータセット間で共通するラベルを抽出."""
        common_ab = self.extract_common_labels(dataset_a, dataset_b)
        common_ac = self.extract_common_labels(dataset_a, dataset_c)

        triple_common = {}
        for a_label in common_ab.keys():
            if a_label in common_ac:
                triple_common[a_label] = {
                    dataset_b: common_ab[a_label],
                    dataset_c: common_ac[a_label],
                }

        if self.logger:
            self.logger.info(f"\n=== Triple Common Labels: {dataset_a}, {dataset_b}, {dataset_c} ===")
            self.logger.info(f"Total common labels: {len(triple_common)}")
            for a_label, mappings in sorted(triple_common.items()):
                b_label = mappings[dataset_b]
                c_label = mappings[dataset_c]
                self.logger.info(f"  {dataset_a}:{a_label} <-> {dataset_b}:{b_label} <-> {dataset_c}:{c_label}")

        return triple_common


class FilteredDataset(Dataset):
    """共通動作ラベルのみでフィルタリングされたデータセット."""

    def __init__(
        self,
        base_dataset: ActionRecognitionDataset,
        allowed_labels: set[str],
        logger: logging.Logger | None = None,
    ):
        self.base_dataset = base_dataset
        self.allowed_labels = allowed_labels
        self.logger = logger

        self.filtered_indices: list[int] = []
        self.filtered_labels: list[str] = []

        for idx in range(len(base_dataset)):
            video_info = base_dataset.videos[idx]
            if video_info.label in allowed_labels:
                self.filtered_indices.append(idx)
                self.filtered_labels.append(video_info.label)

        unique_labels = sorted(set(self.filtered_labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        if self.logger:
            self.logger.info("\n=== Filtered Dataset Statistics ===")
            self.logger.info(f"Original dataset size: {len(base_dataset)}")
            self.logger.info(f"Filtered dataset size: {len(self.filtered_indices)}")
            self.logger.info(f"Number of classes: {len(unique_labels)}")

            label_counts: dict[str, int] = {}
            for label in self.filtered_labels:
                label_counts[label] = label_counts.get(label, 0) + 1

            self.logger.info("\nSamples per class:")
            for label, count in sorted(label_counts.items()):
                self.logger.info(f"  {label}: {count}")

    def __len__(self) -> int:
        return len(self.filtered_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        original_idx = self.filtered_indices[idx]
        video, _ = self.base_dataset[original_idx]

        label = self.filtered_labels[idx]
        label_idx = self.label_to_idx[label]

        return video, label_idx


class CrossDatasetTrainingSet(Dataset):
    def __init__(
        self,
        primary_dataset: FilteredDataset,
        auxiliary_dataset: FilteredDataset | None,
        auxiliary_percent: float,
        logger: logging.Logger | None = None,
        seed: int = 42,
    ):
        self.primary_dataset = primary_dataset
        self.auxiliary_dataset = auxiliary_dataset
        self.label_to_idx = primary_dataset.label_to_idx
        self.logger = logger

        random.seed(seed)

        self.primary_indices = list(range(len(primary_dataset)))
        self.num_primary = len(self.primary_indices)

        self.auxiliary_indices: list[int] = []
        self.num_auxiliary = 0
        self.max_auxiliary = 0
        self.actual_percent = 0.0

        if auxiliary_dataset is not None and auxiliary_percent > 0:
            self.max_auxiliary = len(auxiliary_dataset)

            desired_samples = int(self.num_primary * auxiliary_percent / 100)

            actual_samples = min(desired_samples, self.max_auxiliary)

            if actual_samples > 0:
                all_indices = list(range(self.max_auxiliary))
                self.auxiliary_indices = random.sample(all_indices, actual_samples)
                self.num_auxiliary = len(self.auxiliary_indices)
                self.actual_percent = 100 * self.num_auxiliary / self.num_primary

        if self.logger:
            self.logger.info("\n=== Cross-Dataset Training Set ===")
            self.logger.info(f"Primary: {self.num_primary} samples (100%)")
            self.logger.info(f"Auxiliary available: {self.max_auxiliary} samples")
            self.logger.info(
                f"Auxiliary requested: {auxiliary_percent:.0f}% of Primary "
                f"= {int(self.num_primary * auxiliary_percent / 100)} samples"
            )
            self.logger.info(
                f"Auxiliary actual: {self.num_auxiliary} samples " f"({self.actual_percent:.1f}% of Primary)"
            )
            self.logger.info(f"Total: {len(self)} samples")

    def __len__(self) -> int:
        return self.num_primary + self.num_auxiliary

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if idx < self.num_primary:
            return self.primary_dataset[self.primary_indices[idx]]
        else:
            aux_idx = self.auxiliary_indices[idx - self.num_primary]
            return self.auxiliary_dataset[aux_idx]


class CrossDatasetTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: CrossDatasetConfig,
        device: torch.device,
        save_dir: Path,
        logger: logging.Logger | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.save_dir = save_dir
        self.logger = logger

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        self.best_train_acc = 0.0
        self.best_val_acc = 0.0
        self.final_train_acc = 0.0
        self.final_val_acc = 0.0
        self.final_train_loss = 0.0
        self.final_val_loss = 0.0
        self.saved_model_path: str | None = None

    def _create_optimizer(self) -> optim.Optimizer:
        return optim.SGD(
            self.model.parameters(),
            lr=self.config.training.initial_lr,
            momentum=self.config.training.momentum,
            weight_decay=self.config.training.weight_decay,
        )

    def _create_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        def warmup_schedule(epoch: int) -> float:
            if epoch < self.config.training.warmup_epochs:
                return (epoch + 1) / self.config.training.warmup_epochs
            return 0.1 ** ((epoch - self.config.training.warmup_epochs) // 10)

        return optim.lr_scheduler.LambdaLR(self.optimizer, warmup_schedule)

    def _train_epoch(self) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        return total_loss / len(self.train_loader), 100.0 * correct / total

    def _validate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        return total_loss / len(self.val_loader), 100.0 * correct / total

    def _save_model(self, acc: float, epoch: int, lr: float) -> None:
        timestamp = get_current_jst_timestamp()

        primary = self.config.primary_dataset
        auxiliary = self.config.auxiliary_dataset
        aux_pct = int(self.config.auxiliary_percent)

        filename = (f"cross_{primary}_{auxiliary}_aux{aux_pct}pct_" f"e{epoch}_acc{acc:.2f}_{timestamp}.pth").replace(
            ".", "_", 1
        )

        state = {
            "model": self.model.state_dict(),
            "acc": acc,
            "epoch": epoch,
            "lr": lr,
            "config": asdict(self.config),
            "timestamp": timestamp,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

        self.save_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.save_dir / filename
        torch.save(state, save_path)
        self.saved_model_path = str(save_path)

        if self.logger:
            self.logger.info(f"Model saved to {save_path}")

    def train(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.training.epochs):
            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.logger:
                self.logger.info(f"\nEpoch {epoch + 1}/{self.config.training.epochs} (lr={current_lr:.6f})")

            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._validate()

            if self.logger:
                self.logger.info(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%")
                self.logger.info(f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%")

            self.scheduler.step()

            if train_acc > self.best_train_acc:
                self.best_train_acc = train_acc
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc

            if epoch == self.config.training.epochs - 1:
                self.final_train_acc = train_acc
                self.final_val_acc = val_acc
                self.final_train_loss = train_loss
                self.final_val_loss = val_loss
                self._save_model(val_acc, epoch + 1, current_lr)


class Kinetics700Evaluator:
    def __init__(
        self,
        model_path: str,
        primary_dataset: str,
        auxiliary_dataset: str,
        metavd_path: Path,
        logger: logging.Logger,
    ):
        self.model_path = model_path
        self.primary_dataset = primary_dataset
        self.auxiliary_dataset = auxiliary_dataset
        self.metavd_path = metavd_path
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self) -> tuple[float, float, dict[str, float]]:
        self.logger.info("\n=== Kinetics700 Evaluation ===")

        extractor = CommonLabelExtractor(self.metavd_path, logger=self.logger)
        triple_common = extractor.extract_triple_common_labels(
            self.primary_dataset, self.auxiliary_dataset, "kinetics700"
        )

        if len(triple_common) == 0:
            self.logger.warning("No common labels found for Kinetics700 evaluation")
            return 0.0, float("inf"), {}

        training_common = extractor.extract_common_labels(self.primary_dataset, self.auxiliary_dataset)
        training_labels_sorted = sorted(training_common.keys())
        training_label_to_idx = {label: idx for idx, label in enumerate(training_labels_sorted)}

        kinetics_labels = set(mappings["kinetics700"] for mappings in triple_common.values())

        self.logger.info(f"Kinetics700 labels to evaluate: {sorted(kinetics_labels)}")

        kinetics_dataset = Kinetics700Dataset(
            split="val",
            sampling_frames=16,
            label_mapper=None,
            is_target=True,
            logger=self.logger,
        )

        kinetics_filtered = FilteredDataset(kinetics_dataset, allowed_labels=kinetics_labels, logger=self.logger)

        if len(kinetics_filtered) == 0:
            self.logger.warning("No samples found in Kinetics700 after filtering")
            return 0.0, float("inf"), {}

        kinetics_to_training = {}
        for training_label, mappings in triple_common.items():
            kinetics_label = mappings["kinetics700"]
            kinetics_to_training[kinetics_label] = training_label

        class RemappedDataset(Dataset):
            def __init__(self, filtered_dataset, kinetics_to_training, training_label_to_idx):
                self.filtered_dataset = filtered_dataset
                self.kinetics_to_training = kinetics_to_training
                self.training_label_to_idx = training_label_to_idx
                self.idx_to_kinetics_label = {idx: label for label, idx in filtered_dataset.label_to_idx.items()}

            def __len__(self):
                return len(self.filtered_dataset)

            def __getitem__(self, idx):
                video, kinetics_idx = self.filtered_dataset[idx]
                kinetics_label = self.idx_to_kinetics_label[kinetics_idx]
                training_label = self.kinetics_to_training[kinetics_label]
                training_idx = self.training_label_to_idx[training_label]
                return video, training_idx

        remapped_dataset = RemappedDataset(kinetics_filtered, kinetics_to_training, training_label_to_idx)

        test_loader = DataLoader(
            remapped_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        num_classes = checkpoint["model"]["fc.weight"].shape[0]

        model_creator = R2Plus1DModel()
        model = model_creator.create_model(num_classes)
        model.load_state_dict(checkpoint["model"])
        model = model.to(self.device)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        class_correct: dict[int, int] = {}
        class_total: dict[int, int] = {}

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

                for target, pred in zip(targets.cpu().numpy(), predicted.cpu().numpy(), strict=False):
                    if target not in class_total:
                        class_total[target] = 0
                        class_correct[target] = 0
                    class_total[target] += 1
                    if target == pred:
                        class_correct[target] += 1

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else float("inf")

        training_idx_to_label = {idx: label for label, idx in training_label_to_idx.items()}
        class_accuracies = {}
        for class_idx in class_total:
            label = training_idx_to_label.get(class_idx, f"class_{class_idx}")
            class_accuracies[label] = 100.0 * class_correct[class_idx] / class_total[class_idx]

        self.logger.info("\nKinetics700 Results:")
        self.logger.info(f"  Accuracy: {accuracy:.2f}%")
        self.logger.info(f"  Loss: {avg_loss:.3f}")
        self.logger.info(f"  Samples: {total}")
        self.logger.info("  Class accuracies:")
        for label, acc in sorted(class_accuracies.items()):
            self.logger.info(f"    {label}: {acc:.2f}%")

        return accuracy, avg_loss, class_accuracies


def get_dataset_class(name: str) -> type[ActionRecognitionDataset]:
    dataset_mapping = {
        "activitynet": ActivityNetDataset,
        "charades": CharadesDataset,
        "kinetics700": Kinetics700Dataset,
        "hmdb51": HMDB51Dataset,
        "stair_actions": STAIRActionsDataset,
        "ucf101": UCF101Dataset,
    }
    if name not in dataset_mapping:
        msg = f"Unsupported dataset: {name}"
        raise ValueError(msg)
    return dataset_mapping[name]


def get_dataset_sizes(
    dataset_a: str,
    dataset_b: str,
    metavd_path: Path,
    logger: logging.Logger | None = None,
) -> tuple[int, int, dict[str, str]]:
    extractor = CommonLabelExtractor(metavd_path, logger=None)
    common_labels = extractor.extract_common_labels(dataset_a, dataset_b)

    if not common_labels:
        raise ValueError(f"No common labels found between {dataset_a} and {dataset_b}")

    label_mapper = ActionLabelMapper(metavd_path)
    label_mapper.load_mapping(dataset_a)

    ds_a_class = get_dataset_class(dataset_a)
    ds_a_full = ds_a_class(
        split="train",
        sampling_frames=16,
        label_mapper=label_mapper,
        is_target=True,
        logger=None,
    )
    ds_a_filtered = FilteredDataset(ds_a_full, set(common_labels.keys()), logger=None)
    size_a = len(ds_a_filtered)

    ds_b_class = get_dataset_class(dataset_b)
    ds_b_full = ds_b_class(
        split="train",
        sampling_frames=16,
        label_mapper=label_mapper,
        is_target=False,
        logger=None,
    )
    ds_b_filtered = FilteredDataset(ds_b_full, set(common_labels.keys()), logger=None)
    size_b = len(ds_b_filtered)

    if logger:
        logger.info("\n=== Dataset Sizes ===")
        logger.info(f"{dataset_a}: {size_a} samples")
        logger.info(f"{dataset_b}: {size_b} samples")
        logger.info(f"Common labels: {len(common_labels)}")

    return size_a, size_b, common_labels


def determine_primary_auxiliary(
    dataset_a: str,
    dataset_b: str,
    size_a: int,
    size_b: int,
) -> tuple[str, str, int, int]:
    if size_a <= size_b:
        return dataset_a, dataset_b, size_a, size_b
    else:
        return dataset_b, dataset_a, size_b, size_a


def create_experiment(
    config: CrossDatasetConfig,
    metavd_path: Path,
    logger: logging.Logger,
) -> tuple[CrossDatasetTrainer, int, int, float, int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"\n{'=' * 60}")
    logger.info("Creating experiment")
    logger.info(f"Primary: {config.primary_dataset}")
    logger.info(f"Auxiliary: {config.auxiliary_dataset} ({config.auxiliary_percent:.0f}%)")
    logger.info(f"{'=' * 60}")

    extractor = CommonLabelExtractor(metavd_path, logger=None)
    common_labels = extractor.extract_common_labels(config.primary_dataset, config.auxiliary_dataset)

    primary_labels = set(common_labels.keys())

    label_mapper = ActionLabelMapper(metavd_path)
    label_mapper.load_mapping(config.primary_dataset)

    primary_class = get_dataset_class(config.primary_dataset)
    primary_train_full = primary_class(
        split="train",
        sampling_frames=config.data.sampling_frames,
        label_mapper=label_mapper,
        is_target=True,
        logger=logger,
    )
    primary_train = FilteredDataset(primary_train_full, primary_labels, logger)

    primary_val_full = primary_class(
        split="test",
        sampling_frames=config.data.sampling_frames,
        label_mapper=label_mapper,
        is_target=True,
        logger=logger,
    )
    primary_val = FilteredDataset(primary_val_full, primary_labels, logger)

    auxiliary_train = None
    if config.auxiliary_percent > 0:
        auxiliary_class = get_dataset_class(config.auxiliary_dataset)
        auxiliary_train_full = auxiliary_class(
            split="train",
            sampling_frames=config.data.sampling_frames,
            label_mapper=label_mapper,
            is_target=False,
            logger=logger,
        )
        auxiliary_train = FilteredDataset(auxiliary_train_full, primary_labels, logger)

    train_dataset = CrossDatasetTrainingSet(
        primary_dataset=primary_train,
        auxiliary_dataset=auxiliary_train,
        auxiliary_percent=config.auxiliary_percent,
        logger=logger,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    val_loader = DataLoader(
        primary_val,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    num_classes = len(primary_train.label_to_idx)
    logger.info(f"\nNumber of classes: {num_classes}")

    model_creator = R2Plus1DModel()
    model = model_creator.create_model(num_classes)
    model = model.to(device)

    trainer = CrossDatasetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        save_dir=TRAINED_MODELS_DIR / f"cross_{config.primary_dataset}_{config.auxiliary_dataset}",
        logger=logger,
    )

    return (
        trainer,
        train_dataset.num_primary,
        train_dataset.num_auxiliary,
        train_dataset.actual_percent,
        num_classes,
    )


def run_percentage_experiments(
    dataset_a: str,
    dataset_b: str,
    metavd_path: Path,
    logger: logging.Logger,
) -> list[ExperimentResult]:
    size_a, size_b, common_labels = get_dataset_sizes(dataset_a, dataset_b, metavd_path, logger)

    primary, auxiliary, primary_size, auxiliary_size = determine_primary_auxiliary(dataset_a, dataset_b, size_a, size_b)

    max_percent = 100 * auxiliary_size / primary_size

    logger.info(f"\n{'=' * 80}")
    logger.info("EXPERIMENT PLAN")
    logger.info(f"{'=' * 80}")
    logger.info(f"Primary: {primary} ({primary_size} samples)")
    logger.info(f"Auxiliary: {auxiliary} ({auxiliary_size} samples)")
    logger.info(f"Max auxiliary: {max_percent:.1f}% of Primary")
    logger.info(f"Common labels: {len(common_labels)}")

    percentages = [0.0, 25.0, 50.0, 75.0, 100.0]

    if max_percent < 100:
        percentages = [p for p in percentages if p <= max_percent]
        if max_percent not in percentages:
            percentages.append(max_percent)
        percentages = sorted(percentages)

    logger.info(f"Percentages to test: {percentages}")

    results = []
    timestamp = get_current_jst_timestamp()

    for percent in percentages:
        logger.info(f"\n{'#' * 80}")
        logger.info(f"EXPERIMENT: Auxiliary = {percent:.0f}% of Primary")
        logger.info(f"{'#' * 80}")

        config = CrossDatasetConfig(
            primary_dataset=primary,
            auxiliary_dataset=auxiliary,
            auxiliary_percent=percent,
        )

        trainer, n_primary, n_auxiliary, actual_pct, num_classes = create_experiment(config, metavd_path, logger)
        trainer.train()

        evaluator = Kinetics700Evaluator(
            model_path=trainer.saved_model_path,
            primary_dataset=primary,
            auxiliary_dataset=auxiliary,
            metavd_path=metavd_path,
            logger=logger,
        )
        kinetics_acc, kinetics_loss, class_accs = evaluator.evaluate()

        result = ExperimentResult(
            primary_dataset=primary,
            auxiliary_dataset=auxiliary,
            auxiliary_percent=percent,
            actual_percent=actual_pct,
            num_primary=n_primary,
            num_auxiliary=n_auxiliary,
            num_total=n_primary + n_auxiliary,
            num_classes=num_classes,
            train_acc=trainer.final_train_acc,
            val_acc=trainer.final_val_acc,
            kinetics_acc=kinetics_acc,
            kinetics_loss=kinetics_loss,
            model_path=trainer.saved_model_path,
            timestamp=timestamp,
        )
        results.append(result)

        logger.info(f"\n>>> Result: Aux {percent:.0f}%")
        logger.info(f"    Train Acc: {result.train_acc:.2f}%")
        logger.info(f"    Val Acc: {result.val_acc:.2f}%")
        logger.info(f"    K700 Acc: {result.kinetics_acc:.2f}%")

    return results


def print_results_table(results: list[ExperimentResult], logger: logging.Logger) -> None:
    logger.info(f"\n{'=' * 80}")
    logger.info("FINAL RESULTS SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(f"Primary: {results[0].primary_dataset}")
    logger.info(f"Auxiliary: {results[0].auxiliary_dataset}")
    logger.info("Evaluation: Kinetics700")

    header = f"{'Aux %':<8} {'Primary':<10} {'Auxiliary':<10} {'Total':<10} " f"{'Train':<10} {'Val':<10} {'K700':<10}"
    logger.info(f"\n{header}")
    logger.info("-" * 78)

    for r in results:
        logger.info(
            f"{r.auxiliary_percent:<8.0f} "
            f"{r.num_primary:<10} "
            f"{r.num_auxiliary:<10} "
            f"{r.num_total:<10} "
            f"{r.train_acc:<10.2f} "
            f"{r.val_acc:<10.2f} "
            f"{r.kinetics_acc:<10.2f}"
        )

    best = max(results, key=lambda x: x.kinetics_acc)
    baseline = next((r for r in results if r.auxiliary_percent == 0), results[0])
    improvement = best.kinetics_acc - baseline.kinetics_acc

    logger.info(f"\nBaseline (0%): K700 Acc = {baseline.kinetics_acc:.2f}%")
    logger.info(f"Best ({best.auxiliary_percent:.0f}%): K700 Acc = {best.kinetics_acc:.2f}%")
    logger.info(f"Improvement: {improvement:+.2f}%")


def save_results(
    results: list[ExperimentResult],
    save_dir: Path,
    logger: logging.Logger,
) -> tuple[Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = results[0].timestamp if results else get_current_jst_timestamp()

    primary = results[0].primary_dataset

    json_path = save_dir / f"cross_{primary}_{auxiliary}_{timestamp}.json"
    json_data = [asdict(r) for r in results]
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    csv_path = save_dir / f"cross_{primary}_{auxiliary}_{timestamp}.csv"
    if results:
        fieldnames = list(asdict(results[0]).keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))

    logger.info("\nResults saved:")
    logger.info(f"  JSON: {json_path}")
    logger.info(f"  CSV: {csv_path}")

    return json_path, csv_path


def main():
    dataset_a = "ucf101"
    dataset_b = "stair_actions"

    timestamp = get_current_jst_timestamp()
    log_dir = LOGS_DIR / "cross_dataset"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"cross_{dataset_a}_{dataset_b}_{timestamp}.log"
    logger = setup_logger("cross_dataset", log_file)

    logger.info("=" * 80)
    logger.info("CROSS-DATASET TRAINING EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Dataset A: {dataset_a}")
    logger.info(f"Dataset B: {dataset_b}")
    logger.info("Evaluation: Kinetics700")

    results = run_percentage_experiments(
        dataset_a=dataset_a,
        dataset_b=dataset_b,
        metavd_path=AUTO_METAVD_DIR / "auto_metavd_mpnet_0.84.csv",
        logger=logger,
    )

    print_results_table(results, logger)

    save_dir = TRAINED_MODELS_DIR / "results" / "cross_dataset"
    save_results(results, save_dir, logger)

    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
