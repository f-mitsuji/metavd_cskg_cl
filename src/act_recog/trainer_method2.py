"""方法Ⅱ：訓練データ削減とデータ拡張の評価

一つのデータセットを選択し、訓練データの事例数を1/nに減らし、
MetaVDを用いてデータを拡張して評価する。

削減率の選択理由：
- n=10 (10%): 実用的なシナリオ
  - ラベル付きデータが限られている状況を模擬
  - データ拡張の効果が最も顕著に現れる
  - 精度低下が大きいため、改善の余地が十分にある
"""

import csv
import json
import logging
import random
from collections.abc import Sequence
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


class CommonLabelExtractor:
    def __init__(self, mapping_file: Path, logger: logging.Logger | None = None):
        self.mapping_file = mapping_file
        self.logger = logger

    def extract_common_labels(self, dataset_a: str, dataset_b: str) -> dict[str, str]:
        mapping_df = pd.read_csv(self.mapping_file)

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


class FilteredDataset(Dataset):
    def __init__(
        self, base_dataset: ActionRecognitionDataset, allowed_labels: set[str], logger: logging.Logger | None = None
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

            label_counts = {}
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


@dataclass(frozen=True)
class Method2ExperimentConfig:
    target_dataset: str
    source_datasets: list[str]
    reduction_rate: float = 0.1
    use_augmentation: bool = True
    random_seed: int = 42
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)


@dataclass
class ExperimentResult:
    experiment_id: int
    target_dataset: str
    source_datasets: list[str]
    reduction_rate: float
    use_augmentation: bool
    num_common_classes: int
    num_target_classes: int
    num_train_samples_full: int
    num_train_samples_reduced: int
    num_train_samples_source: int
    num_train_samples_total: int
    num_test_samples: int
    best_train_acc: float
    best_val_acc: float
    final_train_acc: float
    final_val_acc: float
    final_train_loss: float
    final_val_loss: float
    epochs: int
    timestamp: str
    model_path: str | None = None


class ResultLogger:
    def __init__(self, save_dir: Path, logger: logging.Logger | None = None):
        self.save_dir = save_dir
        self.logger = logger
        self.results: list[ExperimentResult] = []
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def add_result(self, result: ExperimentResult) -> None:
        self.results.append(result)
        if self.logger:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"EXPERIMENT {result.experiment_id} RESULTS")
            self.logger.info(f"{'='*80}")
            self.logger.info(f"Target: {result.target_dataset}")
            self.logger.info(f"Augmentation: {'Yes' if result.use_augmentation else 'No'}")
            if result.use_augmentation:
                self.logger.info(f"Source datasets: {', '.join(result.source_datasets)}")
            self.logger.info(f"Reduction rate: {result.reduction_rate*100:.0f}%")
            self.logger.info(f"Number of classes: {result.num_common_classes}")
            self.logger.info(f"Training samples: {result.num_train_samples_total}")
            self.logger.info(f"  - Target (reduced): {result.num_train_samples_reduced}")
            self.logger.info(f"  - Source: {result.num_train_samples_source}")
            self.logger.info(f"Test samples: {result.num_test_samples}")
            self.logger.info(f"Final Train Acc: {result.final_train_acc:.2f}%")
            self.logger.info(f"Final Val Acc: {result.final_val_acc:.2f}%")
            self.logger.info(f"Best Val Acc: {result.best_val_acc:.2f}%")

    def save_results(self, timestamp: str) -> tuple[Path, Path, Path]:
        json_path = self.save_dir / f"method2_results_{timestamp}.json"
        json_data = [asdict(result) for result in self.results]
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        csv_path = self.save_dir / f"method2_results_{timestamp}.csv"
        if self.results:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
                writer.writeheader()
                for result in self.results:
                    row = asdict(result)
                    row["source_datasets"] = ",".join(row["source_datasets"])
                    writer.writerow(row)

        md_path = self.save_dir / f"method2_report_{timestamp}.md"
        self._generate_markdown_report(md_path)

        if self.logger:
            self.logger.info("\nResults saved to:")
            self.logger.info(f"  JSON: {json_path}")
            self.logger.info(f"  CSV: {csv_path}")
            self.logger.info(f"  Markdown: {md_path}")

        return json_path, csv_path, md_path

    def _generate_markdown_report(self, output_path: Path) -> None:
        with output_path.open("w", encoding="utf-8") as f:
            f.write("# Method 2: Data Reduction with Augmentation - Experimental Results\n\n")

            if self.results:
                result = self.results[0]
                f.write("## Experimental Configuration\n\n")
                f.write(f"- **Target Dataset**: {result.target_dataset}\n")
                f.write(f"- **Reduction Rate**: {result.reduction_rate*100:.0f}%\n")
                f.write(f"- **Training Epochs**: {result.epochs}\n\n")

            f.write("## Results Comparison\n\n")
            f.write(
                "| Experiment | Augmentation | Source Datasets | Common Classes | Training Samples | Test Acc (%) |\n"
            )
            f.write(
                "|------------|--------------|-----------------|----------------|------------------|---------------|\n"
            )

            for result in self.results:
                aug = "Yes" if result.use_augmentation else "No"
                sources = ", ".join(result.source_datasets) if result.source_datasets else "None"
                f.write(
                    f"| {result.experiment_id} | {aug} | {sources} | {result.num_common_classes} | "
                    f"{result.num_train_samples_total} | {result.final_val_acc:.2f} |\n"
                )

            f.write("\n## Detailed Results\n\n")
            for result in self.results:
                f.write(f"### Experiment {result.experiment_id}\n\n")
                f.write("- **Configuration**:\n")
                f.write(f"  - Target: {result.target_dataset}\n")
                f.write(f"  - Augmentation: {'Yes' if result.use_augmentation else 'No'}\n")
                if result.use_augmentation:
                    f.write(f"  - Source Datasets: {', '.join(result.source_datasets)}\n")
                f.write("\n- **Data Statistics**:\n")
                f.write(f"  - Common Classes: {result.num_common_classes}\n")
                f.write(f"  - Training Samples (Total): {result.num_train_samples_total}\n")
                f.write(f"    - Target (Full, Filtered): {result.num_train_samples_full}\n")
                f.write(f"    - Target (Reduced): {result.num_train_samples_reduced}\n")
                f.write(f"    - Source: {result.num_train_samples_source}\n")
                f.write(f"  - Test Samples: {result.num_test_samples}\n")
                f.write("\n- **Performance**:\n")
                f.write(f"  - Final Train Accuracy: {result.final_train_acc:.2f}%\n")
                f.write(f"  - Final Val Accuracy: {result.final_val_acc:.2f}%\n")
                f.write(f"  - Best Val Accuracy: {result.best_val_acc:.2f}%\n")
                f.write(f"  - Final Train Loss: {result.final_train_loss:.4f}\n")
                f.write(f"  - Final Val Loss: {result.final_val_loss:.4f}\n")
                if result.model_path:
                    f.write(f"  - Model Path: {result.model_path}\n")
                f.write("\n")

            f.write("## Analysis\n\n")
            if len(self.results) >= 2:
                baseline = self.results[0]
                augmented = self.results[1]
                improvement = augmented.final_val_acc - baseline.final_val_acc
                f.write("### Data Augmentation Effect\n\n")
                f.write(f"- **Baseline** (Target only, {baseline.reduction_rate*100:.0f}%): ")
                f.write(f"{baseline.final_val_acc:.2f}%\n")
                f.write(f"- **With Augmentation** (Target + Source): {augmented.final_val_acc:.2f}%\n")
                f.write(f"- **Improvement**: {improvement:+.2f}%\n\n")

                if improvement > 0:
                    f.write(
                        "Data augmentation with cross-dataset transfer shows positive effect, "
                        "helping to compensate for the reduced training data.\n"
                    )
                else:
                    f.write(
                        "Data augmentation did not improve performance in this experiment. "
                        "This might be due to domain mismatch or insufficient alignment "
                        "between source and target datasets.\n"
                    )


class ReducedDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        reduction_rate: float,
        random_seed: int = 42,
        logger: logging.Logger | None = None,
    ):
        self.base_dataset = base_dataset
        self.reduction_rate = reduction_rate
        self.logger = logger

        random.seed(random_seed)

        original_size = len(base_dataset)
        reduced_size = max(1, int(original_size * reduction_rate))

        all_indices = list(range(original_size))
        random.shuffle(all_indices)
        self.indices = all_indices[:reduced_size]

        if self.logger:
            self.logger.info(f"Original dataset size: {original_size}")
            self.logger.info(f"Reduced dataset size: {len(self.indices)} ({reduction_rate*100:.0f}%)")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        original_idx = self.indices[idx]
        return self.base_dataset[original_idx]


class UnifiedActionDataset(Dataset):
    def __init__(self, target_dataset: Dataset, source_datasets: Sequence[Dataset]):
        self.target_dataset = target_dataset
        self.source_datasets = source_datasets

        self.target_size = len(target_dataset)
        self.source_sizes = [len(ds) for ds in source_datasets]
        self.total_size = self.target_size + sum(self.source_sizes)

        self.cumulative_sizes = [self.target_size]
        for size in self.source_sizes:
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int):
        if idx < self.target_size:
            return self.target_dataset[idx]

        idx -= self.target_size
        for i, size in enumerate(self.source_sizes):
            if idx < size:
                return self.source_datasets[i][idx]
            idx -= size

        msg = f"Index {idx} out of range"
        raise IndexError(msg)


class ActionRecognitionTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Method2ExperimentConfig,
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
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config.training.initial_lr,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay,
        )

        self.warmup_epochs = config.training.warmup_epochs
        self.total_epochs = config.training.epochs

        self.train_losses: list[float] = []
        self.train_accs: list[float] = []
        self.val_losses: list[float] = []
        self.val_accs: list[float] = []
        self.best_val_acc = 0.0
        self.best_model_path: Path | None = None

    def _adjust_learning_rate(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            lr = self.config.training.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.config.training.initial_lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            lr = float(lr)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for videos, labels in self.train_loader:
            videos = videos.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(videos)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for videos, labels in self.val_loader:
                videos = videos.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(self) -> None:
        if self.logger:
            self.logger.info("\n=== Starting Training ===")

        for epoch in range(self.config.training.epochs):
            self._adjust_learning_rate(epoch)
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_path = self.save_dir / f"best_model_epoch{epoch+1}.pth"
                torch.save(self.model.state_dict(), self.best_model_path)

            if self.logger and (epoch + 1) % 5 == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(
                    f"Epoch [{epoch+1}/{self.config.training.epochs}] "
                    f"LR: {current_lr:.6f} | "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                )

        if self.logger:
            self.logger.info("\n=== Training Completed ===")
            self.logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")

    def get_results(
        self,
        experiment_id: int,
        num_common_classes: int,
        num_target_classes: int,
        num_train_samples_full: int,
        num_train_samples_reduced: int,
        num_train_samples_source: int,
        num_test_samples: int,
        timestamp: str,
    ) -> ExperimentResult:
        return ExperimentResult(
            experiment_id=experiment_id,
            target_dataset=self.config.target_dataset,
            source_datasets=self.config.source_datasets,
            reduction_rate=self.config.reduction_rate,
            use_augmentation=self.config.use_augmentation,
            num_common_classes=num_common_classes,
            num_target_classes=num_target_classes,
            num_train_samples_full=num_train_samples_full,
            num_train_samples_reduced=num_train_samples_reduced,
            num_train_samples_source=num_train_samples_source,
            num_train_samples_total=num_train_samples_reduced + num_train_samples_source,
            num_test_samples=num_test_samples,
            best_train_acc=max(self.train_accs),
            best_val_acc=self.best_val_acc,
            final_train_acc=self.train_accs[-1],
            final_val_acc=self.val_accs[-1],
            final_train_loss=self.train_losses[-1],
            final_val_loss=self.val_losses[-1],
            epochs=self.config.training.epochs,
            timestamp=timestamp,
            model_path=str(self.best_model_path) if self.best_model_path else None,
        )


def get_dataset_class(dataset_name: str):
    dataset_map = {
        "hmdb51": HMDB51Dataset,
        "ucf101": UCF101Dataset,
        "kinetics700": Kinetics700Dataset,
        "activitynet": ActivityNetDataset,
        "charades": CharadesDataset,
        "stair_actions": STAIRActionsDataset,
    }
    if dataset_name not in dataset_map:
        msg = f"Unknown dataset: {dataset_name}"
        raise ValueError(msg)
    return dataset_map[dataset_name]


def create_method2_experiment(
    config: Method2ExperimentConfig, model_class, metavd_path: Path, logger: logging.Logger | None = None
) -> tuple[ActionRecognitionTrainer, int, int, int, int, int, int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if logger:
        logger.info(f"Using device: {device}")
        logger.info("\n=== Method 2 Experiment Configuration ===")
        logger.info(f"Target dataset: {config.target_dataset}")
        logger.info(f"Source datasets: {config.source_datasets}")
        logger.info(f"Reduction rate: {config.reduction_rate*100:.0f}%")
        logger.info(f"Use augmentation: {config.use_augmentation}")
        logger.info(f"Random seed: {config.random_seed}")

    extractor = CommonLabelExtractor(metavd_path, logger=logger)
    all_common_labels = set()

    if config.use_augmentation and config.source_datasets:
        for source_name in config.source_datasets:
            common_labels_dict = extractor.extract_common_labels(config.target_dataset, source_name)
            all_common_labels.update(common_labels_dict.keys())

        if logger:
            logger.info("\n=== All Common Labels ===")
            logger.info(f"Total unique common labels: {len(all_common_labels)}")
            logger.info(f"Labels: {sorted(all_common_labels)}")

    target_label_mapper = ActionLabelMapper(metavd_path)
    target_label_mapper.load_mapping(config.target_dataset)

    if logger:
        logger.info(f"\n=== Loading Target Dataset: {config.target_dataset} ===")

    target_dataset_class = get_dataset_class(config.target_dataset)

    target_train_full = target_dataset_class(
        split="train",
        sampling_frames=config.data.sampling_frames,
        label_mapper=target_label_mapper,
        is_target=True,
        logger=logger,
    )

    target_val_full = target_dataset_class(
        split="test",
        sampling_frames=config.data.sampling_frames,
        label_mapper=target_label_mapper,
        is_target=True,
        logger=logger,
    )

    if config.use_augmentation and all_common_labels:
        if logger:
            logger.info("\n=== Filtering Target Dataset to Common Classes ===")

        target_train_filtered = FilteredDataset(target_train_full, allowed_labels=all_common_labels, logger=logger)

        target_val_filtered = FilteredDataset(target_val_full, allowed_labels=all_common_labels, logger=logger)

        num_common_classes = len(target_train_filtered.label_to_idx)
        num_train_full_filtered = len(target_train_filtered)
        num_test = len(target_val_filtered)
    else:
        target_train_filtered = target_train_full
        target_val_filtered = target_val_full
        num_common_classes = len(target_train_full.label_to_idx)
        num_train_full_filtered = len(target_train_full)
        num_test = len(target_val_full)

    if logger:
        logger.info("\n=== Reducing Training Data ===")

    target_train_reduced = ReducedDataset(
        target_train_filtered, reduction_rate=config.reduction_rate, random_seed=config.random_seed, logger=logger
    )

    num_train_reduced = len(target_train_reduced)
    num_train_source = 0

    source_train_datasets = []
    if config.use_augmentation and config.source_datasets:
        for source_name in config.source_datasets:
            if logger:
                logger.info(f"\n=== Loading Source Dataset: {source_name} ===")

            source_label_mapper = ActionLabelMapper(metavd_path)
            source_label_mapper.load_mapping(config.target_dataset)

            source_dataset_class = get_dataset_class(source_name)
            source_train_full = source_dataset_class(
                split="train",
                sampling_frames=config.data.sampling_frames,
                label_mapper=source_label_mapper,
                is_target=False,
                logger=logger,
            )

            common_labels_dict = extractor.extract_common_labels(config.target_dataset, source_name)

            target_common_labels = set(common_labels_dict.keys())

            if logger:
                logger.info(f"\n=== Filtering Source Dataset: {source_name} to Common Classes ===")
                logger.info(f"Common labels (target): {sorted(target_common_labels)}")

            source_train_filtered = FilteredDataset(
                source_train_full, allowed_labels=target_common_labels, logger=logger
            )

            source_train_datasets.append(source_train_filtered)
            num_train_source += len(source_train_filtered)

    unified_train_dataset = UnifiedActionDataset(target_train_reduced, source_train_datasets)

    if logger:
        logger.info("\n=== Final Training Dataset Statistics ===")
        logger.info(f"Number of common classes: {num_common_classes}")
        logger.info(f"Total training samples: {len(unified_train_dataset)}")
        logger.info(f"  - Target (full, filtered): {num_train_full_filtered}")
        logger.info(f"  - Target (reduced): {num_train_reduced}")
        logger.info(f"  - Source: {num_train_source}")
        logger.info("\n=== Test Data: Target Dataset Only ===")
        logger.info(f"Test on: {config.target_dataset} (common classes only)")
        logger.info(f"Test samples: {num_test}")

    train_loader = DataLoader(
        unified_train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    val_loader = DataLoader(
        target_val_filtered,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    if logger:
        logger.info("\n=== Model Information ===")
        logger.info(f"Number of classes: {num_common_classes}")

    model_creator = model_class()
    model = model_creator.create_model(num_common_classes)
    model = model.to(device)

    trainer = ActionRecognitionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        save_dir=TRAINED_MODELS_DIR,
        logger=logger,
    )

    return (
        trainer,
        num_common_classes,
        num_common_classes,
        num_train_full_filtered,
        num_train_reduced,
        num_train_source,
        num_test,
    )


def run_method2_experiments(
    target_dataset: str, source_datasets: list[str], reduction_rate: float, metavd_path: Path, logger: logging.Logger
) -> ResultLogger:
    result_logger = ResultLogger(TRAINED_MODELS_DIR / "results", logger)
    timestamp = get_current_jst_timestamp()

    experiments = [
        Method2ExperimentConfig(
            target_dataset=target_dataset,
            source_datasets=[],
            reduction_rate=reduction_rate,
            use_augmentation=False,
        ),
        Method2ExperimentConfig(
            target_dataset=target_dataset,
            source_datasets=source_datasets,
            reduction_rate=reduction_rate,
            use_augmentation=True,
        ),
    ]

    for i, config in enumerate(experiments, 1):
        logger.info("\n" + "=" * 80)
        logger.info(f"EXPERIMENT {i}/2")
        logger.info("=" * 80)
        logger.info(f"Config: {asdict(config)}")

        trainer, num_common_classes, num_classes, num_full, num_reduced, num_source, num_test = (
            create_method2_experiment(config=config, model_class=R2Plus1DModel, metavd_path=metavd_path, logger=logger)
        )
        trainer.train()

        result = trainer.get_results(
            experiment_id=i,
            num_common_classes=num_common_classes,
            num_target_classes=num_classes,
            num_train_samples_full=num_full,
            num_train_samples_reduced=num_reduced,
            num_train_samples_source=num_source,
            num_test_samples=num_test,
            timestamp=timestamp,
        )
        result_logger.add_result(result)

    result_logger.save_results(timestamp)

    return result_logger


def main():
    timestamp = get_current_jst_timestamp()
    (LOGS_DIR / "method2").mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"method2_experiment_{timestamp}.log"
    logger = setup_logger("method2", log_file)
    logger.info("Starting Method 2 Experiments")

    target_dataset = "hmdb51"
    source_datasets = ["ucf101", "stair_actions"]
    reduction_rate = 0.1

    logger.info("\n=== Method 2: Data Reduction with Augmentation ===")
    logger.info(f"Target dataset: {target_dataset}")
    logger.info(f"Source datasets: {source_datasets}")
    logger.info(f"Reduction rate: {reduction_rate*100:.0f}%")
    logger.info("\n=== Evaluation Strategy ===")
    logger.info(f"Training: {target_dataset} ({reduction_rate*100:.0f}%) + {', '.join(source_datasets)}")
    logger.info(f"Testing: {target_dataset} (full test set ONLY)")
    logger.info(f"\nThis focuses evaluation on {target_dataset} to measure:")
    logger.info(f"  - Generalization from 10% to unseen 90% of {target_dataset}")
    logger.info("  - Effect of cross-dataset augmentation on target dataset")

    result_logger = run_method2_experiments(
        target_dataset=target_dataset,
        source_datasets=source_datasets,
        reduction_rate=reduction_rate,
        metavd_path=AUTO_METAVD_DIR / "auto_metavd_mpnet_0.84.csv",
        logger=logger,
    )

    logger.info("\n=== All Method 2 Experiments Completed ===")
    logger.info("Check the results directory for detailed analysis.")


if __name__ == "__main__":
    main()
