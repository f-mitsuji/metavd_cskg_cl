"""方法Ⅰ: 共通動作ラベルに絞った評価.

二つのデータセットを選択し、MetaVDにおいてequal関係にある動作ラベル（共通動作ラベル）を抽出した上で、
それらの動作ラベルおよび該当ラベルが付与された動画を対象に評価を行う。
"""

import csv
import json
import logging
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


@dataclass(frozen=True)
class Method1ExperimentConfig:
    dataset_a: str
    dataset_b: str
    target_dataset: str
    use_augmentation: bool = True
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)


@dataclass
class ExperimentResult:
    experiment_id: int
    dataset_a: str
    dataset_b: str
    target_dataset: str
    use_augmentation: bool
    num_common_labels: int
    num_train_samples_target: int
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
            self.logger.info(f"Common labels: {result.num_common_labels}")
            self.logger.info(
                f"Training samples: {result.num_train_samples_total} "
                f"(Target: {result.num_train_samples_target}, Source: {result.num_train_samples_source})"
            )
            self.logger.info(f"Test samples: {result.num_test_samples}")
            self.logger.info(f"Final Train Acc: {result.final_train_acc:.2f}%")
            self.logger.info(f"Final Val Acc: {result.final_val_acc:.2f}%")
            self.logger.info(f"Best Val Acc: {result.best_val_acc:.2f}%")

    def save_results(self, timestamp: str) -> tuple[Path, Path, Path]:
        json_path = self.save_dir / f"method1_results_{timestamp}.json"
        json_data = [asdict(result) for result in self.results]
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        csv_path = self.save_dir / f"method1_results_{timestamp}.csv"
        if self.results:
            fieldnames = list(asdict(self.results[0]).keys())
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in self.results:
                    writer.writerow(asdict(result))

        summary_path = self.save_dir / f"method1_summary_{timestamp}.txt"
        with summary_path.open("w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("METHOD 1 EXPERIMENT SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            if self.results:
                f.write(f"Dataset Pair: {self.results[0].dataset_a} <-> {self.results[0].dataset_b}\n")
                f.write(f"Number of common labels: {self.results[0].num_common_labels}\n")
                f.write(f"Total experiments: {len(self.results)}\n\n")

            for result in self.results:
                f.write("-" * 80 + "\n")
                f.write(f"Experiment {result.experiment_id}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Target Dataset: {result.target_dataset}\n")
                f.write(f"Data Augmentation: {'Yes' if result.use_augmentation else 'No'}\n")
                if result.use_augmentation:
                    source = result.dataset_b if result.target_dataset == result.dataset_a else result.dataset_a
                    f.write(f"Source Dataset: {source}\n")
                f.write("\nTraining Data:\n")
                f.write(f"  - Target samples: {result.num_train_samples_target}\n")
                f.write(f"  - Source samples: {result.num_train_samples_source}\n")
                f.write(f"  - Total samples: {result.num_train_samples_total}\n")
                f.write(f"  - Test samples: {result.num_test_samples}\n")
                f.write("\nResults:\n")
                f.write(f"  - Final Train Acc: {result.final_train_acc:.2f}%\n")
                f.write(f"  - Final Val Acc: {result.final_val_acc:.2f}%\n")
                f.write(f"  - Best Val Acc: {result.best_val_acc:.2f}%\n")
                f.write(f"  - Final Train Loss: {result.final_train_loss:.3f}\n")
                f.write(f"  - Final Val Loss: {result.final_val_loss:.3f}\n")
                f.write(f"\nModel saved at: {result.model_path}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("COMPARISON\n")
            f.write("=" * 80 + "\n\n")

            targets = {r.target_dataset for r in self.results}
            for target in sorted(targets):
                target_results = [r for r in self.results if r.target_dataset == target]
                if len(target_results) >= 2:
                    no_aug = next(r for r in target_results if not r.use_augmentation)
                    with_aug = next(r for r in target_results if r.use_augmentation)

                    f.write(f"Target: {target}\n")
                    f.write(f"  Baseline (no augmentation):    {no_aug.final_val_acc:.2f}%\n")
                    f.write(f"  With augmentation:             {with_aug.final_val_acc:.2f}%\n")
                    improvement = with_aug.final_val_acc - no_aug.final_val_acc
                    f.write(f"  Improvement:                   {improvement:+.2f}%\n")
                    f.write("\n")

        if self.logger:
            self.logger.info("\nResults saved:")
            self.logger.info(f"  - JSON: {json_path}")
            self.logger.info(f"  - CSV: {csv_path}")
            self.logger.info(f"  - Summary: {summary_path}")

        return json_path, csv_path, summary_path


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


class UnifiedActionDataset(Dataset):
    def __init__(self, target_dataset: FilteredDataset, source_datasets: Sequence[FilteredDataset]):
        self.target_dataset = target_dataset
        self.source_datasets = source_datasets

        self.dataset_sizes = [len(target_dataset)] + [len(ds) for ds in source_datasets]
        self.cumulative_sizes = [0]
        cum_sum = 0
        for size in self.dataset_sizes:
            cum_sum += size
            self.cumulative_sizes.append(cum_sum)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        dataset_idx = 0
        while idx >= self.cumulative_sizes[dataset_idx + 1]:
            dataset_idx += 1

        relative_idx = idx - self.cumulative_sizes[dataset_idx]

        if dataset_idx == 0:
            return self.target_dataset[relative_idx]
        return self.source_datasets[dataset_idx - 1][relative_idx]


class ActionRecognitionTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Method1ExperimentConfig,
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

    def _process_batch(self, inputs: torch.Tensor, targets: torch.Tensor, *, is_train: bool) -> tuple[float, int, int]:
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        if is_train:
            self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        if is_train:
            loss.backward()
            self.optimizer.step()

        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        total = targets.size(0)

        return loss.item(), correct, total

    def _process_epoch(self, *, is_train: bool) -> tuple[float, float]:
        total_loss = 0.0
        correct = 0
        total = 0
        loader = self.train_loader if is_train else self.val_loader

        for inputs, targets in loader:
            loss, batch_correct, batch_total = self._process_batch(inputs, targets, is_train=is_train)
            total_loss += loss
            correct += batch_correct
            total += batch_total

        return total_loss / len(loader), 100.0 * correct / total

    def _train_epoch(self) -> tuple[float, float]:
        self.model.train()
        return self._process_epoch(is_train=True)

    def _validate(self) -> tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            return self._process_epoch(is_train=False)

    def _create_experiment_name(self, name: str, epoch: int, lr: float, acc: float, timestamp: str) -> str:
        dataset_a = self.config.dataset_a
        dataset_b = self.config.dataset_b
        target = self.config.target_dataset
        aug = "aug" if self.config.use_augmentation else "noaug"

        params = [f"e{epoch}", f"lr{lr}".replace(".", "_"), f"acc{acc:.2f}".replace(".", "_")]
        params_str = "_".join(params)

        return f"method1_{dataset_a}_{dataset_b}_target{target}_{aug}_{params_str}_{timestamp}_{name}"

    def _save_model(self, name: str, acc: float, epoch: int, lr: float) -> None:
        if self.logger:
            self.logger.info(f"Saving {name} model..")

        timestamp = get_current_jst_timestamp()
        experiment_name = self._create_experiment_name(name=name, epoch=epoch, lr=lr, acc=acc, timestamp=timestamp)

        state = {
            "model": self.model.state_dict(),
            "acc": acc,
            "epoch": epoch,
            "lr": lr,
            "config": asdict(self.config),
            "timestamp": timestamp,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }

        save_path = self.save_dir / f"{experiment_name}.pth"
        torch.save(state, save_path)

        self.saved_model_path = str(save_path)

        if self.logger:
            self.logger.info(f"Model saved to {save_path}")

    def train(self) -> None:
        best_acc = 0.0
        self.save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.training.epochs):
            if self.logger:
                self.logger.info(f"\nEpoch {epoch+1}/{self.config.training.epochs}")

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.logger:
                self.logger.info(f"Current Learning Rate: {current_lr:.6f}")

            train_loss, train_acc = self._train_epoch()
            if self.logger:
                self.logger.info(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%")

            val_loss, val_acc = self._validate()
            if self.logger:
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
                self._save_model("final", val_acc, epoch + 1, current_lr)

            if val_acc > best_acc:
                best_acc = val_acc

    def get_results(
        self,
        experiment_id: int,
        num_common_labels: int,
        num_train_samples_target: int,
        num_train_samples_source: int,
        num_test_samples: int,
        timestamp: str,
    ) -> ExperimentResult:
        return ExperimentResult(
            experiment_id=experiment_id,
            dataset_a=self.config.dataset_a,
            dataset_b=self.config.dataset_b,
            target_dataset=self.config.target_dataset,
            use_augmentation=self.config.use_augmentation,
            num_common_labels=num_common_labels,
            num_train_samples_target=num_train_samples_target,
            num_train_samples_source=num_train_samples_source,
            num_train_samples_total=num_train_samples_target + num_train_samples_source,
            num_test_samples=num_test_samples,
            best_train_acc=self.best_train_acc,
            best_val_acc=self.best_val_acc,
            final_train_acc=self.final_train_acc,
            final_val_acc=self.final_val_acc,
            final_train_loss=self.final_train_loss,
            final_val_loss=self.final_val_loss,
            epochs=self.config.training.epochs,
            timestamp=timestamp,
            model_path=self.saved_model_path,
        )


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


def create_method1_experiment(
    config: Method1ExperimentConfig, model_class, metavd_path: Path, logger: logging.Logger | None = None
) -> tuple[ActionRecognitionTrainer, int, int, int, int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if logger:
        logger.info(f"Using device: {device}")
        logger.info("\n=== Method 1 Experiment Configuration ===")
        logger.info(f"Dataset A: {config.dataset_a}")
        logger.info(f"Dataset B: {config.dataset_b}")
        logger.info(f"Target (evaluation): {config.target_dataset}")
        logger.info(f"Use augmentation: {config.use_augmentation}")

    extractor = CommonLabelExtractor(metavd_path, logger)
    common_labels = extractor.extract_common_labels(config.dataset_a, config.dataset_b)

    if not common_labels:
        msg = f"No common labels found between {config.dataset_a} and {config.dataset_b}"
        raise ValueError(msg)

    num_common_labels = len(common_labels)

    if config.target_dataset == config.dataset_a:
        target_name = config.dataset_a
        source_name = config.dataset_b
        target_allowed_labels = set(common_labels.keys())
        source_allowed_labels = set(common_labels.keys())
    elif config.target_dataset == config.dataset_b:
        target_name = config.dataset_b
        source_name = config.dataset_a
        target_allowed_labels = set(common_labels.values())
        source_allowed_labels = set(common_labels.values())
    else:
        msg = f"target_dataset must be either {config.dataset_a} or {config.dataset_b}"
        raise ValueError(msg)

    if logger:
        logger.info(f"\n=== Loading Target Dataset: {target_name} ===")

    label_mapper = ActionLabelMapper(metavd_path)
    label_mapper.load_mapping(target_name)

    target_dataset_class = get_dataset_class(target_name)

    target_train_full = target_dataset_class(
        split="train",
        sampling_frames=config.data.sampling_frames,
        label_mapper=label_mapper,
        is_target=True,
        logger=logger,
    )

    target_val_full = target_dataset_class(
        split="test",
        sampling_frames=config.data.sampling_frames,
        label_mapper=label_mapper,
        is_target=True,
        logger=logger,
    )

    if logger:
        logger.info(f"\n=== Filtering Target Dataset: {target_name} ===")

    target_train_dataset = FilteredDataset(target_train_full, target_allowed_labels, logger)

    target_val_dataset = FilteredDataset(target_val_full, target_allowed_labels, logger)

    num_train_target = len(target_train_dataset)
    num_test = len(target_val_dataset)
    num_train_source = 0

    source_train_datasets = []
    if config.use_augmentation:
        if logger:
            logger.info(f"\n=== Loading Source Dataset: {source_name} ===")

        source_dataset_class = get_dataset_class(source_name)
        source_train_full = source_dataset_class(
            split="train",
            sampling_frames=config.data.sampling_frames,
            label_mapper=label_mapper,
            is_target=False,
            logger=logger,
        )

        if logger:
            logger.info(f"\n=== Filtering Source Dataset: {source_name} ===")

        source_train_dataset = FilteredDataset(source_train_full, source_allowed_labels, logger)
        source_train_datasets.append(source_train_dataset)
        num_train_source = len(source_train_dataset)

    unified_train_dataset = UnifiedActionDataset(target_train_dataset, source_train_datasets)

    if logger:
        logger.info("\n=== Final Training Dataset Statistics ===")
        logger.info(f"Total training samples: {len(unified_train_dataset)}")
        logger.info(f"  - Target samples: {len(target_train_dataset)}")
        if source_train_datasets:
            logger.info(f"  - Source samples: {len(source_train_datasets[0])}")

    train_loader = DataLoader(
        unified_train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    val_loader = DataLoader(
        target_val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    num_classes = len(target_train_dataset.label_to_idx)
    if logger:
        logger.info("\n=== Model Information ===")
        logger.info(f"Number of classes: {num_classes}")

    model_creator = model_class()
    model = model_creator.create_model(num_classes)
    model = model.to(device)

    trainer = ActionRecognitionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        save_dir=TRAINED_MODELS_DIR / f"method1_{config.dataset_a}_{config.dataset_b}",
        logger=logger,
    )

    return trainer, num_common_labels, num_train_target, num_train_source, num_test


def run_method1_experiments(dataset_a: str, dataset_b: str, metavd_path: Path, logger: logging.Logger) -> ResultLogger:
    """方法Ⅰの全実験パターンを実行.

    実験パターン:
    1. Dataset A (target) - 拡張なし
    2. Dataset A (target) - Dataset B で拡張
    3. Dataset B (target) - 拡張なし
    4. Dataset B (target) - Dataset A で拡張

    Returns:
        ResultLogger: 実験結果を含むロガー
    """
    result_logger = ResultLogger(TRAINED_MODELS_DIR / "results" / "method1" / f"{dataset_a}_{dataset_b}", logger)
    timestamp = get_current_jst_timestamp()

    experiments = [
        Method1ExperimentConfig(
            dataset_a=dataset_a,
            dataset_b=dataset_b,
            target_dataset=dataset_a,
            use_augmentation=False,
        ),
        Method1ExperimentConfig(
            dataset_a=dataset_a,
            dataset_b=dataset_b,
            target_dataset=dataset_a,
            use_augmentation=True,
        ),
        Method1ExperimentConfig(
            dataset_a=dataset_a,
            dataset_b=dataset_b,
            target_dataset=dataset_b,
            use_augmentation=False,
        ),
        Method1ExperimentConfig(
            dataset_a=dataset_a,
            dataset_b=dataset_b,
            target_dataset=dataset_b,
            use_augmentation=True,
        ),
    ]

    for i, config in enumerate(experiments, 1):
        logger.info("\n%s", "=" * 80)
        logger.info(f"EXPERIMENT {i}/{len(experiments)}")
        logger.info("=" * 80)
        logger.info(f"Config: {asdict(config)}")

        trainer, num_common, num_train_target, num_train_source, num_test = create_method1_experiment(
            config=config, model_class=R2Plus1DModel, metavd_path=metavd_path, logger=logger
        )
        trainer.train()

        result = trainer.get_results(
            experiment_id=i,
            num_common_labels=num_common,
            num_train_samples_target=num_train_target,
            num_train_samples_source=num_train_source,
            num_test_samples=num_test,
            timestamp=timestamp,
        )
        result_logger.add_result(result)

        logger.info(f"\nCompleted experiment {i}/{len(experiments)}")

    json_path, csv_path, summary_path = result_logger.save_results(timestamp)

    logger.info("\n%s", "=" * 80)
    logger.info("ALL EXPERIMENTS COMPLETED")
    logger.info("=" * 80)
    logger.info("\nResults have been saved to:")
    logger.info(f"  - JSON:    {json_path}")
    logger.info(f"  - CSV:     {csv_path}")
    logger.info(f"  - Summary: {summary_path}")

    return result_logger


def main():
    dataset_a = "hmdb51"
    dataset_b = "ucf101"

    timestamp = get_current_jst_timestamp()
    (LOGS_DIR / "method1").mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"method1_experiment_{dataset_a}_{dataset_b}_{timestamp}.log"
    logger = setup_logger("method1", log_file)
    logger.info("Starting Method 1 Experiments")

    logger.info("\n=== Method 1: Common Label Evaluation ===")
    logger.info(f"Dataset A: {dataset_a}")
    logger.info(f"Dataset B: {dataset_b}")

    result_logger = run_method1_experiments(
        dataset_a=dataset_a,
        dataset_b=dataset_b,
        metavd_path=AUTO_METAVD_DIR / "auto_metavd_mpnet_0.84.csv",
        logger=logger,
    )

    logger.info("\n=== All Method 1 Experiments Completed ===")
    logger.info("Check the results directory for detailed analysis.")


if __name__ == "__main__":
    main()
