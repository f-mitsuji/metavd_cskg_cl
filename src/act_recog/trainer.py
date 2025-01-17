import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src.act_recog.config import ExperimentConfig
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
    LOGS_DIR,
    METAVD_DIR,
    TRAINED_MODELS_DIR,
)
from src.utils import get_current_jst_timestamp, setup_logger


class UnifiedActionDataset(Dataset):
    def __init__(self, target_dataset: ActionRecognitionDataset, source_datasets: Sequence[ActionRecognitionDataset]):
        self.target_dataset = target_dataset
        self.source_datasets = source_datasets

        # データセットサイズの計算
        self.dataset_sizes = [len(target_dataset)] + [len(ds) for ds in source_datasets]
        self.cumulative_sizes = [0]
        cum_sum = 0
        for size in self.dataset_sizes:
            cum_sum += size
            self.cumulative_sizes.append(cum_sum)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # どのデータセットからサンプリングするか決定
        dataset_idx = 0
        while idx >= self.cumulative_sizes[dataset_idx + 1]:
            dataset_idx += 1

        # 各データセット内でのインデックスを計算
        relative_idx = idx - self.cumulative_sizes[dataset_idx]

        # データの取得
        if dataset_idx == 0:
            return self.target_dataset[relative_idx]
        return self.source_datasets[dataset_idx - 1][relative_idx]


class ActionRecognitionTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ExperimentConfig,
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
        target = self.config.target_dataset

        source = "-".join(sorted(self.config.source_datasets)) if self.config.source_datasets else "no_source"

        params = [f"e{epoch}", f"lr{lr}".replace(".", "_"), f"acc{acc:.2f}".replace(".", "_")]

        params_str = "_".join(params)

        return f"{target}_from_{source}_{params_str}_{timestamp}_{name}"

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

            if epoch == self.config.training.epochs - 1:
                self._save_model("final", val_acc, epoch + 1, current_lr)

            if val_acc > best_acc:
                # self._save_model("best", val_acc, epoch + 1, current_lr)
                best_acc = val_acc


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


def create_experiment(
    config: ExperimentConfig, model_class, metavd_path: Path, logger: logging.Logger | None = None
) -> ActionRecognitionTrainer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if logger:
        logger.info(f"Using device: {device}")

    label_mapper = ActionLabelMapper(metavd_path)
    label_mapper.load_mapping(config.target_dataset)

    dataset_class = get_dataset_class(config.target_dataset)

    target_train_dataset = dataset_class(
        split="train",
        sampling_frames=config.data.sampling_frames,
        label_mapper=label_mapper,
        is_target=True,
        logger=logger,
    )

    target_val_dataset = dataset_class(
        split="test",
        sampling_frames=config.data.sampling_frames,
        label_mapper=label_mapper,
        is_target=True,
        logger=logger,
    )

    target_class_counts: dict[str, int] = {}
    for label in target_train_dataset.labels:
        target_class_counts[label] = target_class_counts.get(label, 0) + 1

    if logger:
        logger.info("\n=== Target Dataset Statistics ===")
        logger.info(f"Total number of train samples: {len(target_train_dataset)}")
        logger.info("\nSamples per class:")
        for label, count in sorted(target_class_counts.items()):
            logger.info(f"{label}: {count}")
        logger.info(f"Total number of test samples: {len(target_val_dataset)}")

    source_train_datasets = []
    source_class_counts: dict[str, int] = {}
    total_source_samples = 0
    for source_name in config.source_datasets:
        if logger:
            logger.info(f"\n=== Loading source dataset: {source_name} ===")

        dataset_class = get_dataset_class(source_name)
        source_dataset = dataset_class(
            split="train",
            sampling_frames=config.data.sampling_frames,
            label_mapper=label_mapper,
            is_target=False,
            logger=logger,
        )
        source_train_datasets.append(source_dataset)

        for label in source_dataset.labels:
            source_class_counts[label] = source_class_counts.get(label, 0) + 1

        total_source_samples += len(source_dataset)

    if source_train_datasets and logger:
        logger.info("\n=== Source Datasets Statistics ===")
        logger.info(f"Total number of augmented samples: {total_source_samples}")
        logger.info("\nAugmented samples per class:")
        for label, count in sorted(source_class_counts.items()):
            original_count = target_class_counts.get(label, 0)
            logger.info(f"{label}: {count} (Original: {original_count}, Total: {original_count + count})")

    unified_train_dataset = UnifiedActionDataset(target_train_dataset, source_train_datasets)

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

    return ActionRecognitionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        save_dir=TRAINED_MODELS_DIR,
        logger=logger,
    )


def main():
    timestamp = get_current_jst_timestamp()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"action_recognition_{timestamp}.log"
    logger = setup_logger("resnet", log_file)
    logger.info("Starting training script")

    config = ExperimentConfig(
        target_dataset="activitynet",
        # target_dataset="stair_actions",
        # target_dataset="charades",
        # target_dataset="hmdb51",
        # target_dataset="kinetics700",
        # source_datasets=["hmdb51"],
        source_datasets=[],
        # source_datasets=["ucf101"],
    )
    logger.info(f"Experiment config: {asdict(config)}")

    trainer = create_experiment(
        config=config, model_class=R2Plus1DModel, metavd_path=METAVD_DIR / "metavd_v1.csv", logger=logger
    )
    trainer.train()


if __name__ == "__main__":
    main()
