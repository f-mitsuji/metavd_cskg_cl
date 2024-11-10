import csv
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
from torchvision.transforms import InterpolationMode, v2

from src.settings import CHARADES_DIR, HMDB51_DIR, LOGS_DIR, METAVD_DIR, TRAINED_MODELS_DIR, UCF101_DIR
from src.utils import get_current_jst_timestamp, setup_logger


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 45
    warmup_epochs: int = 10
    initial_lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class DataConfig:
    sampling_frames: int = 16
    batch_size: int = 32
    num_workers: int = 16
    pin_memory: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    target_dataset: str
    source_datasets: list[str]
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)


@dataclass
class VideoInfo:
    path: Path
    label: str
    original_label: str | None = None
    start_time: float | None = None
    end_time: float | None = None


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


class VideoProcessor:
    def __init__(self, sampling_frames: int, *, is_train: bool):
        self.sampling_frames = sampling_frames
        self.is_train = is_train
        self.transform = self._create_transforms()

    def _create_transforms(self) -> v2.Compose:
        # R2Plus1D_18_Weights.KINETICS400_V1.transforms()
        transforms = [
            v2.Resize((128, 171), interpolation=InterpolationMode.BILINEAR, antialias=True),
            v2.RandomCrop((112, 112)) if self.is_train else v2.CenterCrop((112, 112)),
            v2.ToDtype(torch.float32, scale=True),
            # Normalize with mean and std from Kinetics400 dataset
            v2.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        ]
        return v2.Compose(transforms)

    def load_video(self, video_info: VideoInfo) -> torch.Tensor:
        if video_info.start_time is not None and video_info.end_time is not None:
            video, _, _ = read_video(
                str(video_info.path),
                start_pts=video_info.start_time,
                end_pts=video_info.end_time,
                pts_unit="sec",
                output_format="TCHW",
            )
        else:
            video, _, _ = read_video(str(video_info.path), pts_unit="sec", output_format="TCHW")

        return self._sample_frames(video)

    def _sample_frames(self, video: torch.Tensor) -> torch.Tensor:
        total_frames = video.size(0)

        if total_frames < self.sampling_frames:
            indices: list[int] = []
            while len(indices) < self.sampling_frames:
                indices.extend(range(total_frames))
            indices = indices[: self.sampling_frames]
        elif self.is_train:
            start_frame = random.randint(0, total_frames - self.sampling_frames)
            indices = list(range(start_frame, start_frame + self.sampling_frames))
        else:
            center_frame = total_frames // 2
            start_frame = center_frame - (self.sampling_frames // 2)
            indices = list(range(start_frame, start_frame + self.sampling_frames))

        sampled_video = video[indices]
        return self.transform(sampled_video)


class ActionRecognitionDataset(Dataset, ABC):
    def __init__(
        self,
        split: str,
        sampling_frames: int,
        label_mapper: ActionLabelMapper,
        *,
        is_target: bool,
        logger: logging.Logger | None = None,
    ):
        super().__init__()
        self.videos: list[VideoInfo] = []
        self.split = split
        self.sampling_frames = sampling_frames
        self.is_target = is_target
        self.label_mapper = label_mapper
        self.logger = logger
        self.is_train = split == "train"

        self.labels: list[str] = []
        self.label_to_idx: dict[str, int] = {}

        self._setup_dataset()
        self.video_processor = VideoProcessor(sampling_frames, is_train=self.is_train)

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        pass

    @property
    @abstractmethod
    def video_path(self) -> Path:
        pass

    @property
    @abstractmethod
    def split_path(self) -> Path:
        pass

    @abstractmethod
    def _setup_dataset(self) -> None:
        pass

    def _log_video_load(
        self,
        video_path: Path,
        label: str,
        original_label: str | None = None,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> None:
        if not self.logger:
            return

        time_info = ""
        if start_time is not None and end_time is not None:
            time_info = f" [time: {start_time:.2f}-{end_time:.2f}]"

        if self.is_target:
            self.logger.info(
                f"Loaded video from {self.dataset_name} (target) [{self.split}]: "
                f"file='{video_path.name}', label='{label}'{time_info}"
            )
        else:
            self.logger.info(
                f"Loaded video from {self.dataset_name} (source) [{self.split}]: "
                f"file='{video_path.name}', original_label='{original_label}' -> target_label='{label}'{time_info}"
            )

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        video_info = self.videos[idx]
        label_idx = self.label_to_idx[video_info.label]

        try:
            video = self.video_processor.load_video(video_info)
            video = video.permute(1, 0, 2, 3)  # R(2+1)D-18 expects CTHW format

        except Exception:
            if self.logger:
                self.logger.exception(f"Error loading video {idx}: {video_info.path}")
            raise

        return video, label_idx


class ActivityNetDataset(ActionRecognitionDataset):
    @property
    def dataset_name(self) -> str:
        return "activitynet"


class CharadesDataset(ActionRecognitionDataset):
    @property
    def dataset_name(self) -> str:
        return "charades"

    @property
    def video_path(self) -> Path:
        return CHARADES_DIR / "Charades_v1_480"

    @property
    def split_path(self) -> Path:
        return CHARADES_DIR / "Charades"

    def _load_class_mapping(self) -> dict[str, str]:
        class_file = self.split_path / "Charades_v1_classes.txt"
        if not class_file.exists():
            msg = f"Class mapping file not found: {class_file}"
            raise FileNotFoundError(msg)

        class_mapping = {}
        with class_file.open() as f:
            for line in f:
                code, name = line.strip().split(" ", 1)
                class_mapping[code] = name

        return class_mapping

    def _setup_dataset(self) -> None:
        class_mapping = self._load_class_mapping()

        split_file = self.split_path / f"Charades_v1_{self.split}.csv"
        if not split_file.exists():
            msg = f"Split file not found: {split_file}"
            raise FileNotFoundError(msg)

        with split_file.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row["id"]
                video_file = self.video_path / f"{video_id}.mp4"
                if not video_file.exists():
                    continue

                actions = row["actions"].split(";")
                for action in actions:
                    if not action.strip():
                        continue

                    expected_parts = 3  # "class start end" triplets
                    parts = action.strip().split()
                    if len(parts) != expected_parts:
                        continue

                    action_code, t1, t2 = parts

                    t1 = float(t1)
                    t2 = float(t2)
                    start_time = min(t1, t2)
                    end_time = max(t1, t2)

                    class_name = class_mapping.get(action_code)
                    if not class_name:
                        if self.logger:
                            self.logger.warning(f"Unknown action code: {action_code}")
                        continue

                    target_label: str | None = None
                    if self.is_target:
                        target_label = class_name
                    else:
                        if not self.label_mapper:
                            continue
                        target_label = self.label_mapper.get_target_label(self.dataset_name, class_name)
                        if not target_label:
                            continue

                    video_info = VideoInfo(
                        path=video_file,
                        label=target_label,
                        original_label=class_name if not self.is_target else None,
                        start_time=float(start_time),
                        end_time=float(end_time),
                    )
                    self.videos.append(video_info)
                    self.labels.append(target_label)
                    self._log_video_load(
                        video_file,
                        target_label,
                        original_label=video_info.original_label,
                        start_time=start_time,
                        end_time=end_time,
                    )

        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}


class HMDB51Dataset(ActionRecognitionDataset):
    @property
    def dataset_name(self) -> str:
        return "hmdb51"

    @property
    def video_path(self) -> Path:
        return HMDB51_DIR

    @property
    def split_path(self) -> Path:
        return HMDB51_DIR / "testTrainMulti_7030_splits"

    def _setup_dataset(self) -> None:
        split_mapping = {"train": 1, "test": 2, "val": 0}

        for class_folder in self.video_path.iterdir():
            if not class_folder.is_dir() or class_folder.name == "testTrainMulti_7030_splits":
                continue

            orig_label = class_folder.name
            target_label: str | None = None

            if self.is_target:
                target_label = orig_label
            else:
                if not self.label_mapper:
                    continue
                target_label = self.label_mapper.get_target_label(self.dataset_name, orig_label)
                if not target_label:
                    continue

            split_path = self.split_path / f"{orig_label}_test_split1.txt"
            if split_path.exists():
                with split_path.open() as f:
                    split_info = {line.split()[0]: int(line.split()[1]) for line in f.readlines()}
                for video_file in class_folder.glob("*.avi"):
                    if video_file.name in split_info and split_info[video_file.name] == split_mapping[self.split]:
                        video_info = VideoInfo(
                            path=video_file,
                            label=target_label,
                            original_label=orig_label if not self.is_target else None,
                        )
                        self.videos.append(video_info)
                        self.labels.append(target_label)
                        self._log_video_load(video_file, target_label, original_label=video_info.original_label)

        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}


class Kinetics7002020Dataset(ActionRecognitionDataset):
    @property
    def dataset_name(self) -> str:
        return "kinetics700"


class STAIRActionsDataset(ActionRecognitionDataset):
    @property
    def dataset_name(self) -> str:
        return "stair_actions"


class UCF101Dataset(ActionRecognitionDataset):
    @property
    def dataset_name(self) -> str:
        return "ucf101"

    @property
    def video_path(self) -> Path:
        return UCF101_DIR

    @property
    def split_path(self) -> Path:
        return UCF101_DIR / "ucfTrainTestlist"

    def _setup_dataset(self) -> None:
        split_file = self.split_path / f"{self.split}list01.txt"
        if not split_file.exists():
            msg = f"Split file not found: {split_file}"
            raise FileNotFoundError(msg)

        with split_file.open() as f:
            for line in f:
                video_path = line.strip() if self.split == "test" else line.strip().split()[0]
                class_name = video_path.split("/")[0]
                target_label: str | None = None

                if self.is_target:
                    target_label = class_name
                else:
                    if not self.label_mapper:
                        continue
                    target_label = self.label_mapper.get_target_label(self.dataset_name, class_name)
                    if not target_label:
                        continue

                video_file = self.video_path / video_path.split("/")[-1]
                if video_file.exists():
                    video_info = VideoInfo(
                        path=video_file, label=target_label, original_label=class_name if not self.is_target else None
                    )
                    self.videos.append(video_info)
                    self.labels.append(target_label)
                    self._log_video_load(video_file, target_label, original_label=video_info.original_label)

        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}


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
            "timestamp": get_current_jst_timestamp(),
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
        "kinetics700-2020": Kinetics7002020Dataset,
        "hmdb51": HMDB51Dataset,
        "stair_actions": STAIRActionsDataset,
        "ucf101": UCF101Dataset,
    }
    if name not in dataset_mapping:
        msg = f"Unsupported dataset: {name}"
        raise ValueError(msg)
    return dataset_mapping[name]


def create_experiment(
    config: ExperimentConfig, metavd_path: Path, logger: logging.Logger | None = None
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
        logger.info(f"Total number of samples: {len(target_train_dataset)}")
        logger.info("\nSamples per class:")
        for label, count in sorted(target_class_counts.items()):
            logger.info(f"{label}: {count}")

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

    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
    nn.init.constant_(model.fc.bias, 0)
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
        # target_dataset="ucf101",
        target_dataset="charades",
        # target_dataset="hmdb51",
        # source_datasets=["hmdb51"],
        source_datasets=[],
        # source_datasets=["ucf101"],
    )
    logger.info(f"Experiment config: {asdict(config)}")

    trainer = create_experiment(config=config, metavd_path=METAVD_DIR / "metavd_v1.csv", logger=logger)
    trainer.train()


if __name__ == "__main__":
    main()
