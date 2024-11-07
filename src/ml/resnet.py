import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, TypeVar

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, RandomCrop, Resize

from src.settings import HMDB51_DIR, LOGS_DIR, METAVD_DIR, STAIR_ACTIONS_DIR, TRAINED_MODELS_DIR, UCF101_DIR
from src.utils import get_current_jst_timestamp, setup_logger

# Type Aliases
SplitType: TypeAlias = Literal["train", "test", "val"]
T = TypeVar("T")

# データセットのパスを管理する辞書
DATASET_PATHS = {
    "activitynet": {
        "root": None,
        "split": None,
    },
    "charades": {
        "root": None,
        "split": None,
    },
    "hmdb51": {
        "root": HMDB51_DIR,
        "split": HMDB51_DIR / "testTrainMulti_7030_splits",
    },
    "kinetics": {
        "root": None,
        "split": None,
    },
    "stair_actions": {
        "root": STAIR_ACTIONS_DIR / "STAIR_Actions_v1.1",
        "split": STAIR_ACTIONS_DIR,
    },
    "ucf101": {
        "root": UCF101_DIR,
        "split": UCF101_DIR / "ucfTrainTestlist",
    },
}


@dataclass(frozen=True)
class ExperimentConfig:
    """実験全体の設定."""

    target_dataset: str
    source_datasets: list[str]
    sampling_frames: int = 16
    batch_size: int = 32
    num_workers: int = 16
    pin_memory: bool = True
    # 学習関連の設定
    epochs: int = 45
    warmup_epochs: int = 10
    initial_lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4


class VideoProcessor:
    """動画の前処理を行うクラス."""

    @staticmethod
    def create_transforms(*, is_train: bool) -> Compose:
        """データ変換処理の作成."""
        transforms = [
            Resize((128, 171)),
            RandomCrop((112, 112)) if is_train else CenterCrop((112, 112)),
            Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        ]
        return Compose(transforms)

    @staticmethod
    def sample_frames(video: torch.Tensor, num_frames: int, *, is_train: bool) -> torch.Tensor:
        """フレームのサンプリング."""
        total_frames = video.size(1)  # Tの次元

        if total_frames < num_frames:
            # フレーム数が足りない場合はループで補完
            indices: list[int] = []
            while len(indices) < num_frames:
                indices.extend(range(total_frames))
            indices = indices[:num_frames]
        elif is_train:
            # ランダムな位置から開始
            start_frame = random.randint(0, total_frames - num_frames)
            indices = list(range(start_frame, start_frame + num_frames))
        else:
            # 中央のフレームを取得
            center_frame = total_frames // 2
            start_frame = center_frame - (num_frames // 2)
            indices = list(range(start_frame, start_frame + num_frames))

        return video[:, indices]  # (C, T, H, W)


class ActionLabelMapper:
    """データセット間のラベルマッピングを管理."""

    def __init__(self, mapping_file: Path):
        self.mapping_file = mapping_file
        # (target_dataset, target_action) -> list[(source_dataset, source_action)]
        self.mapping: dict[tuple[str, str], list[tuple[str, str]]] = {}
        # (source_dataset, source_action) -> target_action
        self.reverse_mapping: dict[tuple[str, str], str] = {}

    def load_mapping(self, target_dataset: str) -> None:
        """マッピング情報の読み込み."""
        mapping_df = pd.read_csv(self.mapping_file)
        self.mapping.clear()
        self.reverse_mapping.clear()

        # ターゲットデータセットがfrom_datasetにある行を探す
        target_actions = mapping_df[mapping_df["from_dataset"] == target_dataset]

        for _, row in target_actions.iterrows():
            if row["relation"] == "equal":
                target_key = (row["from_dataset"], row["from_action_name"])
                source_pair = (row["to_dataset"], row["to_action_name"])

                # 正方向のマッピング(ターゲット→ソース)
                if target_key not in self.mapping:
                    self.mapping[target_key] = []
                self.mapping[target_key].append(source_pair)

                # 逆方向のマッピング(ソース→ターゲット)
                self.reverse_mapping[source_pair] = row["from_action_name"]

    def get_target_label(self, source_dataset: str, source_action: str) -> str | None:
        """ソースデータセットのラベルに対応するターゲットラベルを取得."""
        return self.reverse_mapping.get((source_dataset, source_action))


class ActionRecognitionDataset(Dataset):
    """行動認識データセットの基底クラス."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        root_path: Path,
        split_path: Path,
        sampling_frames: int,
        label_mapper: ActionLabelMapper,
        *,
        is_target: bool,
        transform: Compose | None = None,
        logger: logging.Logger | None = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.root_path = root_path
        self.split_path = split_path
        self.sampling_frames = sampling_frames
        self.is_target = is_target
        self.label_mapper = label_mapper
        self.transform = transform
        self.logger = logger
        self.is_train = split == "train"

        self.video_paths: list[Path] = []
        self.labels: list[str] = []
        self.label_to_idx: dict[str, int] = {}

        self._setup_dataset()

    def _setup_dataset(self) -> None:
        """データセットの初期化(サブクラスで実装)."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        video_path = str(self.video_paths[idx])
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]

        try:
            video, _, _ = read_video(video_path, pts_unit="sec")
            video = self._preprocess_video(video)
        except Exception:
            if self.logger:
                self.logger.exception(f"Error loading video {idx}: {video_path}")
            raise
        else:
            return video, label_idx

    def _preprocess_video(self, video: torch.Tensor) -> torch.Tensor:
        """動画の前処理."""
        # uint8からfloat32に変換 (0-255 -> 0-1)
        video = video.float() / 255.0

        # NHWC -> NCTHW -> CTHW
        video = video.permute(3, 0, 1, 2)  # (C, T, H, W)

        # フレームのサンプリング
        video = VideoProcessor.sample_frames(video, self.sampling_frames, is_train=self.is_train)

        if self.transform:
            # トランスフォームは (T, C, H, W) を期待するので変換
            video = video.permute(1, 0, 2, 3)  # (T, C, H, W)
            video = self.transform(video)
            # 戻す
            video = video.permute(1, 0, 2, 3)  # (C, T, H, W)

        return video

    def _log_video_load(self, video_path: Path, label: str, original_label: str | None = None) -> None:
        """動画の読み込みをログ."""
        if not self.logger:
            return

        if self.is_target:
            self.logger.info(
                f"Loaded video from {self.dataset_name} (target) [{self.split}]: "
                f"file='{video_path.name}', label='{label}'"
            )
        else:
            self.logger.info(
                f"Loaded video from {self.dataset_name} (source) [{self.split}]: "
                f"file='{video_path.name}', original_label='{original_label}' -> target_label='{label}'"
            )


class ActivityNetDataset(ActionRecognitionDataset):
    pass


class CharadesDataset(ActionRecognitionDataset):
    pass


class HMDB51Dataset(ActionRecognitionDataset):
    def _setup_dataset(self) -> None:
        """データセットの初期化."""
        split_mapping = {"train": 1, "test": 2, "val": 0}

        for class_folder in self.root_path.iterdir():
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
                        self.video_paths.append(video_file)
                        self.labels.append(target_label)
                        self._log_video_load(
                            video_file, target_label, original_label=orig_label if not self.is_target else None
                        )

        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}


class KineticsDataset(ActionRecognitionDataset):
    pass


class STAIRActionsDataset(ActionRecognitionDataset):
    pass


class UCF101Dataset(ActionRecognitionDataset):
    def _setup_dataset(self) -> None:
        """データセットの初期化."""
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

                video_file = self.root_path / video_path.split("/")[-1]
                if video_file.exists():
                    self.video_paths.append(video_file)
                    self.labels.append(target_label)
                    self._log_video_load(
                        video_file, target_label, original_label=class_name if not self.is_target else None
                    )

        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}


class UnifiedActionDataset(Dataset):
    """複数データセットを統合するデータセット."""

    def __init__(
        self,
        target_dataset: ActionRecognitionDataset,
        source_datasets: Sequence[ActionRecognitionDataset],
    ):
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
    """行動認識モデルの学習を管理."""

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
            lr=self.config.initial_lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

    def _create_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        def warmup_schedule(epoch: int) -> float:
            if epoch <= self.config.warmup_epochs:
                return (epoch + 1) / 10
            return 0.1 ** (epoch // 10)

        return optim.lr_scheduler.LambdaLR(self.optimizer, warmup_schedule)

    def train(self) -> None:
        best_acc = 0.0
        self.save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.epochs + 1):
            if self.logger:
                self.logger.info(f"\nEpoch {epoch}/{self.config.epochs}")

            # 訓練
            train_loss, train_acc = self._train_epoch()
            if self.logger:
                self.logger.info(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%")

            # 検証
            val_loss, val_acc = self._validate()
            if self.logger:
                self.logger.info(f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%")

            # スケジューラの更新
            self.scheduler.step()

            # モデルの保存
            if epoch == self.config.epochs:
                self._save_model("final", val_acc, epoch)

            if val_acc > best_acc:
                self._save_model("best", val_acc, epoch)
                best_acc = val_acc

    def _train_epoch(self) -> tuple[float, float]:
        self.model.train()
        return self._process_epoch(is_train=True)

    def _validate(self) -> tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            return self._process_epoch(is_train=False)

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

    def _save_model(self, name: str, acc: float, epoch: int) -> None:
        if self.logger:
            self.logger.info(f"Saving {name} model..")

        state = {
            "model": self.model.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }

        save_path = self.save_dir / f"{name}_model.pth"
        torch.save(state, save_path)


def get_dataset_class(name: str) -> type[ActionRecognitionDataset]:
    """データセット名からデータセットクラスを取得."""
    dataset_mapping = {
        "activitynet": ActivityNetDataset,
        "charades": CharadesDataset,
        "kinetics": KineticsDataset,
        "hmdb51": HMDB51Dataset,
        "stair_actions": STAIRActionsDataset,
        "ucf101": UCF101Dataset,
    }
    if name not in dataset_mapping:
        msg = f"Unsupported dataset: {name}"
        raise ValueError(msg)
    return dataset_mapping[name]


def create_experiment(
    config: ExperimentConfig,
    metavd_path: Path,
    logger: logging.Logger | None = None,
) -> tuple[ActionRecognitionTrainer, dict]:
    """実験環境の作成."""
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if logger:
        logger.info(f"Using device: {device}")

    # ラベルマッピングの作成
    label_mapper = ActionLabelMapper(metavd_path)
    label_mapper.load_mapping(config.target_dataset)

    # トランスフォームの作成
    train_transform = VideoProcessor.create_transforms(is_train=True)
    val_transform = VideoProcessor.create_transforms(is_train=False)

    # ターゲットデータセットの作成
    target_paths = DATASET_PATHS[config.target_dataset]
    dataset_class = get_dataset_class(config.target_dataset)

    target_train_dataset = dataset_class(
        dataset_name=config.target_dataset,
        split="train",
        root_path=target_paths["root"],
        split_path=target_paths["split"],
        sampling_frames=config.sampling_frames,
        label_mapper=label_mapper,
        is_target=True,
        transform=train_transform,
        logger=logger,
    )

    target_val_dataset = dataset_class(
        dataset_name=config.target_dataset,
        split="test",
        root_path=target_paths["root"],
        split_path=target_paths["split"],
        sampling_frames=config.sampling_frames,
        label_mapper=label_mapper,
        is_target=True,
        transform=val_transform,
        logger=logger,
    )

    # ソースデータセットの作成
    source_train_datasets = []
    for source_name in config.source_datasets:
        source_paths = DATASET_PATHS[source_name]
        dataset_class = get_dataset_class(source_name)

        source_dataset = dataset_class(
            dataset_name=source_name,
            split="train",
            root_path=source_paths["root"],
            split_path=source_paths["split"],
            sampling_frames=config.sampling_frames,
            label_mapper=label_mapper,
            is_target=False,
            transform=train_transform,
            logger=logger,
        )
        source_train_datasets.append(source_dataset)

    # 統合データセット
    unified_train_dataset = UnifiedActionDataset(target_train_dataset, source_train_datasets)

    # データローダーの作成
    train_loader = DataLoader(
        unified_train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    val_loader = DataLoader(
        target_val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    # モデルの作成
    num_classes = len(target_train_dataset.label_to_idx)
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
    nn.init.constant_(model.fc.bias, 0)
    model = model.to(device)

    # トレーナーの作成
    trainer = ActionRecognitionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        save_dir=TRAINED_MODELS_DIR,
        logger=logger,
    )

    metadata = {
        "num_classes": num_classes,
        "target_dataset": config.target_dataset,
        "source_datasets": config.source_datasets,
        "label_mapping": label_mapper.mapping,
    }

    return trainer, metadata


def main():
    # ロガーの設定
    timestamp = get_current_jst_timestamp()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"training_{timestamp}.log"
    logger = setup_logger("video_training", log_file)
    logger.info("Starting training script")

    config = ExperimentConfig(
        # target_dataset="ucf101",
        target_dataset="hmdb51",
        # source_datasets=["hmdb51"],
        source_datasets=[],
    )

    # 実験の実行
    trainer, metadata = create_experiment(
        config=config,
        metavd_path=METAVD_DIR / "metavd_v1.csv",
        logger=logger,
    )

    if logger:
        logger.info("Experiment metadata:")
        for key, value in metadata.items():
            logger.info(f"{key}: {value}")

    trainer.train()


if __name__ == "__main__":
    main()
