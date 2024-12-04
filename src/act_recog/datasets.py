import csv
import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision.transforms import InterpolationMode, v2

from src.act_recog.config import VideoInfo
from src.act_recog.label_mapper import ActionLabelMapper
from src.settings import ACTIVITYNET_DIR, CHARADES_DIR, HMDB51_DIR, KINETICS700_DIR, STAIR_ACTIONS_DIR, UCF101_DIR


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

    @property
    def video_path(self) -> Path:
        return ACTIVITYNET_DIR

    @property
    def split_path(self) -> Path:
        pass

    def _load_class_mapping(self) -> dict[str, str]:
        pass


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


class Kinetics700Dataset(ActionRecognitionDataset):
    @property
    def dataset_name(self) -> str:
        return "kinetics700"

    @property
    def video_path(self) -> Path:
        return KINETICS700_DIR

    @property
    def split_path(self) -> Path:
        return KINETICS700_DIR / "annotations"

    def _setup_dataset(self) -> None:
        if self.split == "test":
            self.split = "val"

        split_file = self.split_path / f"{self.split}.csv"
        with split_file.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["label"]
                video_id = row["youtube_id"]
                start_time = row["time_start"]
                end_time = row["time_end"]
                video_file = (
                    self.video_path / self.split / label / f"{video_id}_{start_time.zfill(6)}_{end_time.zfill(6)}.mp4"
                )
                if not video_file.exists():
                    continue

                target_label: str | None = None
                if self.is_target:
                    target_label = label
                else:
                    if not self.label_mapper:
                        continue
                    target_label = self.label_mapper.get_target_label(self.dataset_name, label)
                    if not target_label:
                        continue

                video_info = VideoInfo(
                    path=video_file,
                    label=target_label,
                    original_label=label if not self.is_target else None,
                )
                self.videos.append(video_info)
                self.labels.append(target_label)
                self._log_video_load(
                    video_file,
                    target_label,
                    original_label=video_info.original_label,
                )

        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}


class STAIRActionsDataset(ActionRecognitionDataset):
    @property
    def dataset_name(self) -> str:
        return "stair_actions"

    @property
    def video_path(self) -> Path:
        return STAIR_ACTIONS_DIR / "train"

    @property
    def split_path(self) -> Path:
        return STAIR_ACTIONS_DIR / "data"

    def _setup_dataset(self) -> None:
        if self.split == "test":
            self.split = "val"
        split_file = self.split_path / f"videolist_{self.split}.txt"

        if not split_file.exists():
            msg = f"Split file not found: {split_file}"
            raise FileNotFoundError(msg)

        with split_file.open() as f:
            for line in f:
                video_path = line.strip().split()[0]
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

                video_file = self.video_path / class_name / video_path.split("/")[-1]
                if video_file.exists():
                    video_info = VideoInfo(
                        path=video_file, label=target_label, original_label=class_name if not self.is_target else None
                    )
                    self.videos.append(video_info)
                    self.labels.append(target_label)
                    self._log_video_load(video_file, target_label, original_label=video_info.original_label)

        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}


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
