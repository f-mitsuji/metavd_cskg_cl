from dataclasses import dataclass, field
from pathlib import Path


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
