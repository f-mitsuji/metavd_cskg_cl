import random
from pathlib import Path

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, RandomCrop, Resize

from src.settings import HMDB51_DIR, METAVD_DIR, TRAINED_MODELS_DIR, UCF101_DIR


def sample_frames(video: torch.Tensor, n_frames: int, is_train: bool) -> torch.Tensor:
    """動画からn_framesフレームをサンプリング.

    Args:
        video (torch.Tensor): 入力動画 (C, T, H, W)
        n_frames (int): サンプリングするフレーム数
        is_train (bool): 学習時かどうか（Trueならランダム、Falseなら中央）

    Returns:
        torch.Tensor: サンプリングされたフレーム (C, n_frames, H, W)
    """
    # 動画の総フレーム数
    total_frames = video.size(1)  # Tの次元

    if total_frames < n_frames:
        # フレーム数が足りない場合はループで補完
        indices = []
        while len(indices) < n_frames:
            indices.extend(range(total_frames))
        indices = indices[:n_frames]
    elif is_train:
        # ランダムな位置から開始
        start_frame = random.randint(0, total_frames - n_frames)
        indices = list(range(start_frame, start_frame + n_frames))
    else:
        # 中央の16フレームを取得
        center_frame = total_frames // 2
        start_frame = center_frame - (n_frames // 2)
        indices = list(range(start_frame, start_frame + n_frames))

    return video[:, indices]  # (C, T, H, W)


class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        label_map: dict[str, str],
        transform=None,
        split: str | None = None,
        split_file: str | None = None,
        n_frames: int = 16,
    ):
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.label_map = label_map
        self.transform = transform
        self.split = split
        self.split_file = Path(split_file) if split_file else None
        self.n_frames = n_frames
        self.is_train = split == "train"

        # 有効なラベルのセットを保持（ターゲットデータセットのラベル体系）
        self.valid_labels = {v for k, v in label_map.items() if k.startswith(self.dataset_name)}
        self.videos, self.labels = self._load_dataset()

        # ラベルをインデックスに変換
        self.label_to_idx = self._create_label_index()
        self.label_indices = [self.label_to_idx[label] for label in self.labels]

    def __len__(self) -> int:
        """データセットの長さを返す."""
        return len(self.videos)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        video_path = str(self.videos[idx])
        label_idx = self.label_indices[idx]

        # 動画を読み込む
        video, _, _ = read_video(video_path, pts_unit="sec")

        # uint8からfloat32に変換 (0-255 -> 0-1)
        video = video.float() / 255.0

        # NHWC -> NCTHW
        # N=1, H=height, W=width, C=channels
        # 最終的に (C, T, H, W) の形式にする
        video = video.permute(3, 0, 1, 2)  # (C, T, H, W)

        # フレームのサンプリング
        video = sample_frames(video, self.n_frames, self.is_train)  # ここでは (C, T, H, W) のまま

        if self.transform:
            # トランスフォームは (T, C, H, W) を期待するので一時的に変換
            video = video.permute(1, 0, 2, 3)  # (T, C, H, W)
            video = self.transform(video)
            # 戻す
            video = video.permute(1, 0, 2, 3)  # (C, T, H, W)

        return video, label_idx

    def _create_label_index(self) -> dict[str, int]:
        """ラベルを数値インデックスにマッピング."""
        unique_labels = sorted(set(self.labels))
        return {label: idx for idx, label in enumerate(unique_labels)}

    def _load_dataset(self) -> tuple[list[Path], list[str]]:
        """データセットの読み込み."""
        if self.dataset_name.lower() == "hmdb51":
            return self._load_hmdb51()
        if self.dataset_name.lower() == "ucf101":
            return self._load_ucf101()
        msg = f"Unsupported dataset: {self.dataset_name}"
        raise ValueError(msg)

    def _load_hmdb51(self) -> tuple[list[Path], list[str]]:
        """HMDB51データセットの読み込み."""
        videos = []
        labels = []

        # 分割ファイルの読み込み
        split_mapping = {"train": 1, "test": 2, "valid": 0}

        # 各クラスフォルダをチェック
        for class_folder in self.root_dir.iterdir():
            if not class_folder.is_dir() or class_folder.name == "splits":
                continue

            label_key = f"{self.dataset_name}_{class_folder.name}"
            # このクラスがラベルマップに含まれているか確認
            target_label = self.label_map.get(label_key)

            # ターゲットデータセットの場合は全クラスを使用
            # ソースデータセットの場合は拡張対象のクラスのみ使用
            if self.dataset_name == "hmdb51" or (target_label and target_label in self.valid_labels):
                # 分割ファイルを読み込む
                if self.split and self.split_file:
                    split_path = self.split_file / f"{class_folder.name}_test_split1.txt"
                    if split_path.exists():
                        with open(split_path) as f:
                            split_info = {line.split()[0]: int(line.split()[1]) for line in f.readlines()}

                for video_file in class_folder.glob("*.avi"):
                    # 分割チェック
                    if self.split:
                        if (
                            video_file.name not in split_info
                            or split_info[video_file.name] != split_mapping[self.split]
                        ):
                            continue

                    videos.append(video_file)
                    labels.append(target_label or class_folder.name)

        return videos, labels

    def _load_ucf101(self) -> tuple[list[Path], list[str]]:
        """UCF101データセットの読み込み."""
        videos = []
        labels = []

        # 分割ファイルの読み込み
        if self.split:
            split_file = self.split_file / f"{self.split}list01.txt"
            if not split_file.exists():
                msg = f"Split file not found: {split_file}"
                raise FileNotFoundError(msg)

            with open(split_file) as f:
                for line in f:
                    # 形式: v_ApplyEyeMakeup_g01_c01.avi
                    if self.split == "test":
                        video_path = line.strip()
                    else:
                        # 形式: v_ApplyEyeMakeup_g01_c01.avi 1
                        video_path = line.strip().split()[0]

                    # ファイル名からクラス名を抽出
                    # v_ApplyEyeMakeup_g01_c01.avi -> ApplyEyeMakeup
                    class_name = video_path.split("_")[1]

                    # このクラスがラベルマップに含まれているか確認
                    label_key = f"{self.dataset_name}_{class_name}"
                    target_label = self.label_map.get(label_key)

                    # ソースデータセットなので、拡張対象のクラスのみ使用
                    if target_label and target_label in self.valid_labels:
                        video_file = self.root_dir / video_path
                        if video_file.exists():
                            videos.append(video_file)
                            labels.append(target_label)

        return videos, labels


class UnifiedVideoDataset(Dataset):
    def __init__(
        self, target_dataset: VideoDataset, source_datasets: list[VideoDataset], transform=None, n_frames: int = 16
    ):
        self.target_dataset = target_dataset
        self.source_datasets = source_datasets
        self.transform = transform
        self.n_frames = n_frames

        # データセットサイズの計算
        self.dataset_sizes = [len(target_dataset)] + [len(ds) for ds in source_datasets]
        self.cumulative_sizes = [0]
        cum_sum = 0
        for size in self.dataset_sizes:
            cum_sum += size
            self.cumulative_sizes.append(cum_sum)

    def __len__(self) -> int:
        """統合データセットの合計長を返す."""
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        # どのデータセットからサンプリングするか決定
        dataset_idx = 0
        while idx >= self.cumulative_sizes[dataset_idx + 1]:
            dataset_idx += 1

        # 各データセット内でのインデックスを計算
        relative_idx = idx - self.cumulative_sizes[dataset_idx]

        # データの取得
        if dataset_idx == 0:
            video, label_idx = self.target_dataset[relative_idx]
        else:
            video, label_idx = self.source_datasets[dataset_idx - 1][relative_idx]

        return video, label_idx


def get_transforms(is_train: bool) -> Compose:
    transforms = []

    # サイズの変更
    transforms.append(Resize((128, 171)))

    # クロップ
    if is_train:
        transforms.append(RandomCrop((112, 112)))
    else:
        transforms.append(CenterCrop((112, 112)))

    # 正規化
    transforms.append(Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]))

    return Compose(transforms)


def create_datasets(
    target_dataset_path: str,
    source_dataset_paths: list[tuple[str, str]],
    metavd_csv_path: str,
    target_dataset_name: str,
    transform=None,
    split: str | None = None,
    split_files: dict[str, str] | None = None,
    n_frames: int = 16,
) -> tuple[UnifiedVideoDataset, VideoDataset, dict[str, int]]:
    # MetaVDのCSVからラベルマップを作成
    mapping_df = pd.read_csv(metavd_csv_path)
    label_map = {}

    # ターゲットデータセットのラベルを基準にマッピングを作成
    for _, row in mapping_df.iterrows():
        if row["relation"] == "equal":
            from_key = f"{row['from_dataset']}_{row['from_action_name']}"
            to_key = f"{row['to_dataset']}_{row['to_action_name']}"
            if row["to_dataset"] == target_dataset_name:
                # ソースからターゲットへのマッピング
                label_map[from_key] = row["to_action_name"]
                # ターゲット自身のマッピング
                label_map[to_key] = row["to_action_name"]

    # データセットの作成
    target_dataset = VideoDataset(
        root_dir=target_dataset_path,
        dataset_name=target_dataset_name,
        label_map=label_map,
        transform=transform,
        split=split,
        split_file=split_files.get(target_dataset_name),
        n_frames=n_frames,
    )

    source_datasets = []
    for path, name in source_dataset_paths:
        source_dataset = VideoDataset(
            root_dir=path,
            dataset_name=name,
            label_map=label_map,
            transform=transform,
            split=split,
            split_file=split_files.get(name),
            n_frames=n_frames,
        )
        source_datasets.append(source_dataset)

    unified_dataset = UnifiedVideoDataset(target_dataset, source_datasets, transform, n_frames=n_frames)

    return unified_dataset, target_dataset, target_dataset.label_to_idx


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データセットの作成
    train_unified_dataset, train_target_dataset, label_to_idx = create_datasets(
        target_dataset_path=HMDB51_DIR,
        source_dataset_paths=[(UCF101_DIR, "ucf101")],
        metavd_csv_path=METAVD_DIR / "metavd_v1.csv",
        target_dataset_name="hmdb51",
        transform=get_transforms(is_train=True),
        split="train",
        split_files={"hmdb51": HMDB51_DIR / "testTrainMulti_7030_splits", "ucf101": UCF101_DIR / "ucfTrainTestlist"},
    )

    test_dataset = VideoDataset(
        root_dir=HMDB51_DIR,
        dataset_name="hmdb51",
        label_map={f"hmdb51_{label}": label for label in set(train_target_dataset.labels)},
        transform=get_transforms(is_train=False),
        split="test",
        split_file=HMDB51_DIR / "testTrainMulti_7030_splits",
    )

    # DataLoaderの作成
    train_loader = DataLoader(train_unified_dataset, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=16, pin_memory=True)

    # モデルの設定
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    num_classes = len(label_to_idx)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
    nn.init.constant_(model.fc.bias, 0)
    model = model.to(device)

    # Loss関数とオプティマイザの設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    def warmup_schedule(epoch):
        if epoch < 10:  # Warmup期間
            return (epoch + 1) / 10
        return 0.1 ** (epoch // 10)  # 10エポックごとにステップ減衰

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_schedule)

    # 学習ループ
    best_acc = 0
    for epoch in range(1, 46):
        print(f"\nEpoch: {epoch}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = validate(model, test_loader, criterion, device)

        print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%")

        scheduler.step()

        # 最終モデルの保存
        if epoch == 45:
            print("Saving final model..")
            final_state = {
                "model": model.state_dict(),
                "acc": test_acc,
                "epoch": epoch,
            }
            TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(final_state, TRAINED_MODELS_DIR / "final_model.pth")

        # ベストモデルの保存
        if test_acc > best_acc:
            print("Saving best model..")
            best_state = {
                "model": model.state_dict(),
                "acc": test_acc,
                "epoch": epoch,
            }
            torch.save(best_state, TRAINED_MODELS_DIR / "best_model.pth")
            best_acc = test_acc


if __name__ == "__main__":
    main()
