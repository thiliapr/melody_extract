# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025-2026 thiliapr <thiliapr@tutanota.com>
# SPDX-Package: melody_extract
# SPDX-PackageHomePage: https://github.com/thiliapr/melody_extract

import argparse
import pathlib
from typing import Optional, TypedDict
import torch
import mido
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.midi import midi_to_tracks
from utils.model import MAX_INTERVAL, WINDOW_SIZE, Model
from utils.tensorboard import add_pr_curve, add_roc_curve

DEFAULT_BATCH_SIZE = 1024


class EvaluationDatasetOutput(TypedDict):
    """
    评估数据集输出的类型定义，包括文件索引、钢琴卷帘数据、时间间隔数据和主旋律标志。

    Attributes:
        file_index: 文件索引，用于标识数据来源的文件
        piano_roll: 钢琴卷帘数据，形状为 (WINDOW_SIZE, 128)，表示每个时间点的音符状态
        interval: 时间间隔数据，形状为 (WINDOW_SIZE,)，表示相邻时间点的时间差
        melody_flag: 主旋律标志，表示该时间点是否为主旋律音符
    """

    file_index: int
    piano_roll: np.ndarray
    interval: np.ndarray
    melody_flag: np.ndarray


class EvaluationDataset(Dataset):
    def __init__(self, midi_files: list[pathlib.Path]):
        self.data = []
        self.sample_indices = []
        for file_index, midi_file_path in enumerate(midi_files):
            # 加载MIDI文件并转换为音轨
            midi_file = mido.MidiFile(midi_file_path, clip=True)
            tracks = midi_to_tracks(midi_file)

            # 收集所有音轨中的所有时间点，去重排序，创建时间点到索引的映射
            times = sorted({time for track in tracks for _, time in track})
            time_to_index = {time: i for i, time in enumerate(times)}

            # 创建主旋律标志数组，标记哪些时间点有主旋律音符，并更新统计数据
            melody_flags = np.zeros(len(times), dtype=bool)
            for _, time in tracks[0]:  # 假设第一个音轨为主旋律
                melody_flags[time_to_index[time]] = True

            # 计算相邻时间点之间的间隔，并限制在最大间隔范围内
            times = np.array(times, dtype=int)
            intervals = np.clip(times[1:] - times[:-1], 0, MAX_INTERVAL)

            # 填充钢琴卷帘
            piano_roll = np.zeros((len(times), 128), dtype=bool)
            for track in tracks:
                for pitch, time in track:
                    piano_roll[time_to_index[time], pitch] = True

            # 对数据进行填充，以便后续窗口操作
            intervals = np.pad(intervals, (WINDOW_SIZE // 2 + 1, WINDOW_SIZE // 2))
            piano_roll = np.pad(piano_roll, ((WINDOW_SIZE // 2, WINDOW_SIZE // 2), (0, 0)))

            # 将处理好的数据添加到列表中
            self.data.append({
                "intervals": intervals,
                "piano_roll": piano_roll,
                "melody_flags": melody_flags
            })

            # 为每个中心索引添加样本
            for center_index in range(len(melody_flags)):
                self.sample_indices.append((file_index, center_index))

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx) -> EvaluationDatasetOutput:
        file_index, center_index = self.sample_indices[idx]
        data = self.data[file_index]
        return {
            "file_index": file_index,
            "piano_roll": data["piano_roll"][center_index:center_index + WINDOW_SIZE],
            "interval": data["intervals"][center_index:center_index + WINDOW_SIZE],
            "melody_flag": data["melody_flags"][center_index],
        }


def collate_fn(batch: list[EvaluationDatasetOutput]) -> dict:
    # 将批次数据整理为张量字典，便于模型输入
    return {
        "file_index": torch.tensor([item["file_index"] for item in batch], dtype=int),
        "piano_roll": torch.tensor(np.stack([item["piano_roll"] for item in batch], dtype=np.float32)),
        "interval": torch.tensor(np.stack([item["interval"] for item in batch], dtype=np.long)),
        "melody_flag": torch.tensor(np.stack([item["melody_flag"] for item in batch], dtype=np.float32)),
    }


def compute_metrics(logit: torch.Tensor, label: torch.Tensor) -> tuple[float, float]:
    return (
        F.binary_cross_entropy_with_logits(logit, label).item(),
        roc_auc_score((label > 0.5), F.sigmoid(logit))
    )


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    # 解析命令行参数，用于配置评估脚本
    parser = argparse.ArgumentParser(description="评估模型在测试集上的性能")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="检查点文件路径")
    parser.add_argument("midi_dir", type=pathlib.Path, help="测试集 MIDI 文件目录")
    parser.add_argument("log_dir", type=pathlib.Path, help="TensorBoard 日志保存目录")
    parser.add_argument("-b", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="批处理大小，默认为 %(default)s")
    parser.add_argument("-p", "--progress-bar", action="store_true", help="显示进度条，默认不显示")
    return parser.parse_args(args)


@torch.inference_mode()
def main(args: argparse.Namespace):
    # 设置设备（GPU 或 CPU）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 创建 TensorBoard 日志记录器，用于记录评估结果
    writer = SummaryWriter(log_dir=args.log_dir)

    # 加载模型检查点文件
    print("读取并加载检查点 ...")
    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=True)
    model = Model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    # 准备评估数据集
    print("准备数据集 ...")
    midi_files = list(args.midi_dir.rglob("*.mid"))
    dataset = EvaluationDataset(midi_files)
    dataloader = iter(DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn))

    # 执行评估过程，收集每个文件的 logits 和 labels
    print("开始评估 ...")
    logits, labels = (
        [[] for _ in range(len(midi_files))]
        for _ in range(2)
    )
    for batch in tqdm(dataloader, desc="Evaluating", disable=not args.progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(batch)
        loss = F.binary_cross_entropy_with_logits(output["logit"], batch["melody_flag"], reduction="none")
        valid_mask = ~loss.isnan()
        for sample_idx in range(len(output["logit"])):
            if not valid_mask[sample_idx]:
                continue
            file_index = batch["file_index"][sample_idx].item()
            logits[file_index].append(output["logit"][sample_idx].cpu())
            labels[file_index].append(batch["melody_flag"][sample_idx].cpu())

    # 将分组后的数据转换为 Tensor 列表
    logits, labels = (
        [torch.stack(x[i]) for i in range(len(midi_files))]
        for x in (logits, labels)
    )

    print("计算指标 ...")
    # 计算每个文件的指标，找出评估性能最差的文件
    metrics = [
        compute_metrics(file_logits, file_labels)
        for file_logits, file_labels in zip(logits, labels)
    ]
    for file_index in sorted(range(len(midi_files)), key=lambda file_index: metrics[file_index][0])[-5:]:
        nickname = midi_files[file_index].stem
        if len(nickname) > 30:
            nickname = nickname[:15] + "..." + nickname[-15:]
        writer.add_text("Evaluate/Worst by Loss", f"{nickname}: Loss = {metrics[file_index][0]:.4f}, AUC = {metrics[file_index][1]:.4f}", checkpoint["completed_steps"])

    # 计算总体指标
    all_logits = torch.cat(logits)
    all_labels = torch.cat(labels)
    overall_loss, overall_auc = compute_metrics(all_logits, all_labels)
    writer.add_text("Evaluate/Overall Loss", overall_loss, checkpoint["completed_steps"])
    writer.add_text("Evaluate/Overall AUC", overall_auc, checkpoint["completed_steps"])
    add_pr_curve(writer, "Evaluate/Overall PR Curve", all_labels.numpy(), F.sigmoid(all_logits).numpy(), checkpoint["completed_steps"])
    add_roc_curve(writer, "Evaluate/Overall ROC Curve", all_labels.numpy(), F.sigmoid(all_logits).numpy(), checkpoint["completed_steps"])

    print(f"总体 Loss: {overall_loss:.4f}, 总体 AUC: {overall_auc:.4f}")
    writer.close()


if __name__ == "__main__":
    main(parse_args())
