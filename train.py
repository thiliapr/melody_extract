# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025-2026 thiliapr <thiliapr@tutanota.com>
# SPDX-Package: melody_extract
# SPDX-PackageHomePage: https://github.com/thiliapr/melody_extract

import argparse
import pathlib
import random
import json
from typing import Iterator, Optional, TypedDict
import mido
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.nn import functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.midi import midi_to_tracks
from utils.model import MAX_INTERVAL, WINDOW_SIZE, Model
from utils.tensorboard import add_pr_curve, add_roc_curve

DEFAULT_BATCH_SIZE = 512
DEFAULT_DROPOUT = 0.5


class Placeholder:
    def return_self(self, *args, **kwargs):
        return self

    __getattr__ = __call__ = return_self


class DatasetOutput(TypedDict):
    """
    数据集输出的类型定义

    Attributes:
        piano_roll: 钢琴卷帘数据，形状为 (WINDOW_SIZE, 128)
        interval: 时间间隔数据，形状为 (WINDOW_SIZE,)
        melody_flag: 主旋律标志
    """

    piano_roll: np.ndarray
    interval: np.ndarray
    melody_flag: np.ndarray


class DatasetWithoutAugmentation(IterableDataset):
    def __init__(self, midi_dir: pathlib.Path):
        super().__init__()
        self.data = []
        for midi_file in tqdm(midi_dir.glob("**/*.mid"), desc="Loading MIDI files"):
            # 加载 MIDI 文件数据
            midi_file = mido.MidiFile(midi_file, clip=True)
            tracks = midi_to_tracks(midi_file)

            # 收集所有音轨中的所有时间点，去重排序，创建时间点到索引的映射
            times = sorted({time for track in tracks for _, time in track})
            time_to_index = {time: i for i, time in enumerate(times)}

            # 创建主旋律标志数组，标记哪些时间点有主旋律音符，并更新统计数据
            melody_flags = np.zeros(len(times), dtype=bool)
            for _, time in tracks[0]:
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

    def __iter__(self) -> Iterator[DatasetOutput]:
        while True:
            # 随机选择一个文件和中心位置
            data = random.choice(self.data)
            center_index = random.randint(0, len(data["melody_flags"]) - 1)

            # 提取窗口数据
            yield {
                "piano_roll": data["piano_roll"][center_index:center_index + WINDOW_SIZE],
                "interval": data["intervals"][center_index:center_index + WINDOW_SIZE],
                "melody_flag": data["melody_flags"][center_index],
            }


class DatasetWithAugmentation(IterableDataset):
    def __init__(self, midi_dir: pathlib.Path):
        super().__init__()

        # 初始化数据列表和全局正负样本权重统计
        self.data: list[list[dict[str, np.ndarray]]] = []
        global_pos_weights = []
        if (metadata_path := midi_dir / "metadata.json").exists():
            metadata = json.loads(metadata_path.read_text("utf-8"))
        else:
            metadata = {}
        self.group_weights = []

        # 遍历 MIDI 目录中的每个组（子目录），并处理其中的 MIDI 文件
        for group_dir in midi_dir.iterdir():
            # 仅处理目录
            if not group_dir.is_dir():
                continue

            # 初始化组内数据和正负样本权重
            group = []
            group_pos_weights = []

            # 遍历组目录中的每个 MIDI 文件，加载并处理数据
            for midi_file in tqdm(group_dir.rglob("*.mid"), desc=f"Loading Group `{group_dir.name}`"):
                # 加载 MIDI 文件数据
                midi_file = mido.MidiFile(midi_file, clip=True)
                tracks = midi_to_tracks(midi_file)
                melody_notes = tracks[0]

                # 收集所有音轨中的所有时间点，去重排序，创建时间点到索引的映射
                times = sorted({time for track in tracks for _, time in track})
                time_to_index = {time: i for i, time in enumerate(times)}

                # 创建主旋律标志数组，标记哪些时间点有主旋律音符
                melody_flags = np.zeros(len(times), dtype=bool)
                for _, time in melody_notes:
                    melody_flags[time_to_index[time]] = True
                
                # 更新正负样本权重统计，计算当前文件中负样本与正样本的比例，并添加到组内统计列表中
                group_pos_weights.append((~melody_flags).sum() / melody_flags.sum())

                # 计算相邻时间点之间的间隔，并限制在最大间隔范围内
                times = np.array(times, dtype=int)
                intervals = np.clip(times[1:] - times[:-1], 0, MAX_INTERVAL)

                # 填充主旋律钢琴卷帘和音高统计
                melody_piano_roll = np.zeros((len(times), 128), dtype=bool)
                melody_max_pitch = np.zeros(len(times), dtype=np.uint8)
                melody_min_pitch = np.full(len(times), 127, dtype=np.uint8)
                for pitch, time in melody_notes:
                    melody_max_pitch[time_to_index[time]] = max(melody_max_pitch[time_to_index[time]], pitch)
                    melody_min_pitch[time_to_index[time]] = min(melody_min_pitch[time_to_index[time]], pitch)
                    melody_piano_roll[time_to_index[time], pitch] = True

                # 填充伴奏钢琴卷帘和音高统计
                accomp_piano_roll = np.zeros((len(times), 128), dtype=bool)
                accomp_max_pitch = np.zeros(len(times), dtype=np.uint8)
                accomp_min_pitch = np.full(len(times), 127, dtype=np.uint8)
                for track in tracks[1:]:
                    for pitch, time in track:
                        accomp_max_pitch[time_to_index[time]] = max(accomp_max_pitch[time_to_index[time]], pitch)
                        accomp_min_pitch[time_to_index[time]] = min(accomp_min_pitch[time_to_index[time]], pitch)
                        accomp_piano_roll[time_to_index[time], pitch] = True

                # 计算每个时间点的音域偏移限制，以及每个窗口的时间间隔最大公约数
                melody_down_octaves = np.empty(len(times), dtype=np.uint8)
                melody_up_octaves = np.empty(len(times), dtype=np.int8)
                accomp_shifts = np.empty(len(times), dtype=np.int8)
                window_gcd = np.empty(len(times), dtype=int)
                for i in range(len(times)):
                    # 计算当前窗口范围（以当前时间点为中心）
                    start = max(0, i - WINDOW_SIZE // 2)
                    end = min(len(times), i + WINDOW_SIZE // 2 + 1)
                    # 计算窗口时间间隔最大公约数
                    window_gcd[i] = np.gcd.reduce(intervals[start:end])
                    # 计算窗口内最大和最小音高
                    window_melody_max = melody_max_pitch[start:end].max()
                    window_melody_min = melody_min_pitch[start:end].min()
                    window_accomp_max = accomp_max_pitch[start:end].max()
                    window_accomp_min = accomp_min_pitch[start:end].min()
                    # 如果当前窗口只有伴奏或者主旋律音符，则跳过当前窗口
                    # 如果最大值小于最小值，这说明这个窗口的统计数据都没有被改变，进而说明这个窗口没有对应类型的音符
                    if window_melody_max < window_melody_min or window_accomp_max < window_accomp_min:
                        melody_up_octaves[i] = -1  # 标志跳过该窗口
                        continue
                    # 计算主旋律在当前窗口内可向下移动的最大八度数，向下移动后还可向上移动的最大八度数
                    melody_down_octaves[i] = window_melody_min // 12
                    melody_up_octaves[i] = (127 - (window_melody_max - melody_down_octaves[i] * 12)) // 12
                    # 计算伴奏移动的半音数，使伴奏居于中下的音域（以12个半音为单位）
                    accomp_up_octaves = (127 - window_accomp_max).item() // 12
                    accomp_down_octaves = window_accomp_min.item() // 12
                    accomp_shifts[i] = max(accomp_up_octaves - accomp_down_octaves - 2, -accomp_down_octaves) * 12

                # 对数据进行填充，以便后续窗口操作
                intervals = np.pad(intervals, (WINDOW_SIZE // 2 + 1, WINDOW_SIZE // 2))
                melody_piano_roll = np.pad(melody_piano_roll, ((WINDOW_SIZE // 2, WINDOW_SIZE // 2), (0, 0)))
                accomp_piano_roll = np.pad(accomp_piano_roll, ((WINDOW_SIZE // 2, WINDOW_SIZE // 2), (0, 0)))

                # 将处理好的数据添加到列表中
                group.append({
                    "intervals": intervals,
                    "melody_piano_roll": melody_piano_roll,
                    "accomp_piano_roll": accomp_piano_roll,
                    "melody_flags": melody_flags,
                    "melody_down_octaves": melody_down_octaves,
                    "melody_up_octaves": melody_up_octaves,
                    "accomp_shifts": accomp_shifts,
                    "window_gcd": window_gcd
                })

            # 跳过没有有效窗口的组
            if group:
                # 添加当前组的正负样本权重统计到全局统计列表中，计算当前组的平均正负样本权重比例，并添加到全局统计列表中
                global_pos_weights.append(sum(group_pos_weights) / len(group_pos_weights))
                self.data.append(group)

                # 从元数据中获取当前组的权重，如果没有则默认为 1.0，并添加到组权重列表中
                self.group_weights.append(metadata.get("weights", {}).get(group_dir.name, 1.0))

        # 统计最佳的正负样本权重比例
        self.pos_weight = torch.tensor(sum(
            group_pos_weight * group_weight
            for group_pos_weight, group_weight in zip(global_pos_weights, self.group_weights)
        ) / sum(group_pos_weights), dtype=torch.float32)

    def __iter__(self) -> Iterator[DatasetOutput]:
        while True:
            # 抽取一个组，然后从组中抽取一个文件，从文件中抽取一个中心位置
            # 然后随机选择一个主旋律的八度偏移（在允许范围内）
            while True:
                group = random.choices(self.data, self.group_weights, k=1)[0]
                file_data = random.choice(group)
                center_index = random.randint(0, len(file_data["melody_flags"]) - 1)
                # 仅当窗口混合有主旋律和伴奏时选取
                melody_up_range = file_data["melody_up_octaves"][center_index]
                if melody_up_range != -1:
                    melody_octave_shift = random.randint(0, melody_up_range)
                    break

            # 伴奏移位，使伴奏居中下
            accomp_piano_roll = np.roll(file_data["accomp_piano_roll"][center_index:center_index + WINDOW_SIZE], file_data["accomp_shifts"][center_index].item(), -1)

            # 计算实际的八度变换（相对值），并对主旋律进行移位
            melody_piano_roll = file_data["melody_piano_roll"][center_index:center_index + WINDOW_SIZE]
            transform_octave = melody_octave_shift - file_data["melody_down_octaves"][center_index].item()
            melody_piano_roll = np.roll(melody_piano_roll, transform_octave * 12, -1)

            # 合并主旋律和伴奏钢琴卷帘，并随机平移整个钢琴卷帘
            piano_roll = melody_piano_roll | accomp_piano_roll
            pitches_existed = np.where(piano_roll.sum(axis=0) > 0)[0]
            if random.random() > 0.5:
                piano_roll = np.roll(piano_roll, random.randint(0, 127 - pitches_existed.max()), -1)
            else:
                piano_roll = np.roll(piano_roll, random.randint(-pitches_existed.min(), 0), -1)

            # 随机缩放时间间隔
            intervals = file_data["intervals"][center_index:center_index + WINDOW_SIZE] / file_data["window_gcd"][center_index]
            intervals = intervals * random.randint(1, int(MAX_INTERVAL / intervals.max().item()))

            yield {
                "piano_roll": piano_roll,
                "interval": file_data["intervals"][center_index:center_index + WINDOW_SIZE],
                "melody_flag": file_data["melody_flags"][center_index],
            }


def collate_fn(batch: list[DatasetOutput]) -> dict:
    return {
        "piano_roll": torch.tensor(np.stack([item["piano_roll"] for item in batch], dtype=np.float32)),
        "interval": torch.tensor(np.stack([item["interval"] for item in batch], dtype=np.long)),
        "melody_flag": torch.tensor(np.stack([item["melody_flag"] for item in batch], dtype=np.float32)),
    }


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, gamma: float, pos_weight: float) -> torch.Tensor:
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
    predicted_probs = F.sigmoid(logits)
    classification_factor = predicted_probs * labels + (1 - predicted_probs) * (1 - labels)
    modulating_factor = (1 - classification_factor) ** gamma
    return modulating_factor * bce_loss


def binary_metrics(logits: list[torch.Tensor], labels: list[torch.Tensor], metric_prefix: str, global_step: int, writer: SummaryWriter) -> tuple[float, float]:
    """
    计算二分类任务的各项评估指标并记录到 TensorBoard

    Args:
        logits: 模型输出的未经过激活函数的预测值列表
        labels: 对应的真实标签列表（0 或 1）
        metric_prefix: 指标名称的前缀，用于 TensorBoard 中的分组显示
        global_step: 当前训练步数，用于 TensorBoard 的时间轴
        writer: TensorBoard 的 SummaryWriter 对象，用于记录指标

    Returns:
        返回一个二元组，包含: (损失值, AUC 值)

    Examples:
        >>> logits = [torch.tensor([0.8]), torch.tensor([-0.2])]
        >>> labels = [torch.tensor([1.0]), torch.tensor([0.0])]
        >>> loss, auc = binary_metrics(logits, labels, "Train", 100, writer)
    """
    # 使用推理模式，关闭梯度计算以提高效率
    with torch.inference_mode():
        # 将列表中的张量堆叠成单个张量以便批量计算
        logits = torch.stack(logits)
        labels = torch.stack(labels)

        # 计算二元交叉熵损失（使用 sigmoid + BCELoss 的数值稳定版本）
        loss = F.binary_cross_entropy_with_logits(logits, labels).item()

        # 将 logits 转换为概率值
        probabilities = F.sigmoid(logits).cpu().numpy()

        # 将真实标签转换为二值
        binary_labels = (labels > 0.5).cpu().numpy()

    # 计算 ROC 曲线和 AUC 值
    roc_auc = roc_auc_score(binary_labels, probabilities)

    # 将指标记录到 TensorBoard 中，使用 metric_prefix 进行分组
    writer.add_scalar(f"Loss/{metric_prefix}", loss, global_step)
    writer.add_scalar(f"AUC/{metric_prefix}", roc_auc, global_step)
    add_pr_curve(writer, f"PR Curve/{metric_prefix}", binary_labels, probabilities, global_step)
    add_roc_curve(writer, f"ROC Curve/{metric_prefix}", binary_labels, probabilities, global_step)
    return loss, roc_auc


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练主旋律提取模型")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="模型检查点保存路径")
    parser.add_argument("train_dir", type=pathlib.Path, help="训练集 MIDI 文件目录")
    parser.add_argument("val_dir", type=pathlib.Path, help="验证集 MIDI 文件目录")
    parser.add_argument("-l", "--log-dir", type=pathlib.Path, default=None, help="TensorBoard 日志保存目录，不指定则不保存日志")
    parser.add_argument("-n", "--num-validations", type=int, default=16, help="训练过程中验证的总次数，总训练步数 = 本参数 * 每训练多少步进行一次验证，默认为 %(default)s")
    parser.add_argument("-vi", "--validation-interval", default=1024, type=int, help="每训练多少步进行一次验证，默认为 %(default)s")
    parser.add_argument("-d", "--dropout", type=float, default=DEFAULT_DROPOUT, help="模型 dropout 比例，取值范围 [0, 1)，默认值为 %(default)s")
    parser.add_argument("-b", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="训练批次大小，验证集批次大小是这个值的两倍，默认值为 %(default)s")
    parser.add_argument("-vb", "--val-batches", default=16, type=int, help="每次验证运行的批次数量，总样本数 = 验证批次大小 * 本参数。默认为 %(default)s")
    parser.add_argument("-g", "--loss-gamma", default=2.0, type=float, help="焦点损失的 gamma 参数，默认为 %(default)s")
    parser.add_argument("-p", "--progress-bar", action="store_true", help="显示进度条，默认不显示")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    print("读取并加载检查点 ...")
    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=True)
    model = Model(dropout=args.dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print("准备数据集 ...")
    train_dataset = DatasetWithAugmentation(args.train_dir)
    val_dataset = DatasetWithoutAugmentation(args.val_dir)
    train_loader = iter(DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn))
    val_loader = iter(DataLoader(val_dataset, batch_size=args.batch_size * 2, collate_fn=collate_fn))

    print("开始训练 ...")
    progress_bar = tqdm(total=args.num_validations * args.validation_interval, disable=not args.progress_bar)
    writer = Placeholder() if args.log_dir is None else SummaryWriter(args.log_dir)
    for val_cycle_idx in range(1, args.num_validations + 1):
        all_outputs, all_labels = [], []
        model.train()
        for _ in range(args.validation_interval):
            data = {k: v.to(device) for k, v in next(train_loader).items()}
            output = model(data)
            loss = focal_loss(output["logit"], data["melody_flag"], gamma=args.loss_gamma, pos_weight=train_dataset.pos_weight)
            valid_mask = ~loss.isnan()
            loss.masked_select(valid_mask).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            progress_bar.update()
            all_outputs.extend(output["logit"][valid_mask])
            all_labels.extend(data["melody_flag"][valid_mask])
        loss, roc_auc = binary_metrics(all_outputs, all_labels, "Train", checkpoint["completed_steps"] + int((val_cycle_idx - 0.5) * args.validation_interval), writer)
        progress_bar.set_postfix(loss=loss, auc=roc_auc)

        with torch.inference_mode():
            all_outputs, all_labels = [], []
            model.eval()
            for _ in range(args.val_batches):
                data = {k: v.to(device) for k, v in next(val_loader).items()}
                output = model(data)
                loss = F.binary_cross_entropy_with_logits(output["logit"], data["melody_flag"], reduction="none")
                valid_mask = ~loss.isnan()
                all_outputs.extend(output["logit"][valid_mask])
                all_labels.extend(data["melody_flag"][valid_mask])
        binary_metrics(all_outputs, all_labels, "Validate", checkpoint["completed_steps"] + int(val_cycle_idx * args.validation_interval), writer)
    
    progress_bar.close()
    writer.close()

    print("保存检查点 ...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "completed_steps": checkpoint["completed_steps"] + args.num_validations * args.validation_interval
    }, args.ckpt_path)


if __name__ == "__main__":
    main(parse_args())
