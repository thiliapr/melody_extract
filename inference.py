# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025-2026 thiliapr <thiliapr@tutanota.com>
# SPDX-Package: melody_extract
# SPDX-PackageHomePage: https://github.com/thiliapr/melody_extract

import argparse
import pathlib
from typing import Optional
import mido
import torch
import numpy as np
from tqdm import tqdm
from utils.midi import TIME_PRECISION, midi_to_tracks, tracks_to_piano_roll
from utils.model import Model, WINDOW_SIZE, MAX_INTERVAL


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    解析命令行参数。

    Args:
        args: 命令行参数列表。如果为 None，则使用 sys.argv。

    Returns:
        包含解析后的参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description="使用训练好的模型对 MIDI 文件进行旋律提取")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="检查点文件路径")
    parser.add_argument("input_path", type=pathlib.Path, help="输入 MIDI 文件路径")
    parser.add_argument("output_path", type=pathlib.Path, help="输出 MIDI 文件路径")
    parser.add_argument("-b", "--batch-size", type=int, default=512, help="批处理大小，默认为 %(default)s")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="旋律音符概率阈值，默认为 %(default)s")
    return parser.parse_args(args)


@torch.inference_mode()
def main(args: argparse.Namespace):
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载检查点
    print("加载检查点 ...")
    ckpt = torch.load(args.ckpt_path, map_location=device)

    # 加载模型状态，并设置为评估模式
    model = Model()
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.eval()

    # 加载输入 MIDI 文件
    print("加载输入 MIDI 文件 ...")
    notes = midi_to_tracks(mido.MidiFile(args.input_path, clip=True))
    piano_roll, intervals = tracks_to_piano_roll(notes)  # [时间步数, 128 个音高]

    # 填充钢琴卷帘和时间，确保覆盖边界情况
    piano_roll = np.pad(piano_roll, ((WINDOW_SIZE // 2, WINDOW_SIZE // 2), (0, 0)))
    intervals = np.pad(np.clip(intervals, 0, MAX_INTERVAL), (WINDOW_SIZE // 2 + 1, WINDOW_SIZE // 2))  # 约束时间间隔不超过学习到的嵌入

    # 使用滑动窗口组织输入数据
    segments = []
    for start_idx in range(0, len(piano_roll) - WINDOW_SIZE + 1):
        segments.append((piano_roll[start_idx:start_idx + WINDOW_SIZE], intervals[start_idx:start_idx + WINDOW_SIZE]))
    print(f"总共有 {len(segments)} 个段落需要处理")

    # 进行模型推理
    print("进行推理 ...")
    probabilities = torch.empty(len(piano_roll) - WINDOW_SIZE + 1, dtype=torch.float32, device=device)  # [time_steps]

    # 分批处理所有音频段
    for batch_start in tqdm(range(0, len(segments), args.batch_size)):
        piano_roll, interval = [torch.from_numpy(np.stack(x)) for x in zip(*segments[batch_start:batch_start + args.batch_size])]

        # 模型预测并应用 sigmoid 得到概率
        probabilities[batch_start:batch_start + args.batch_size] = model({
            "piano_roll": piano_roll.float(),
            "interval": interval
        })["logits"].sigmoid()  # [batch_size]

    # 合并各轨道的音符
    all_notes = {
        (pitch, time)
        for track_notes in notes
        for pitch, time in track_notes
    }

    # 按时间步组织音符
    notes_by_time = {}
    for pitch, time in all_notes:
        notes_by_time.setdefault(time, []).append(pitch)

    # 按时间排序
    notes_by_time = sorted(notes_by_time.items(), key=lambda x: x[0])

    # 根据概率阈值提取旋律音符
    melody_notes = []
    accompaniment_notes = []
    for time_idx in range(len(probabilities)):
        time, pitches = notes_by_time[time_idx]
        if probabilities[time_idx] >= args.threshold:
            # 选择该时间步音高最高的音符作为旋律音符
            melody_pitch = max(pitches)
            melody_notes.append((time, melody_pitch))
            # 剩余音符作为伴奏
            pitches.remove(melody_pitch)
        accompaniment_notes.extend((time, pitch) for pitch in pitches)

    print(f"提取出 {len(melody_notes)} 个旋律音符，剩余 {len(accompaniment_notes)} 个伴奏音符")
    print(f"{len(segments)} 个时间步中有 {(probabilities >= args.threshold).sum().item()} 个时间步包含旋律音符")

    # 生成 MIDI 轨道
    melody_track = mido.MidiTrack([mido.MetaMessage("track_name", name="Melody", time=0)])
    other_track = mido.MidiTrack([mido.MetaMessage("track_name", name="Accompaniment", time=0)])

    for track, track_notes in [(melody_track, melody_notes), (other_track, accompaniment_notes)]:
        # 设置轨道参数
        track.append(mido.Message("program_change", program=0, time=0))
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(128), time=0))

        # 添加音符事件（按下和释放）
        events = []
        for time, pitch in sorted(track_notes, key=lambda x: x[0]):
            events.append((time * TIME_PRECISION, "note_on", pitch))
            events.append(((time + 1) * TIME_PRECISION, "note_off", pitch))
        events.sort(key=lambda x: (x[0]))

        # 转换为 MIDI 事件
        last_event_time = 0
        for time, event_type, pitch in events:
            delta_time = time - last_event_time
            last_event_time = time

            if event_type == "note_on":
                track.append(mido.Message("note_on", note=pitch, velocity=100, time=delta_time))
            else:
                track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta_time))

        # 添加轨道结束标记
        track.append(mido.MetaMessage("end_of_track", time=1))

    # 保存输出 MIDI 文件
    print("保存输出 MIDI 文件 ...")
    mido.MidiFile(tracks=[mido.MidiTrack([
        mido.MetaMessage("track_name", name="Melody Extraction Result", time=0),
        mido.MetaMessage("end_of_track", time=0)
    ]), melody_track, other_track]).save(args.output_path)

    print("旋律提取完成！")


if __name__ == "__main__":
    main(parse_args())