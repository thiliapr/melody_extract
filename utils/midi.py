# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2024-2026 thiliapr <thiliapr@tutanota.com>
# SPDX-Package: melody_extract
# SPDX-PackageHomePage: https://github.com/thiliapr/melody_extract

import mido
import numpy as np

TIME_PRECISION = 120


def midi_to_tracks(midi_file: mido.MidiFile) -> list[list[tuple[int, int]]]:
    all_track_notes: list[list[tuple[int, int]]] = []  # 每个轨道的音符信息
    drum_channels = {9}  # 记录打击乐通道

    for track in midi_file.tracks:
        current_time = 0  # 当前的绝对时间
        current_track_notes = []  # 当前轨道的音符信息
        for msg in track:
            # 计算绝对时间
            current_time += msg.time * 480 // midi_file.ticks_per_beat

            # 处理音色变化事件，动态更新打击乐通道
            if msg.type == "program_change":
                if msg.channel == 9:
                    continue  # 通道 9 固定为打击乐，跳过处理

                # 根据 GM 标准，特定音色范围属于打击乐器
                # 打击乐音色范围: 96-103 (Sound Effects) 或 112-127 (Percussive)
                if (96 <= msg.program <= 103) or msg.program >= 112:
                    drum_channels.add(msg.channel)
                else:
                    drum_channels.discard(msg.channel)

            # 处理音符开始事件，排除打击乐通道
            elif msg.type == "note_on" and msg.velocity != 0 and msg.channel not in drum_channels:
                current_track_notes.append((msg.note, current_time))

        # 如果当前轨道有音符，则添加到轨道列表中
        if current_track_notes:
            all_track_notes.append(current_track_notes)

    # 计算所有音符中的最早时间点，用于时间轴对齐
    earliest_note_time = min(time for track_notes in all_track_notes for _, time in track_notes)

    # 对所有轨道的音符时间进行标准化处理
    all_track_notes = [
        [
            (pitch, int((time - earliest_note_time) / TIME_PRECISION + 0.5))
            for pitch, time in track_notes
        ]
        for track_notes in all_track_notes
    ]

    return all_track_notes


def tracks_to_piano_roll(tracks: list[list[tuple[int, int]]]) -> tuple[np.ndarray, np.ndarray]:
    # 合并各轨道音符
    notes = [note for track in tracks for note in track]

    # 获取音符出现的时间
    times = sorted({time for _, time in notes})
    time_to_index = {time: i for i, time in enumerate(times)}

    # 生成钢琴卷帘
    piano_roll = np.zeros((len(times), 128), dtype=bool)
    for pitch, time in notes:
        piano_roll[time_to_index[time], pitch] = True

    # 生成时间间隔
    times = np.array(times)
    intervals = times[1:] - times[:-1]

    return piano_roll, intervals