# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2024-2026 thiliapr <thiliapr@tutanota.com>
# SPDX-Package: melody_extract
# SPDX-PackageHomePage: https://github.com/thiliapr/melody_extract

import argparse
import pathlib
import random
from typing import Optional
import torch
import numpy as np
from torch import optim
from utils.model import Model

DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-2


def set_seed(seed: int):
    """
    设置所有随机源的种子以确保实验可复现性。

    工作流程:
    1. 设置 Python 内置 random 模块的种子
    2. 设置 NumPy 的随机种子
    3. 设置 PyTorch 的 CPU 和 GPU 随机种子
    4. 配置 CuDNN 使用确定性算法并关闭 benchmark 模式

    Args:
        seed: 要设置的随机种子值

    Examples:
        >>> set_seed(8964)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多 GPU 情况
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    解析命令行参数。

    Args:
        args: 命令行参数列表。如果为 None，则使用 sys.argv。

    Returns:
        包含解析后的参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description="初始化一个检查点")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="检查点保存路径，建议以 .ckpt 结尾")
    parser.add_argument("-lr", "--learning-rate", default=DEFAULT_LEARNING_RATE, type=float, help="优化器学习率，默认为 %(default)s")
    parser.add_argument("-wd", "--weight-decay", default=DEFAULT_WEIGHT_DECAY, type=float, help="优化器权重衰减系数，默认为 %(default)s")
    parser.add_argument("-u", "--seed", default=19890604, type=int, help="初始化检查点的种子，保证训练过程可复现，默认为 %(default)s")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 设置随机种子，确保可复现性
    set_seed(args.seed)

    # 初始化模型
    model = Model()

    # 初始化优化器和梯度缩放器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 保存为检查点
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "completed_steps": 0
    }, args.ckpt_path)

    # 打印初始化成功信息
    print(f"检查点初始化成功，已保存到 {args.ckpt_path}")


if __name__ == "__main__":
    main(parse_args())