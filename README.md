# thiliapr/melody_extract
这是一个用于从多轨 MIDI 中提取主旋律音符的 PyTorch 模型仓库。

## License
![GNU AGPL Version 3 Logo](https://www.gnu.org/graphics/agplv3-with-text-162x68.png)

thiliapr/melody_extract 是自由软件，遵循[Affero GNU 通用公共许可证第 3 版或任何后续版本](https://www.gnu.org/licenses/agpl-3.0.html)。你可以自由地使用、修改和分发该软件，但不提供任何明示或暗示的担保。

## 主要脚本
- [init_checkpoint.py](init_checkpoint.py): 初始化检查点
- [train.py](train.py): 训练主旋律提取模型
- [evaluate.py](evaluate.py): 评估模型在测试集上的性能
- [inference.py](inference.py): 使用训练好的模型对 MIDI 文件进行旋律提取

## 快速开始
### 你需要什么
- 一个带 [NVIDIA GPU](https://www.nvidia.com/en-us/products/workstations/) 的电脑，或者……保持耐心
- 两个包含 MIDI 文件的文件夹，一个作为训练集，一个作为验证集，其中的 MIDI 文件的要求
  - 第一个有音符的轨道包含所有主旋律音符，且只包含主旋律音符（学术一点说，召回率和精确率必须同时达到 100%）
  - 训练集的 MIDI 文件必须有至少两个有音符的轨道（数据增强要求），验证集可忽略该规定
  - 第一个轨道 ≠ 第一个有音符的轨道，比如若干个轨道可以只包含 MetaMessage，指定轨道名（可能是作者名、曲名之类），数据提取时会忽略这些无音符轨道
  - 牢记[垃圾进，垃圾出](https://www.ebsco.com/research-starters/computer-science/garbage-garbage-out-gigo)这个机器学习普世真理

### 训练集目录结构应该长什么样
#### 结构示意图
```plaintext
trainset/
├── touhou/
│   ├── Touhou 7 PCB.mid
│   ├── Touhou 19 UDoALG.mid
├── fix/
│   ├── A File that Tests Badly.mid
└── metadata.json
```

### `metadata.json` 示例
```json
{
  "weights": {
    "touhou": 1.989,
    "fix": 0.604,
    "组的目录名": 0.721
  }
}
```

#### 解释
- 你得把文件分组，所有 MIDI 文件必须在子目录下，而不是直接在根目录下
- `metadata.json`: 必须在根目录下。通过此文件，你可以配置每个组的权重，这会影响他们在训练时被抽取的概率
- 只要求对训练集进行此操作，验证集你随意，只要包含 MIDI 文件就行

### 安装依赖
PyTorch 需要根据你的操作系统与硬件手动安装，请访问[官方安装页面](https://pytorch.org/)获取适合您系统的安装命令。  
安装 PyTorch 完成后，你得安装 `requirements.txt` 的依赖。  
我强烈推荐你使用 [venv](https://docs.python.org/3/library/venv.html) 来创建虚拟环境并安装，除非你打算在这个系统只运行这个程序。

```bash
# （可选）创建虚拟环境并激活
python -m venv .venv
./.venv/Scripts/activate

# 安装依赖。首先手动按 https://pytorch.org/ 的指引安装 PyTorch，然后安装其余 Python 依赖
# 注意: 不要真的按照脚本的 PyTorch 安装命令去安装，除非你真的只有 CPU
pip install torch
pip install -r requirements.txt
```

### 使用示例
```bash
# 检查点初始化、训练模型
python init_checkpoint.py ckpt.ckpt
python train.py ckpt.ckpt /path/to/train_midi_dir /path/to/val_midi_dir -p -l logdir
# 打开 tensorboard，找到 PR Curve/Validate，找到你想要的精确率-召回率阈值
tensorboard --logdir logdir
python inference.py ckpt.ckpt input.mid output.mid -p -t 0.5
```

## 评估模型
### 什么时候可能需要评估模型
- 你从别人手中获得了一个检查点，但是你不知道它的表现如何
- 你想改进你的模型，想要找出模型性能表现得最差的几个文件，加进训练集里
- 你想利用一个已经充分恰当训练的模型，找出测试集分类可能错误的文件（比如模型在特定文件的预测和给定标签偏差很远，但是在大部分文件表现良好，那么可能是这个特定文件有问题）
- 还有其他情况自己想想

### 评估结果包含什么
- 一些文本（`Evaluate/Worst by Loss`），包含本次评估损失最高的五个文件，以及模型评估它们的损失和 AUC 分数
- 两个文本（`Evaluate/Overall Loss`和`Evaluate/Overall AUC`），分别是总体评估的损失和 AUC 分数
- 两个图像（`Evaluate/Overall ROC Curve`和`Evaluate/Overall PR Curve`），分别是总体评估的 ROC 曲线和 PR 曲线，你可以根据他们选择合适的阈值

### 使用示例
```bash
python evaluate.py ckpt.ckpt /path/to/test_midi_dir logdir
tensorboard --logdir logdir
```
