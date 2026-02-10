# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 thiliapr <thiliapr@tutanota.com>
# SPDX-Package: melody_extract
# SPDX-PackageHomePage: https://github.com/thiliapr/melody_extract

from typing import Optional
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.plugins.pr_curve.plugin_data_pb2 import PrCurvePluginData


def add_pr_curve_raw(
    writer: SummaryWriter,
    tag: str,
    true_positive_counts: np.ndarray,
    false_positive_counts: np.ndarray,
    true_negative_counts: np.ndarray,
    false_negative_counts: np.ndarray,
    precision: np.ndarray,
    recall: np.ndarray,
    global_step: Optional[int] = None,
    num_thresholds: int = 2048,
    walltime: Optional[float] = None
):
    """
    本函数删除了 PyTorch 截至 2026-01-25 都没有删除的限制（num_thresholds 不能大于 127）。
    经过我的测试，即使 num_thresholds = 2048 也不会有任何问题，但 PyTorch 源码却标注"weird, value > 127 breaks protobuf"。
    这的确很 weird。

    Source: https://github.com/pytorch/pytorch/blob/0cd681d12e6879f242edb3bf3c1810bf41bb69c1/torch/utils/tensorboard/summary.py#L811
    Issue: https://github.com/pytorch/pytorch/issues/173311
    """
    data = np.stack((true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts, precision, recall))
    pr_curve_plugin_data = PrCurvePluginData(version=0, num_thresholds=num_thresholds).SerializeToString()
    plugin_data = SummaryMetadata.PluginData(plugin_name="pr_curves", content=pr_curve_plugin_data)
    smd = SummaryMetadata(plugin_data=plugin_data)
    tensor = TensorProto(
        dtype="DT_FLOAT",
        float_val=data.reshape(-1).tolist(),
        tensor_shape=TensorShapeProto(
            dim=[
                TensorShapeProto.Dim(size=data.shape[0]),
                TensorShapeProto.Dim(size=data.shape[1]),
            ]
        ),
    )
    summary = Summary(value=[Summary.Value(tag=tag, metadata=smd, tensor=tensor)])
    writer._get_file_writer().add_summary(summary, global_step, walltime)


def add_pr_curve(writer: SummaryWriter, tag: str, labels: np.ndarray, predictions: np.ndarray, global_step: Optional[int] = None, num_thresholds: int = 2048, walltime: Optional[float] = None):
    # 计算精确率-召回率曲线
    thresholds = np.linspace(0, 1, num_thresholds)
    labels = labels.astype(bool)
    pred_labels = predictions[None, :] >= thresholds[:, None]  # [num_thresholds, len(predictions)]
    tp = np.sum(pred_labels & labels, axis=1)
    fp = np.sum(pred_labels & ~labels, axis=1)
    tn = np.sum(~pred_labels & ~labels, axis=1)
    fn = np.sum(~pred_labels & labels, axis=1)
    precision = tp / np.maximum(1e-7, tp + fp)
    recall = tp / np.maximum(1e-7, tp + fn)

    # 绘制曲线
    add_pr_curve_raw(
        writer, tag, tp, fp, tn, fn, precision, recall,
        global_step=global_step,
        num_thresholds=num_thresholds,
        walltime=walltime
    )


def add_roc_curve(writer: SummaryWriter, tag: str, labels: np.ndarray, predictions: np.ndarray, global_step: Optional[int] = None, num_thresholds: int = 2048, walltime: Optional[float] = None):
    # 计算 ROC 曲线
    thresholds = np.linspace(0, 1, num_thresholds)
    labels = labels.astype(bool)
    pred_labels = predictions[None, :] >= thresholds[:, None]  # [num_thresholds, len(predictions)]
    tp = np.sum(pred_labels & labels, axis=1)
    fp = np.sum(pred_labels & ~labels, axis=1)
    tn = np.sum(~pred_labels & ~labels, axis=1)
    fn = np.sum(~pred_labels & labels, axis=1)
    tpr = tp / np.maximum(1e-7, tp + fn)
    fpr = fp / np.maximum(1e-7, fp + tn)

    # 绘制曲线，这里复用 pr_curve
    add_pr_curve_raw(
        writer, tag, tp, fp, tn, fn, tpr, fpr,
        global_step=global_step,
        num_thresholds=num_thresholds,
        walltime=walltime
    )
