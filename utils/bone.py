import config as cfg
import numpy as np
import torch


def caculate_length(j3d_, label=None):
    if isinstance(j3d_, torch.Tensor):
        j3d = j3d_.clone()
        j3d = j3d.detach().cpu()
        j3d = j3d.numpy()
    else:
        j3d = j3d_.copy()

    if len(j3d.shape) != 2:
        j3d = j3d.squeeze()

    bone = [
        j3d[i] - j3d[cfg.SNAP_PARENT[i]]
        for i in range(21)
    ]
    bone_len = np.linalg.norm(
        bone, ord=2, axis=-1, keepdims=True  # 21*1
    )

    if label == "full":
        return bone_len
    elif label == "useful":
        return bone_len[cfg.USEFUL_BONE]
    else:
        raise ValueError("{} not in ['full'|'useful']".format(label))
