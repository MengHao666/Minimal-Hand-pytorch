import sys
import pickle
import torch
import os
from torch.utils import data
from termcolor import colored, cprint
import numpy as np

sik1m_inst = 0


class _SIK1M(data.Dataset):
    """
    The Loader for joints so3 and quat
    """

    def __init__(
            self,
            data_root="data",
            data_source=None
    ):
        print("Initialize _SIK1M instance")
        bone_len_path = os.path.join(data_root, 'data_bone.npy')
        shape_path = os.path.join(data_root, 'data_shape.npy')
        self.bone_len = np.load(bone_len_path)
        self.shape = np.load(shape_path)

    def __len__(self):
        return self.shape.shape[0]

    def __getitem__(self, index):
        temp_bone_len = self.bone_len[index]
        temp_shape = self.shape[index]

        metas = {
            'rel_bone_len': temp_bone_len,
            'shape': temp_shape
        }
        return metas


class SIK1M(data.Dataset):
    def __init__(
            self,
            data_split="train",
            data_root="data",
            split_ratio=0.8
    ):
        global sik1m_inst
        if not sik1m_inst:
            sik1m_inst = _SIK1M(data_root=data_root)
        self.sik1m = sik1m_inst
        self.permu = list(range(len(self.sik1m)))
        self.alllen = len(self.sik1m)
        self.data_split = data_split
        # add the 0.1* the std of Relative bone length as noise, you can change it or not add
        self.noise = np.array([0.02906406, 0.02663224, 0.01769793, 0.0274501, 0.02573783, 0.0222863,
                               0., 0.02855567, 0.02330295, 0.0253288, 0.0266308, 0.02495683, 0.03685857, 0.02430637,
                               0.02349446])
        self.noise = self.noise / 10.0
        if data_split == "train":
            self.vislen = int(len(self.sik1m) * split_ratio)
            self.sub_permu = self.permu[:self.vislen]
        elif data_split in ["val", "test"]:
            self.vislen = self.alllen - int(len(self.sik1m) * split_ratio)
            self.sub_permu = self.permu[(self.alllen - self.vislen):]
        else:
            self.vislen = len(self.sik1m)
            self.sub_permu = self.permu[:self.vislen]

    def __len__(self):
        return self.vislen

    def __getitem__(self, index):
        item = self.sik1m[self.sub_permu[index]]
        temp = np.random.randn(15, )
        temp = np.multiply(self.noise, temp)
        item['rel_bone_len'] += temp
        return item


def main():
    sik1m_train = SIK1M(
        data_split="train",
        data_root="data"
    )
    sik1m_test = SIK1M(
        data_split="test"
    )

    metas = sik1m_train[2]
    print(metas)
    metas = sik1m_train[2]
    print(metas)


if __name__ == "__main__":
    main()
