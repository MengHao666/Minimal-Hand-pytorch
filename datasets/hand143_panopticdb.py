# Copyright (c) Hao Meng. All Rights Reserved.
r"""
Hands from Panoptic Studio by Multiview Bootstrapping (14817 annotations)
Hand Keypoint Detection in Single Images using Multiview Bootstrapping, CVPR 2017
Link to dataset: http://domedb.perception.cs.cmu.edu/handdb.html
Download:http://domedb.perception.cs.cmu.edu/panopticDB/hands/hand143_panopticdb.tar
"""

import json
import os
import pickle

import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image
from termcolor import colored
from tqdm import tqdm

import config as cfg
import utils.handutils as handutils

CACHE_HOME = os.path.expanduser(cfg.DEFAULT_CACHE_DIR)

snap_joint_name2id = {w: i for i, w in enumerate(cfg.snap_joint_names)}


class Hand143_panopticdb(torch.utils.data.Dataset):
    def __init__(
            self,
            data_root="/home/chen/datasets/CMU/hand143_panopticdb",
            data_split='train',
            hand_side='right',
            njoints=21,
            use_cache=True,
            vis=False
    ):

        if not os.path.exists(data_root):
            raise ValueError("data_root: %s not exist" % data_root)

        self.name = 'hand143_panopticdb'
        self.data_split = data_split
        self.hand_side = hand_side
        self.clr_paths = []
        self.kp2ds = []
        self.centers = []
        self.my_scales = []
        self.njoints = njoints
        self.reslu = [1920, 1080]
        self.vis = vis

        self.root_id = snap_joint_name2id['loc_bn_palm_L']  # 0
        self.mid_mcp_id = snap_joint_name2id['loc_bn_mid_L_01']  # 9

        # [train|test|val|train_val|all]
        if data_split == 'train':
            self.sequence = ['training', ]
        else:
            print("hand143_panopticdb only has train_set!")
            return

        self.cache_folder = os.path.join(CACHE_HOME, "my-train", "hand143_panopticdb")
        os.makedirs(self.cache_folder, exist_ok=True)
        cache_path = os.path.join(
            self.cache_folder, "{}.pkl".format(self.data_split)
        )

        if os.path.exists(cache_path) and use_cache:
            with open(cache_path, "rb") as fid:
                annotations = pickle.load(fid)
                self.clr_paths = annotations["clr_paths"]
                self.kp2ds = annotations["kp2ds"]
                self.centers = annotations["centers"]
                self.my_scales = annotations["my_scales"]
            print("hand143_panopticdb {} gt loaded from {}".format(self.data_split, cache_path))
            return

        self.clr_root_list = [
            os.path.join(data_root, "imgs")
        ]

        self.ann_list = [
            os.path.join(
                data_root,
                "hands_v143_14817.json"
            )
        ]

        for clr_root, ann in zip(self.clr_root_list, self.ann_list):

            jsonPath = os.path.join(ann)
            with open(jsonPath, 'r') as fid:
                dat_all = json.load(fid)
                dat_all = dat_all['root']

            for i in tqdm(range(len(dat_all))):
                clrpth = os.path.join(clr_root, '%.8d.jpg' % i)
                self.clr_paths.append(clrpth)

                dat = dat_all[i]
                kp2d = np.array(dat['joint_self'])[:, : 2]  # kp 2d left & right hand
                center = handutils.get_annot_center(kp2d)
                my_scale = handutils.get_ori_crop_scale(mask=None, side=None, mask_flag=False, kp2d=kp2d)

                kp2d = kp2d[np.newaxis, :, :]
                self.kp2ds.append(kp2d)

                center = center[np.newaxis, :]
                self.centers.append(center)

                my_scale = (np.atleast_1d(my_scale))[np.newaxis, :]
                self.my_scales.append(my_scale)

        self.kp2ds = np.concatenate(self.kp2ds, axis=0).astype(np.float32)  # (N, 21, 2)
        self.centers = np.concatenate(self.centers, axis=0).astype(np.float32)  # (N, 1)
        self.my_scales = np.concatenate(self.my_scales, axis=0).astype(np.float32)  # (N, 1)

        if use_cache:
            full_info = {
                "clr_paths": self.clr_paths,
                "kp2ds": self.kp2ds,
                "centers": self.centers,
                "my_scales": self.my_scales,
            }
            with open(cache_path, "wb") as fid:
                pickle.dump(full_info, fid)
                print("Wrote cache for dataset hand143_panopticdb {} to {}".format(
                    self.data_split, cache_path
                ))
        return

    def _is_valid(self, clr, index):
        valid_data = isinstance(clr, (np.ndarray, PIL.Image.Image))

        if not valid_data:
            raise Exception("Encountered error processing cmu_1_[{}]".format(index))
        return valid_data

    def __len__(self):
        return len(self.clr_paths)

    def __str__(self):
        info = "Hand143_panopticdb {} set. lenth {}".format(
            self.data_split, len(self.clr_paths)
        )
        return colored(info, 'blue', attrs=['bold'])

    def get_sample(self, index):
        flip = True if self.hand_side != 'right' else False

        # prepare color image
        clr = Image.open(self.clr_paths[index]).convert("RGB")
        self._is_valid(clr, index)

        # prepare kp2d
        kp2d = self.kp2ds[index].copy()
        center = self.centers[index].copy()
        my_scale = self.my_scales[index].copy()
        if flip:
            clr = clr.transpose(Image.FLIP_LEFT_RIGHT)
            center[0] = clr.size[0] - center[0]
            kp2d[:, 0] = clr.size[0] - kp2d[:, 0]

        sample = {
            'index': index,
            'clr': clr,
            'kp2d': kp2d,
            'center': center,
            'my_scale': my_scale,
        }

        # visualization
        if self.vis:
            plt.figure(figsize=(20, 20))
            clr_ = np.array(clr)

            plt.subplot(1, 2, 1)
            clr1 = clr_.copy()
            plt.imshow(clr1)
            plt.title('color image')

            plt.subplot(1, 2, 2)
            clr2 = clr_.copy()
            plt.imshow(clr2)
            plt.plot(200, 100, 'r.', linewidth=10)  # opencv convention
            for p in range(kp2d.shape[0]):
                plt.plot(kp2d[p][0], kp2d[p][1], 'r.')
                plt.text(kp2d[p][0], kp2d[p][1], '{0}'.format(p), fontsize=5)
            plt.title('2D annotations')

            plt.show()

        return sample


def main():
    hand143_panopticdb = Hand143_panopticdb(
        data_root="/home/chen/datasets/CMU/hand143_panopticdb",
        data_split='train',
        hand_side='right',
        njoints=21,
        use_cache=True,
        vis=True
    )
    print("len(hand143_panopticdb)=", len(hand143_panopticdb))

    for i in tqdm(range(len(hand143_panopticdb))):
        print("i=", i)
        hand143_panopticdb.get_sample(i)


if __name__ == "__main__":
    main()
