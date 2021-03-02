# Copyright (c) Hao Meng. All Rights Reserved.
r"""
Hands with Manual Keypoint Annotations (Training: 1912 annotations, Testing: 846 annotations)
Hand Keypoint Detection in Single Images using Multiview Bootstrapping, CVPR 2017
Link to dataset: http://domedb.perception.cs.cmu.edu/handdb.html
Download:http://domedb.perception.cs.cmu.edu/panopticDB/hands/hand_labels.zip
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


class Hand_labels(torch.utils.data.Dataset):
    def __init__(
            self,
            data_root="/home/chen/datasets/CMU/hand_labels",
            data_split='train',
            hand_side='right',
            njoints=21,
            use_cache=True,
            vis=False
    ):

        if not os.path.exists(data_root):
            raise ValueError("data_root: %s not exist" % data_root)

        self.vis = vis
        self.name = 'CMU:hand_labels'
        self.data_split = data_split
        self.hand_side = hand_side
        self.clr_paths = []
        self.kp2ds = []
        self.centers = []
        self.sides = []
        self.my_scales = []
        self.njoints = njoints
        self.reslu = [1920, 1080]

        self.root_id = snap_joint_name2id['loc_bn_palm_L']  # 0
        self.mid_mcp_id = snap_joint_name2id['loc_bn_mid_L_01']  # 9

        # [train|test|val|train_val|all]
        if data_split == 'train':
            self.sequence = ['manual_train', ]
        elif data_split == 'test':
            self.sequence = ['manual_test', ]
        elif data_split == 'val':
            self.sequence = ['manual_test', ]
        elif data_split == 'train_val':
            self.sequence = ['manual_train', ]
        elif data_split == 'all':
            self.sequence = ['manual_train', 'manual_test']
        else:
            raise ValueError("hand_labels only has train_set!")

        self.cache_folder = os.path.join(CACHE_HOME, "my-train", "hand_labels")
        os.makedirs(self.cache_folder, exist_ok=True)
        cache_path = os.path.join(
            self.cache_folder, "{}.pkl".format(self.data_split)
        )

        if os.path.exists(cache_path) and use_cache:
            with open(cache_path, "rb") as fid:
                annotations = pickle.load(fid)
                self.sides = annotations["sides"]
                self.clr_paths = annotations["clr_paths"]
                self.kp2ds = annotations["kp2ds"]
                self.centers = annotations["centers"]
                self.my_scales = annotations["my_scales"]
            print("hand_labels {} gt loaded from {}".format(self.data_split, cache_path))
            return

        datapath_list = [
            os.path.join(data_root, seq) for seq in self.sequence
        ]

        for datapath in datapath_list:
            files = sorted([f for f in os.listdir(datapath) if f.endswith('.json')])

            for idx in tqdm(range(len(files))):
                f = files[idx]
                with open(os.path.join(datapath, f), 'r') as fid:
                    dat = json.load(fid)

                kp2d = np.array(dat['hand_pts'])[:, : 2]
                is_left = dat['is_left']
                self.sides.append("left" if is_left else "right")

                clr_pth = os.path.join(datapath, f[0:-5] + '.jpg')
                self.clr_paths.append(clr_pth)
                center = handutils.get_annot_center(kp2d)
                my_scale = handutils.get_ori_crop_scale(mask=False, mask_flag=False, side=None, kp2d=kp2d)

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
                "sides": self.sides,
                "clr_paths": self.clr_paths,
                "kp2ds": self.kp2ds,
                "centers": self.centers,
                "my_scales": self.my_scales,
            }
            with open(cache_path, "wb") as fid:
                pickle.dump(full_info, fid)
                print("Wrote cache for dataset hand_labels {} to {}".format(
                    self.data_split, cache_path
                ))
        return

    def _is_valid(self, clr, index):
        valid_data = isinstance(clr, (np.ndarray, PIL.Image.Image))

        if not valid_data:
            raise Exception("Encountered error processing CMU_2_[{}]".format(index))
        return valid_data

    def __len__(self):
        return len(self.clr_paths)

    def __str__(self):
        info = "hand_labels {} set. lenth {}".format(
            self.data_split, len(self.clr_paths)
        )
        return colored(info, 'blue', attrs=['bold'])

    def get_sample(self, index):
        flip = True if self.hand_side != self.sides[index] else False

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
            clr_ = np.array(clr)
            plt.figure(figsize=(20, 20))
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
    hand_labels = Hand_labels(
        data_root="/home/chen/datasets/CMU/hand_labels",
        data_split='train',
        hand_side='right',
        njoints=21,
        use_cache=True,
        vis=True
    )
    print("len(hand_labels)=", len(hand_labels))

    for i in tqdm(range(len(hand_labels))):
        print("i=", i)
        data = hand_labels.get_sample(i)


if __name__ == "__main__":
    main()
