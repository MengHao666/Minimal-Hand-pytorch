# Copyright (c) Hao Meng. All Rights Reserved.
r"""
GANeratedDataset
GANerated Hands for Real-Time 3D Hand Tracking from Monocular RGB, CVPR 2018
Link to dataset: https://handtracker.mpi-inf.mpg.de/projects/GANeratedHands/GANeratedDataset.htm
"""

import os
import pickle
from builtins import print

import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from termcolor import colored
from tqdm import tqdm

import config as cfg
import utils.handutils as handutils

CACHE_HOME = os.path.expanduser(cfg.DEFAULT_CACHE_DIR)
snap_joint_name2id = {w: i for i, w in enumerate(cfg.snap_joint_names)}


class GANeratedDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_root,
                 data_split='train',
                 hand_side='right',
                 njoints=21,
                 use_cache=True,
                 vis=False):
        if not os.path.exists(data_root):
            raise ValueError("data_root: %s not exist" % data_root)
        self.name = 'GANeratedHands Dataset'
        self.data_split = data_split
        self.hand_side = hand_side
        self.clr_paths = []
        self.kp2ds = []
        self.joints = []
        self.centers = []
        self.my_scales = []
        self.njoints = njoints
        self.reslu = [256, 256]

        self.vis = vis

        self.root_id = snap_joint_name2id['loc_bn_palm_L']  # 0
        self.mid_mcp_id = snap_joint_name2id['loc_bn_mid_L_01']  # 9

        self.intr = np.array([
            [617.173, 0, 315.453],
            [0, 617.173, 242.259],
            [0, 0, 1]])

        # [train|test|val|train_val|all]
        if data_split == 'train':
            self.sequence = ['training', ]
        else:
            print("GANeratedDataset only has train set!")
            return None

        self.cache_folder = os.path.join(CACHE_HOME, "my-train", "GANeratedHands")
        os.makedirs(self.cache_folder, exist_ok=True)
        cache_path = os.path.join(
            self.cache_folder, "{}.pkl".format(self.data_split)
        )

        if os.path.exists(cache_path) and use_cache:
            with open(cache_path, "rb") as fid:
                annotations = pickle.load(fid)
                self.clr_paths = annotations["clr_paths"]
                self.kp2ds = annotations["kp2ds"]
                self.joints = annotations["joints"]
                self.centers = annotations["centers"]
                self.my_scales = annotations["my_scales"]
            print("GANeratedHands {} gt loaded from {}".format(self.data_split, cache_path))
            return

        print("init GANeratedHands {}, It will take a while at first time".format(data_split))

        for img_type in ['noObject/', 'withObject/']:
            folders = os.listdir(data_root + img_type)
            folders = sorted(folders)
            folders = [img_type + x + '/' for x in folders if len(x) == 4]

            for folder in folders:
                images = os.listdir(os.path.join(data_root + folder))
                images = [data_root + folder + x for x in images if x.find('.png') > 0]
                images = sorted(images)

                self.clr_paths.extend(images)

        for idx in tqdm(range(len(self.clr_paths))):
            img_name = self.clr_paths[idx]

            fn_2d_keypoints = img_name.replace('color_composed.png', 'joint2D.txt')
            arr_2d_keypoints = np.loadtxt(fn_2d_keypoints, delimiter=',')
            arr_2d_keypoints = arr_2d_keypoints.reshape([-1, 2])

            center = handutils.get_annot_center(arr_2d_keypoints)
            self.centers.append(center[np.newaxis, :])

            my_scale = handutils.get_ori_crop_scale(mask=None, mask_flag=False, side=None, kp2d=arr_2d_keypoints,
                                                    )
            my_scale = (np.atleast_1d(my_scale))[np.newaxis, :]
            self.my_scales.append(my_scale)

            arr_2d_keypoints = arr_2d_keypoints[np.newaxis, :, :]
            self.kp2ds.append(arr_2d_keypoints)

            fn_3d_keypoints = img_name.replace('color_composed.png', 'joint_pos_global.txt')
            arr_3d_keypoints = np.loadtxt(fn_3d_keypoints, delimiter=',')
            arr_3d_keypoints = arr_3d_keypoints.reshape([-1, 3])
            arr_3d_keypoints = arr_3d_keypoints[np.newaxis, :, :]
            self.joints.append(arr_3d_keypoints)

        self.joints = np.concatenate(self.joints, axis=0).astype(np.float32)
        self.kp2ds = np.concatenate(self.kp2ds, axis=0).astype(np.float32)  # (N, 21, 2)
        self.centers = np.concatenate(self.centers, axis=0).astype(np.float32)  # (N, 2)
        self.my_scales = np.concatenate(self.my_scales, axis=0).astype(np.float32)

        if use_cache:
            full_info = {
                "clr_paths": self.clr_paths,
                "joints": self.joints,
                "kp2ds": self.kp2ds,
                "centers": self.centers,
                "my_scales": self.my_scales,
            }
            with open(cache_path, "wb") as fid:
                pickle.dump(full_info, fid)
                print("Wrote cache for dataset GANeratedDataset {} to {}".format(
                    self.data_split, cache_path
                ))
        return

    def __len__(self):
        """for GANeratedHands Dataset total (1,500 * 2) * 2 * 6 = 36,000 samples
        """
        return len(self.clr_paths)

    def __str__(self):
        info = "GANeratedHands {} set. lenth {}".format(
            self.data_split, len(self.clr_paths)
        )
        return colored(info, 'blue', attrs=['bold'])

    def _is_valid(self, clr, index):
        valid_data = isinstance(clr, (np.ndarray, PIL.Image.Image))

        if not valid_data:
            raise Exception("Encountered error processing GAN[{}]".format(index))
        return valid_data

    def get_sample(self, index):
        flip = True if self.hand_side != "left" else False

        intr = self.intr

        # prepare color image
        clr = Image.open(self.clr_paths[index]).convert("RGB")
        self._is_valid(clr, index)

        # prepare kp2d
        kp2d = self.kp2ds[index].copy()

        # prepare joint
        joint = self.joints[index].copy()  # (21, 3)
        center = self.centers[index].copy()
        my_scale = self.my_scales[index].copy()
        if flip:
            clr = clr.transpose(Image.FLIP_LEFT_RIGHT)
            center[0] = clr.size[0] - center[0]
            kp2d[:, 0] = clr.size[0] - kp2d[:, 0]
            joint[:, 0] = -joint[:, 0]

        sample = {
            'index': index,
            'clr': clr,
            'kp2d': kp2d,
            'center': center,
            'my_scale': my_scale,
            'joint': joint,
            'intr': intr,
        }

        # visualization
        if self.vis:
            fig = plt.figure(figsize=(20, 20))
            clr_ = np.array(clr)

            plt.subplot(1, 3, 1)
            clr1 = clr_.copy()
            plt.imshow(clr1)

            plt.subplot(1, 3, 2)
            clr2 = clr_.copy()
            plt.imshow(clr2)

            for p in range(kp2d.shape[0]):
                plt.plot(kp2d[p][0], kp2d[p][1], 'r.')
                plt.text(kp2d[p][0], kp2d[p][1], '{0}'.format(p), fontsize=5)

            ax = fig.add_subplot(133, projection='3d')
            plt.plot(joint[:, 0], joint[:, 1], joint[:, 2], 'yo', label='keypoint')
            plt.plot(joint[:5, 0], joint[:5, 1],
                     joint[:5, 2],
                     'r',
                     label='thumb')
            plt.plot(joint[[0, 5, 6, 7, 8, ], 0], joint[[0, 5, 6, 7, 8, ], 1],
                     joint[[0, 5, 6, 7, 8, ], 2],
                     'b',
                     label='index')
            plt.plot(joint[[0, 9, 10, 11, 12, ], 0], joint[[0, 9, 10, 11, 12], 1],
                     joint[[0, 9, 10, 11, 12], 2],
                     'b',
                     label='middle')
            plt.plot(joint[[0, 13, 14, 15, 16], 0], joint[[0, 13, 14, 15, 16], 1],
                     joint[[0, 13, 14, 15, 16], 2],
                     'b',
                     label='ring')
            plt.plot(joint[[0, 17, 18, 19, 20], 0], joint[[0, 17, 18, 19, 20], 1],
                     joint[[0, 17, 18, 19, 20], 2],
                     'b',
                     label='pinky')
            # snap convention
            plt.plot(joint[4][0], joint[4][1], joint[4][2], 'rD', label='thumb')
            plt.plot(joint[8][0], joint[8][1], joint[8][2], 'ro', label='index')
            plt.plot(joint[12][0], joint[12][1], joint[12][2], 'ro', label='middle')
            plt.plot(joint[16][0], joint[16][1], joint[16][2], 'ro', label='ring')
            plt.plot(joint[20][0], joint[20][1], joint[20][2], 'ro', label='pinky')

            plt.title('3D annotations')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.legend()
            ax.view_init(-90, -90)
            plt.show()

        return sample


if __name__ == '__main__':
    data_split = 'train'
    gan = GANeratedDataset(
        data_root='/home/chen/datasets/GANeratedHands_Release/data/',
        data_split=data_split,
        hand_side='right',
        njoints=21,
        use_cache=False,
        vis=True)
    print("len(gan)=", len(gan))
    for i in range(len(gan)):
        print("i=", i)
        data = gan.get_sample(i)
