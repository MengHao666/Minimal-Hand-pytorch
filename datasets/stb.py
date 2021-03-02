# Copyright (c) Lixin YANG. All Rights Reserved.
r"""
STB dataset
A Hand joint Tracking Benchmark from Stereo Matching, ICIP 2017
"""

import math
import os
import pickle

import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch.utils.data
from PIL import Image
from termcolor import colored
from tqdm import tqdm

import config as cfg
import utils.handutils as handutils

CACHE_HOME = os.path.expanduser(cfg.DEFAULT_CACHE_DIR)

# some globals, ugly but work
sk_fx_color = 607.92271
sk_fy_color = 607.88192
sk_tx_color = 314.78337
sk_ty_color = 236.42484

bb_fx = 822.79041
bb_fy = 822.79041
bb_tx = 318.47345
bb_ty = 250.31296

sk_rot_vec = [0.00531, -0.01196, 0.00301]
sk_trans_vec = [-24.0381, -0.4563, -1.2326]  # mm

snap_joint_name2id = {w: i for i, w in enumerate(cfg.snap_joint_names)}
stb_joint_name2id = {w: i for i, w in enumerate(cfg.stb_joints)}

stb_to_snap_id = [snap_joint_name2id[joint_name] for joint_name in cfg.stb_joints]


def sk_rot_mx(rot_vec):
    """
    use Rodrigues' rotation formula to transform the rotation vector into rotation matrix
    :param rot_vec:
    :return:
    """
    theta = np.linalg.norm(rot_vec)
    vector = np.array(rot_vec) * math.sin(theta / 2.0) / theta
    a = math.cos(theta / 2.0)
    b = -vector[0]
    c = -vector[1]
    d = -vector[2]
    return np.array(
        [
            [
                a * a + b * b - c * c - d * d,
                2 * (b * c + a * d),
                2 * (b * d - a * c)
            ],
            [
                2 * (b * c - a * d),
                a * a + c * c - b * b - d * d,
                2 * (c * d + a * b)
            ],
            [
                2 * (b * d + a * c),
                2 * (c * d - a * b),
                a * a + d * d - b * b - c * c
            ]
        ]
    )


def sk_xyz_depth2color(depth_xyz, trans_vec, rot_mx):
    """
    in the STB dataset: 'rotation and translation vector can transform the coordinates
                         relative to color camera to those relative to depth camera'.
    however here we want depth_xyz -> color_xyz
    a inverse transformation happen:
    T = [rot_mx | trans_vec | 0  1], Tinv = T.inv, then output Tinv * depth_xyz

    :param depth_xyz: N x 21 x 3, trans_vec: 3, rot_mx: 3 x 3
    :return: color_xyz: N x 21 x 3
    """
    color_xyz = depth_xyz - np.tile(trans_vec, [depth_xyz.shape[0], depth_xyz.shape[1], 1])
    return color_xyz.dot(rot_mx)


def stb_palm2wrist(joint_xyz):
    root = snap_joint_name2id['loc_bn_palm_L']  # 0

    index = snap_joint_name2id['loc_bn_index_L_01']  # 5
    mid = snap_joint_name2id['loc_bn_mid_L_01']  # 9
    ring = snap_joint_name2id['loc_bn_ring_L_01']  # 13
    pinky = snap_joint_name2id['loc_bn_pinky_L_01']  # 17

    def _new_root(joint_xyz, id, root_id):
        return joint_xyz[:, id, :] + \
               2.25 * (joint_xyz[:, root_id, :] - joint_xyz[:, id, :])  # N x K x 3

    joint_xyz[:, root, :] = \
        _new_root(joint_xyz, index, root) + \
        _new_root(joint_xyz, mid, root) + \
        _new_root(joint_xyz, ring, root) + \
        _new_root(joint_xyz, pinky, root)
    joint_xyz[:, root, :] = joint_xyz[:, root, :] / 4.0

    return joint_xyz


def _stb_palm2wrist(joint_xyz):
    root_id = snap_joint_name2id['loc_bn_palm_L']  # 0
    mid_root_id = snap_joint_name2id['loc_bn_mid_L_01']  # 9
    joint_xyz[:, root_id, :] = \
        joint_xyz[:, mid_root_id, :] + \
        2.2 * (joint_xyz[:, root_id, :] - joint_xyz[:, mid_root_id, :])  # N x K x 3
    return joint_xyz


def ge_palm2wrist(pose_xyz):
    root_id = snap_joint_name2id['loc_bn_palm_L']
    ring_root_id = snap_joint_name2id['loc_bn_ring_L_01']
    pose_xyz[:, root_id, :] = pose_xyz[:, ring_root_id, :] + \
                              2.0 * (pose_xyz[:, root_id, :] - pose_xyz[:, ring_root_id, :])  # N x K x 3
    return pose_xyz


class STBDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_root,
            data_split='train',
            hand_side='right',
            njoints=21,
            use_cache=True,
            visual=False
    ):
        if not os.path.exists(data_root):
            raise ValueError("data_root: %s not exist" % data_root)
        self.name = 'stb'
        self.data_split = data_split
        self.hand_side = hand_side
        self.img_paths = []
        self.dep_paths = []
        self.joints = []
        self.kp2ds = []
        self.centers = []
        self.my_scales = []
        self.njoints = njoints  # total 21 hand parts
        self.visual = visual

        self.root_id = snap_joint_name2id['loc_bn_palm_L']
        self.mid_mcp_id = snap_joint_name2id['loc_bn_mid_L_01']
        ann_base = os.path.join(data_root, "labels")
        img_base = os.path.join(data_root, "images")
        sk_rot = sk_rot_mx(sk_rot_vec)

        self.sk_intr = np.array([
            [sk_fx_color, 0.0, sk_tx_color],
            [0.0, sk_fy_color, sk_ty_color],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)  # (3,3)

        self.sequence = []
        if data_split == 'train':
            self.sequence = [
                "B2Counting",
                "B2Random",
                "B3Counting",
                "B3Random",
                "B4Counting",
                "B4Random",
                "B5Counting",
                "B5Random",
                "B6Counting",
                "B6Random"
            ]
        elif data_split == 'test':
            self.sequence = [
                "B1Counting",
                "B1Random"
            ]
        elif data_split == 'val':
            self.sequence = [
                "B2Counting",
                "B2Random"
            ]
        elif data_split == "train_val":
            self.sequence = [
                "B3Counting",
                "B3Random",
                "B4Counting",
                "B4Random",
                "B5Counting",
                "B5Random",
                "B6Counting",
                "B6Random"
            ]
        elif data_split == "all":
            self.sequence = [
                "B1Counting",
                "B1Random",
                "B2Counting",
                "B2Random",
                "B3Counting",
                "B3Random",
                "B4Counting",
                "B4Random",
                "B5Counting",
                "B5Random",
                "B6Counting",
                "B6Random"
            ]
        else:
            raise ValueError("split {} not in [train|test|val|train_val|all]")

        self.cache_folder = os.path.join(CACHE_HOME, "my-{}".format(data_split), "stb")
        os.makedirs(self.cache_folder, exist_ok=True)
        cache_path = os.path.join(
            self.cache_folder, "{}.pkl".format(self.data_split)
        )
        if os.path.exists(cache_path) and use_cache:
            with open(cache_path, "rb") as fid:
                annotations = pickle.load(fid)
                self.img_paths = annotations["img_paths"]
                self.dep_paths = annotations["dep_paths"]
                self.joints = annotations["joints"]
                self.kp2ds = annotations["kp2ds"]
                self.centers = annotations["centers"]
                self.my_scales = annotations["my_scales"]
            print("stb {} gt loaded from {}".format(self.data_split, cache_path))
            return

        self.imgpath_list = [
            os.path.join(img_base, seq) for seq in self.sequence
        ]

        imgsk_prefix = "SK_color"
        depsk_prefix = "SK_depth_seg"

        annsk_list = [
            os.path.join(
                ann_base,
                "{}_{}.mat".format(seq, imgsk_prefix[:2])
            ) for seq in self.sequence
        ]

        self.ann_list = annsk_list

        for imgpath, ann in zip(self.imgpath_list, self.ann_list):
            ''' we only use SK image '''
            assert "SK" in ann
            ''' 1. load joint '''
            rawmat = sio.loadmat(ann)
            rawjoint = rawmat["handPara"].transpose((2, 1, 0))  # N x K x 3
            num = rawjoint.shape[0]  # N

            rawjoint = sk_xyz_depth2color(rawjoint, sk_trans_vec, sk_rot)
            # reorder idx
            joint = rawjoint[:, stb_to_snap_id, :]
            # scale to meter
            joint = joint / 1000.0
            # root from palm to wrist
            # joint = _stb_palm2wrist(joint)  # N x K x 3 # yang lixin
            joint = ge_palm2wrist(joint)  # N x K x 3  #liu hao ge // vae//
            self.joints.append(joint)

            ''' 4. load images pth '''
            for idx in range(joint.shape[0]):
                self.img_paths.append(os.path.join(
                    imgpath, "{}_{}.png".format(imgsk_prefix, idx)
                ))
                self.dep_paths.append(os.path.join(
                    imgpath, "{}_{}.png".format(depsk_prefix, idx)
                ))

        self.joints = np.concatenate(self.joints, axis=0).astype(np.float32)  ##(30000, 21, 3)

        for i in tqdm(range(len(self.img_paths))):
            joint = self.joints[i]
            kp2d_homo = self.sk_intr.dot(joint.T).T
            kp2d = kp2d_homo / kp2d_homo[:, 2:3]
            kp2d = kp2d[:, :2]
            center = handutils.get_annot_center(kp2d)

            # caculate my_scale
            dep = Image.open(self.dep_paths[i]).convert("RGB")
            rel_dep = self.real_dep_img(dep)
            mask_rel_dep = np.argwhere(rel_dep > 1e-6)
            # my_scale = handutils.get_ori_crop_scale(mask_rel_dep, side=0, kp2d=kp2d) # ori
            my_scale = handutils.get_ori_crop_scale(mask_rel_dep, side=0, kp2d=kp2d,
                                                    mask_flag=False)  # get bbx only from kp2d ,比起ori差距不大,略好一点点一点点
            my_scale = (np.atleast_1d(my_scale))[np.newaxis, :]
            self.my_scales.append(my_scale)

            self.kp2ds.append(kp2d[np.newaxis, :, :])
            self.centers.append(center[np.newaxis, :])
            # self.scales.append((np.atleast_1d(scale))[np.newaxis, :])

        self.kp2ds = np.concatenate(self.kp2ds, axis=0).astype(np.float32)  # (N, 21, 2)
        self.centers = np.concatenate(self.centers, axis=0).astype(np.float32)  # (N, 2)
        # self.scales = np.concatenate(self.scales, axis=0).astype(np.float32)  # (N, 1)
        self.my_scales = np.concatenate(self.my_scales, axis=0).astype(np.float32)  # (N, 1)
        if use_cache:
            full_info = {
                "img_paths": self.img_paths,
                "dep_paths": self.dep_paths,
                "joints": self.joints,
                "kp2ds": self.kp2ds,
                "centers": self.centers,
                # "scales": self.scales,
                "my_scales": self.my_scales,
            }
            with open(cache_path, "wb") as fid:
                pickle.dump(full_info, fid)
                print("Wrote cache for dataset stb {} to {}".format(
                    self.data_split, cache_path
                ))
        return

    def __len__(self):
        """for STB dataset total (1,500 * 2) * 2 * 6 = 36,000 samples

        :return - if is train: 30,000 samples
        :return - if is eval:   6,000 samples
        """
        return len(self.img_paths)

    def __str__(self):
        info = "STB {} set. lenth {}".format(
            self.data_split, len(self.img_paths)
        )
        return colored(info, 'blue', attrs=['bold'])

    def _is_valid(self, clr, index):
        valid_data = isinstance(clr, (np.ndarray, PIL.Image.Image))

        if not valid_data:
            raise Exception("Encountered error processing stb[{}]".format(index))
        return valid_data

    def get_sample(self, index):  # replace __getitem__
        flip = True if self.hand_side != 'left' else False

        intr = self.sk_intr

        # prepare color image
        clr = Image.open(self.img_paths[index]).convert("RGB")
        self._is_valid(clr, index)

        # prepare joint
        joint = self.joints[index].copy()  # (21, 3)

        # prepare kp2d
        kp2d = self.kp2ds[index].copy()

        center = self.centers[index].copy()
        # scale = self.scales[index].copy()

        my_scale = self.my_scales[index].copy()

        if self.dep_paths[index]:
            dep = Image.open(self.dep_paths[index]).convert("RGB")
            ### dep values now are stored as |mod|div|0| (RGB)
            self._is_valid(dep, index)
            valid_dep = True
        else:
            dep = None
            valid_dep = False

        if flip:
            clr = clr.transpose(Image.FLIP_LEFT_RIGHT)
            center[0] = clr.size[0] - center[0]
            kp2d[:, 0] = clr.size[0] - kp2d[:, 0]
            joint[:, 0] = -joint[:, 0]
            if valid_dep:
                dep = dep.transpose(Image.FLIP_LEFT_RIGHT)

            # visualization
        if self.visual:
            clr_ = np.array(clr)
            dep_ = np.array(dep)

            fig = plt.figure(figsize=(20, 20))

            plt.subplot(2, 4, 1)
            clr1 = clr_.copy()
            rel_dep = self.real_dep_img(dep_)
            mask_rel_dep = np.argwhere(rel_dep > 1e-6)
            rmin, cmin = mask_rel_dep.min(0)
            rmax, cmax = mask_rel_dep.max(0)
            cv2.rectangle(clr1, (cmin, rmin), (cmax, rmax), (255), thickness=3)
            plt.imshow(clr1)
            plt.title('Color+BounduingBox')

            plt.subplot(2, 4, 2)
            dep1 = dep_.copy()
            plt.imshow(dep1)
            plt.title('Depth')

            clr_dep = clr_ + dep_
            plt.subplot(2, 4, 3)
            plt.imshow(clr_dep)
            plt.title('Color+Depth')

            rel_dep_ = rel_dep.copy()
            plt.subplot(2, 4, 4)
            plt.imshow(rel_dep_)
            plt.title('real_Depth')

            plt.subplot(2, 4, 5)
            plt.imshow(clr.copy())
            plt.title('Color')

            plt.subplot(2, 4, 6)
            plt.imshow(clr.copy())
            plt.plot(kp2d[:, :1], kp2d[:, 1:], 'ro')
            plt.title('Color+2D annotations')

            ax = fig.add_subplot(247, projection='3d')
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
            # plt.plot(joint [1:, 0], joint [1:, 1], joint [1:, 2], 'o')

            plt.title('3D annotations')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.legend()
            ax.view_init(-90, -90)
            plt.show()

        sample = {
            'index': index,
            'clr': clr,
            # 'dep': dep,  # if has renturn PIL image
            'kp2d': kp2d,
            'center': center,
            # 'scale': scale,
            'my_scale': my_scale,
            'joint': joint,
            'intr': intr,
            # 'valid_dep': valid_dep,
        }

        return sample

    def norm_dep_img(self, dep_, joint_z):
        if isinstance(dep_, PIL.Image.Image):
            dep_ = np.array(dep_)
            assert (dep_.shape[-1] == 3)  # used as "RGB"

        ''' Converts a RGB-coded depth into float valued depth. '''
        ''' dep values now are stored as |mod|div|0| (RGB) '''
        dep = (dep_[:, :, 1] * 2 ** 8 + dep_[:, :, 0]).astype('float32')
        dep /= 1000.0  # depth now in meter

        lower_bound = joint_z.min() - 0.05  # meter
        upper_bound = joint_z.max() + 0.05

        np.putmask(dep, dep <= lower_bound, upper_bound)
        min_dep = dep.min() - 1e-3  # slightly compensate
        np.putmask(dep, dep >= upper_bound, 0.0)
        max_dep = dep.max() + 1e-3
        np.putmask(dep, dep <= min_dep, max_dep)
        range_dep = max_dep - min_dep
        dep = (-1 * dep + max_dep) / range_dep
        return dep

    def real_dep_img(self, dep_):
        if isinstance(dep_, PIL.Image.Image):
            dep_ = np.array(dep_)
            assert (dep_.shape[-1] == 3)  # used as "RGB"

        ''' Converts a RGB-coded depth into float valued depth. '''
        ''' dep values now are stored as |mod|div|0| (RGB) '''
        dep = (dep_[:, :, 1] * 2 ** 8 + dep_[:, :, 0]).astype('float32')
        dep /= 1000.0  # depth now in meter

        return dep


def main():
    # go through the whole dataset
    data_split = "test"
    stb = STBDataset(
        data_root="/home/chen/datasets/STB",
        data_split=data_split,
        hand_side="right",
        visual=True
    )
    print("len(stb)=", len(stb))

    for i in tqdm(range(len(stb))):
        print("i=", i)
        data = stb.get_sample(i)


if __name__ == "__main__":
    main()
