# Copyright (c) Hao Meng. All Rights Reserved.
r"""
Hand dataset controll all sub dataset
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image, ImageFilter
from termcolor import colored
from tqdm import tqdm

import config as cfg
import utils.func as func
import utils.handutils as handutils
import utils.heatmaputils as hmutils
import utils.imgutils as imutils
from datasets.dexter_object import DexterObjectDataset
from datasets.ganerated_hands import GANeratedDataset
from datasets.hand143_panopticdb import Hand143_panopticdb
from datasets.hand_labels import Hand_labels
from datasets.rhd import RHDDataset
from datasets.stb import STBDataset

snap_joint_name2id = {w: i for i, w in enumerate(cfg.snap_joint_names)}


class HandDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_split='train',
            data_root="/disk1/data",
            subset_name=['rhd', 'stb'],
            hand_side='right',
            sigma=1.0,
            inp_res=128,
            hm_res=32,
            njoints=21,
            train=True,
            scale_jittering=0.1,
            center_jettering=0.1,
            max_rot=np.pi,
            hue=0.15,
            saturation=0.5,
            contrast=0.5,
            brightness=0.5,
            blur_radius=0.5, vis=False
    ):

        self.inp_res = inp_res  # 128 # network input resolution
        self.hm_res = hm_res  # 32  # out_testset hm resolution
        self.njoints = njoints
        self.sigma = sigma
        self.max_rot = max_rot

        # Training attributes
        self.train = train
        self.scale_jittering = scale_jittering
        self.center_jittering = center_jettering

        # Color jitter attributes
        self.hue = hue
        self.contrast = contrast
        self.brightness = brightness
        self.saturation = saturation
        self.blur_radius = blur_radius

        self.datasets = []
        self.ref_bone_link = (0, 9)  # mid mcp
        self.joint_root_idx = 9  # root

        self.vis = vis

        if 'stb' in subset_name:
            self.stb = STBDataset(
                data_root=os.path.join(data_root, 'STB'),
                data_split=data_split,
                hand_side=hand_side,
                njoints=njoints,
            )
            print(self.stb)
            self.datasets.append(self.stb)

        if 'rhd' in subset_name:
            self.rhd = RHDDataset(
                data_root=os.path.join(data_root, 'RHD/RHD_published_v2'),
                data_split=data_split,
                hand_side=hand_side,
                njoints=njoints,
            )
            print(self.rhd)
            self.datasets.append(self.rhd)

        if 'cmu' in subset_name:
            self.hand143_panopticdb = Hand143_panopticdb(
                data_root=os.path.join(data_root, 'CMU/hand143_panopticdb'),
                data_split=data_split,
                hand_side=hand_side,
                njoints=njoints,
            )
            print(self.hand143_panopticdb)
            self.datasets.append(self.hand143_panopticdb)

            self.hand_labels = Hand_labels(
                data_root=os.path.join(data_root, 'CMU/hand_labels'),
                data_split=data_split,
                hand_side=hand_side,
                njoints=njoints,
            )
            print(self.hand_labels)
            self.datasets.append(self.hand_labels)

            info = "CMU {} set. lenth {}".format(
                data_split, len(self.hand_labels) + len(self.hand143_panopticdb)
            )
            print(colored(info, 'yellow', attrs=['bold']))

        if 'gan' in subset_name:
            self.gan = GANeratedDataset(
                data_root=os.path.join(data_root, 'GANeratedHands_Release/data/'),
                data_split=data_split,
                hand_side=hand_side,
                njoints=njoints,
            )
            print(self.gan)
            self.datasets.append(self.gan)

        if 'do' in subset_name:
            self.do = DexterObjectDataset(
                data_root=os.path.join(data_root, 'dexter+object'),
                data_split=data_split,
                hand_side=hand_side,
                njoints=njoints,
            )
            print(self.do)
            self.datasets.append(self.do)

        self.total_data = 0
        for ds in self.datasets:
            self.total_data += len(ds)

    def __getitem__(self, index):
        rng = np.random.RandomState(seed=random.randint(0, 1024))
        try:
            sample, ds = self._get_sample(index)
        except Exception:
            index = np.random.randint(0, len(self))
            sample, ds = self._get_sample(index)

        clr = sample['clr']
        my_clr1 = clr.copy()
        center = sample['center']
        scale = sample['my_scale']
        if 'intr' in sample.keys():
            intr = sample['intr']

        # Data augmentation
        if self.train:
            center_offsets = (
                    self.center_jittering
                    * scale
                    * rng.uniform(low=-1, high=1, size=2)
            )
            center = center + center_offsets.astype(int)

            # Scale jittering
            scale_jittering = self.scale_jittering * rng.randn() + 1
            scale_jittering = np.clip(
                scale_jittering,
                1 - self.scale_jittering,
                1 + self.scale_jittering,
            )
            scale = scale * scale_jittering
            rot = rng.uniform(low=-self.max_rot, high=self.max_rot)
        else:
            rot = 0

        rot_mat = np.array([
            [np.cos(rot), -np.sin(rot), 0],
            [np.sin(rot), np.cos(rot), 0],
            [0, 0, 1],
        ]).astype(np.float32)

        if 'intr' in sample.keys():
            affinetrans, post_rot_trans = handutils.get_affine_transform(
                center=center,
                scale=scale,
                optical_center=[intr[0, 2], intr[1, 2]],
                out_res=[self.inp_res, self.inp_res],
                rot=rot
            )
        else:
            affinetrans, post_rot_trans = handutils.get_affine_transform_test(
                center, scale, [self.inp_res, self.inp_res], rot=rot
            )

        ''' prepare kp2d '''
        kp2d = sample['kp2d']
        kp2d_ori = kp2d.copy()
        kp2d = handutils.transform_coords(kp2d, affinetrans)

        ''' Generate GT Gussian hm and hm veil '''
        hm = np.zeros(
            (self.njoints, self.hm_res, self.hm_res),
            dtype='float32'
        )  # (CHW)
        hm_veil = np.ones(self.njoints, dtype='float32')
        for i in range(self.njoints):
            kp = (
                    (kp2d[i] / self.inp_res) * self.hm_res
            ).astype(np.int32)  # kp uv: [0~256] -> [0~64]
            hm[i], aval = hmutils.gen_heatmap(hm[i], kp, self.sigma)
            hm_veil[i] *= aval

        joint = np.zeros([21, 3])
        delta_map = np.zeros([21, 3, 32, 32])
        location_map = np.zeros([21, 3, 32, 32])
        flag = 0

        if 'joint' in sample.keys():

            flag = 1
            ''' prepare joint '''
            joint = sample['joint']
            if self.train:
                joint = rot_mat.dot(
                    joint.transpose(1, 0)
                ).transpose()

            joint_bone = 0
            for jid, nextjid in zip(self.ref_bone_link[:-1], self.ref_bone_link[1:]):
                joint_bone += np.linalg.norm(joint[nextjid] - joint[jid])
            joint_root = joint[self.joint_root_idx]
            joint_bone = np.atleast_1d(joint_bone)

            '''prepare location maps L'''
            jointR = joint - joint_root[np.newaxis, :]  # root relative
            jointRS = jointR / joint_bone  # scale invariant
            # '''jointRS.shape= (21, 3) to locationmap(21,3,32,32)'''
            location_map = jointRS[:, :, np.newaxis, np.newaxis].repeat(32, axis=-2).repeat(32, axis=-1)

            '''prepare delta maps D'''
            kin_chain = [
                jointRS[i] - jointRS[cfg.SNAP_PARENT[i]]
                for i in range(21)
            ]
            kin_chain = np.array(kin_chain)  # id 0's parent is itself #21*3
            kin_len = np.linalg.norm(
                kin_chain, ord=2, axis=-1, keepdims=True  # 21*1
            )
            kin_chain[1:] = kin_chain[1:] / kin_len[1:]
            # '''kin_chain(21, 3) to delta_map(21,3,32,32)'''
            delta_map = kin_chain[:, :, np.newaxis, np.newaxis].repeat(32, axis=-2).repeat(32, axis=-1)

        if 'tip' in sample.keys():
            joint = sample['tip']
            if self.train:
                joint = rot_mat.dot(
                    joint.transpose(1, 0)
                ).transpose()

        ''' prepare clr image '''
        if self.train:
            blur_radius = random.random() * self.blur_radius
            clr = clr.filter(ImageFilter.GaussianBlur(blur_radius))
            clr = imutils.color_jitter(
                clr,
                brightness=self.brightness,
                saturation=self.saturation,
                hue=self.hue,
                contrast=self.contrast,
            )

        # Transform and crop
        clr = handutils.transform_img(
            clr, affinetrans, [self.inp_res, self.inp_res]
        )
        clr = clr.crop((0, 0, self.inp_res, self.inp_res))
        my_clr2 = clr.copy()

        ''' implicit HWC -> CHW, 255 -> 1 '''
        clr = func.to_tensor(clr).float()
        ''' 0-mean, 1 std,  [0,1] -> [-0.5, 0.5] '''
        clr = func.normalize(clr, [0.5, 0.5, 0.5], [1, 1, 1])

        # visualization
        if self.vis:

            clr1 = my_clr1.copy()

            fig = plt.figure(figsize=(20, 10))
            plt.subplot(1, 4, 1)
            plt.imshow(np.asarray(clr1))
            plt.title('ori_Color+2D annotations')
            plt.plot(kp2d_ori[0, 0], kp2d_ori[0, 1], 'ro', markersize=5)
            plt.text(kp2d_ori[0][0], kp2d_ori[0][1], '0', color="w", fontsize=7.5)
            for p in range(1, kp2d_ori.shape[0]):
                plt.plot(kp2d_ori[p][0], kp2d_ori[p][1], 'bo', markersize=5)
                plt.text(kp2d_ori[p][0], kp2d_ori[p][1], '{0}'.format(p), color="w", fontsize=5)

            plt.subplot(1, 4, 2)
            clr2 = np.array(my_clr2.copy())
            plt.imshow(clr2)
            plt.plot(kp2d[0, 0], kp2d[0, 1], 'ro', markersize=5)
            plt.text(kp2d[0][0], kp2d[0][1], '0', color="w", fontsize=7.5)
            for p in range(1, kp2d.shape[0]):
                plt.plot(kp2d[p][0], kp2d[p][1], 'bo', markersize=5)
                plt.text(kp2d[p][0], kp2d[p][1], '{0}'.format(p), color="w", fontsize=5)
            plt.title('cropped_Color+2D annotations')

            plt.subplot(1, 4, 3)
            clr3 = my_clr2.copy().resize((self.hm_res, self.hm_res), Image.ANTIALIAS)
            tmp = clr3.convert('L')
            tmp = np.array(tmp)
            for k in range(hm.shape[0]):
                tmp = tmp + hm[k] * 64
            plt.imshow(tmp)
            plt.title('heatmap')

            if 'joint' in sample.keys():
                ax = fig.add_subplot(144, projection='3d')

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
                plt.plot(joint[8][0], joint[8][1], joint[8][2], 'r*', label='index')
                plt.plot(joint[12][0], joint[12][1], joint[12][2], 'rs', label='middle')
                plt.plot(joint[16][0], joint[16][1], joint[16][2], 'ro', label='ring')
                plt.plot(joint[20][0], joint[20][1], joint[20][2], 'rv', label='pinky')

                plt.title('3D annotations')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                plt.legend()
                ax.view_init(-90, -90)

            plt.show()

        ## to torch tensor
        clr = clr
        hm = torch.from_numpy(hm).float()
        hm_veil = torch.from_numpy(hm_veil).float()
        joint = torch.from_numpy(joint).float()
        location_map = torch.from_numpy(location_map).float()
        delta_map = torch.from_numpy(delta_map).float()

        metas = {
            'index': index,
            'clr': clr,
            'hm': hm,
            'hm_veil': hm_veil,
            'location_map': location_map,
            'delta_map': delta_map,
            'flag_3d': flag,
            "joint": joint
        }

        return metas

    def _get_sample(self, index):
        base = 0
        dataset = None
        for ds in self.datasets:
            if index < base + len(ds):
                sample = ds.get_sample(index - base)
                dataset = ds
                break
            else:
                base += len(ds)
        return sample, dataset

    def __len__(self):
        return self.total_data


if __name__ == '__main__':
    test_set = HandDataset(
        data_split='test',
        train=False,
        scale_jittering=0.1,
        center_jettering=0.1,
        max_rot=0.5 * np.pi,
        subset_name=["rhd", "stb", "do", "eo"],
        data_root="/home/chen/datasets/", vis=True
    )

    for id in tqdm(range(0, len(test_set), 10)):
        print("id=", id)
        data = test_set[id]
