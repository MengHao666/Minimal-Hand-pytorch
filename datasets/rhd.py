# Copyright (c) Lixin YANG. All Rights Reserved.
r"""
Randered dataset
Learning to Estimate 3D Hand joint from Single RGB Images, ICCV 2017
"""

import os
import pickle

import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image
from progress.bar import Bar
from termcolor import colored
from tqdm import tqdm

import config as cfg
import utils.handutils as handutils

CACHE_HOME = os.path.expanduser(cfg.DEFAULT_CACHE_DIR)

snap_joint_name2id = {w: i for i, w in enumerate(cfg.snap_joint_names)}
rhd_joint_name2id = {w: i for i, w in enumerate(cfg.rhd_joints)}
rhd_to_snap_id = [snap_joint_name2id[joint_name] for joint_name in cfg.rhd_joints]


class RHDDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_root="/disk1/data/RHD/RHD_published_v2",
            data_split='train',
            hand_side='right',
            njoints=21,
            use_cache=True,
            visual=False
    ):

        if not os.path.exists(data_root):
            raise ValueError("data_root: %s not exist" % data_root)

        self.name = 'rhd'
        self.data_split = data_split
        self.hand_side = hand_side
        self.clr_paths = []
        self.mask_paths = []
        self.joints = []
        self.kp2ds = []
        self.centers = []
        self.my_scales = []
        self.sides = []
        self.intrs = []
        self.njoints = njoints  # total 21 hand parts
        self.reslu = [320, 320]

        self.visual = visual

        self.root_id = snap_joint_name2id['loc_bn_palm_L']  # 0
        self.mid_mcp_id = snap_joint_name2id['loc_bn_mid_L_01']  # 9

        # [train|test|val|train_val|all]
        if data_split == 'train':
            self.sequence = ['training', ]
        elif data_split == 'test':
            self.sequence = ['evaluation', ]
        elif data_split == 'val':
            self.sequence = ['evaluation', ]
        elif data_split == 'train_val':
            self.sequence = ['training', ]
        elif data_split == 'all':
            self.sequence = ['training', 'evaluation']
        else:
            raise ValueError("split {} not in [train|test|val|train_val|all]".format(data_split))

        self.cache_folder = os.path.join(CACHE_HOME, "my-{}".format(data_split), "rhd")
        os.makedirs(self.cache_folder, exist_ok=True)
        cache_path = os.path.join(
            self.cache_folder, "{}.pkl".format(self.data_split)
        )
        if os.path.exists(cache_path) and use_cache:
            with open(cache_path, "rb") as fid:
                annotations = pickle.load(fid)
                self.sides = annotations["sides"]
                self.clr_paths = annotations["clr_paths"]
                self.mask_paths = annotations["mask_paths"]
                self.joints = annotations["joints"]
                self.kp2ds = annotations["kp2ds"]
                self.intrs = annotations["intrs"]
                self.centers = annotations["centers"]
                self.my_scales = annotations["my_scales"]
            print("rhd {} gt loaded from {}".format(self.data_split, cache_path))
            return

        datapath_list = [
            os.path.join(data_root, seq) for seq in self.sequence
        ]
        annoname_list = [
            "anno_{}.pickle".format(seq) for seq in self.sequence
        ]
        anno_list = [
            os.path.join(datapath, annoname) \
            for datapath, annoname in zip(datapath_list, annoname_list)
        ]
        clr_root_list = [
            os.path.join(datapath, "color") for datapath in datapath_list
        ]
        dep_root_list = [
            os.path.join(datapath, "depth") for datapath in datapath_list
        ]
        mask_root_list = [
            os.path.join(datapath, "mask") for datapath in datapath_list
        ]

        print("init RHD {}, It will take a while at first time".format(data_split))
        for anno, clr_root, dep_root, mask_root \
                in zip(
            anno_list,
            clr_root_list,
            dep_root_list,
            mask_root_list
        ):

            with open(anno, 'rb') as fi:
                rawdatas = pickle.load(fi)
                fi.close()

            bar = Bar('RHD', max=len(rawdatas))
            for i in tqdm(range(len(rawdatas))):

                raw = rawdatas[i]
                rawkp2d = raw['uv_vis'][:, : 2]  # kp 2d left & right hand
                rawvis = raw['uv_vis'][:, 2]

                rawjoint = raw['xyz']  # x, y, z coordinates of the keypoints, in meters
                rawintr = raw['K']

                ''' "both" means left, right'''
                kp2dboth = [
                    rawkp2d[:21][rhd_to_snap_id, :],
                    rawkp2d[21:][rhd_to_snap_id, :]
                ]
                visboth = [
                    rawvis[:21][rhd_to_snap_id],
                    rawvis[21:][rhd_to_snap_id]
                ]
                jointboth = [
                    rawjoint[:21][rhd_to_snap_id, :],
                    rawjoint[21:][rhd_to_snap_id, :]
                ]

                intrboth = [rawintr, rawintr]
                sideboth = ['l', 'r']

                l_kp_count = np.sum(raw['uv_vis'][:21, 2] == 1)
                r_kp_count = np.sum(raw['uv_vis'][21:, 2] == 1)
                vis_side = 'l' if l_kp_count > r_kp_count else 'r'

                for kp2d, vis, joint, side, intr \
                        in zip(kp2dboth, visboth, jointboth, sideboth, intrboth):
                    if side != vis_side:
                        continue

                    clrpth = os.path.join(clr_root, '%.5d.png' % i)
                    maskpth = os.path.join(mask_root, '%.5d.png' % i)
                    self.clr_paths.append(clrpth)
                    self.mask_paths.append(maskpth)
                    self.sides.append(side)

                    joint = joint[np.newaxis, :, :]
                    self.joints.append(joint)

                    center = handutils.get_annot_center(kp2d)
                    kp2d = kp2d[np.newaxis, :, :]
                    self.kp2ds.append(kp2d)

                    center = center[np.newaxis, :]
                    self.centers.append(center)

                    mask = Image.open(maskpth).convert("RGB")
                    mask = np.array(mask)[:, :, 2:]
                    my_scale = handutils.get_ori_crop_scale(mask, side, kp2d.squeeze(0))
                    my_scale = (np.atleast_1d(my_scale))[np.newaxis, :]
                    self.my_scales.append(my_scale)

                    intr = intr[np.newaxis, :]
                    self.intrs.append(intr)

                bar.suffix = ('({n}/{all}), total:{t:}s, eta:{eta:}s').format(
                    n=i + 1, all=len(rawdatas), t=bar.elapsed_td, eta=bar.eta_td)
                bar.next()

            bar.finish()
        self.joints = np.concatenate(self.joints, axis=0).astype(np.float32)  # (N, 21, 3)

        self.kp2ds = np.concatenate(self.kp2ds, axis=0).astype(np.float32)  # (N, 21, 2)
        self.centers = np.concatenate(self.centers, axis=0).astype(np.float32)  # (N, 1)
        self.my_scales = np.concatenate(self.my_scales, axis=0).astype(np.float32)  # (N, 1)
        self.intrs = np.concatenate(self.intrs, axis=0).astype(np.float32)  # (N, 3,3)

        if use_cache:
            full_info = {
                "sides": self.sides,
                "clr_paths": self.clr_paths,
                "mask_paths": self.mask_paths,
                "joints": self.joints,
                "kp2ds": self.kp2ds,
                "intrs": self.intrs,
                "centers": self.centers,
                "my_scales": self.my_scales,
            }
            with open(cache_path, "wb") as fid:
                pickle.dump(full_info, fid)
                print("Wrote cache for dataset rhd {} to {}".format(
                    self.data_split, cache_path
                ))
        return

    def get_sample(self, index):
        side = self.sides[index]
        """ 'r' in 'left' / 'l' in 'right' """
        flip = True if (side not in self.hand_side) else False

        clr = Image.open(self.clr_paths[index]).convert("RGB")
        self._is_valid(clr, index)
        mask = Image.open(self.mask_paths[index]).convert("RGB")
        self._is_valid(mask, index)

        # prepare jont
        joint = self.joints[index].copy()

        # prepare kp2d
        kp2d = self.kp2ds[index].copy()

        center = self.centers[index].copy()
        # scale = self.scales[index].copy()

        my_scale = self.my_scales[index].copy()

        if flip:
            clr = clr.transpose(Image.FLIP_LEFT_RIGHT)
            center[0] = clr.size[0] - center[0]  # clr.size[0] represents width of image
            kp2d[:, 0] = clr.size[0] - kp2d[:, 0]
            joint[:, 0] = -joint[:, 0]

        sample = {
            'index': index,
            'clr': clr,
            'kp2d': kp2d,
            'center': center,
            'my_scale': my_scale,
            'joint': joint,
            'intr': self.intrs[index],
        }

        if self.visual:
            fig = plt.figure(figsize=(20, 20))
            plt.subplot(1, 3, 1)
            plt.imshow(clr.copy())
            plt.title('Color')

            plt.subplot(1, 3, 2)
            plt.imshow(clr.copy())
            plt.plot(kp2d[:, :1], kp2d[:, 1:], 'ro')
            plt.title('Color+2D annotations')

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
            # plt.plot(joint [1:, 0], joint [1:, 1], joint [1:, 2], 'o')

            plt.title('3D annotations')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.legend()
            ax.view_init(-90, -90)
            plt.show()

        return sample

    def _apply_mask(self, dep, mask, side):
        ''' follow the label rules in RHD datasets '''
        if side is 'l':
            valid_mask_id = [i for i in range(2, 18)]
        else:
            valid_mask_id = [i for i in range(18, 34)]

        mask = np.array(mask)[:, :, 2:]
        dep = np.array(dep)
        ll = valid_mask_id[0]
        uu = valid_mask_id[-1]
        mask[mask < ll] = 0
        mask[mask > uu] = 0
        mask[mask > 0] = 1
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        dep = np.multiply(dep, mask)
        dep = Image.fromarray(dep, mode="RGB")
        return dep

    def __len__(self):
        return len(self.clr_paths)

    def __str__(self):
        info = "RHD {} set. lenth {}".format(
            self.data_split, len(self.clr_paths)
        )
        return colored(info, 'yellow', attrs=['bold'])

    def norm_dep_img(self, dep_):
        """RHD depthmap to depth image
        """
        if isinstance(dep_, PIL.Image.Image):
            dep_ = np.array(dep_)
            assert (dep_.shape[-1] == 3)  # used to be "RGB"

        ''' Converts a RGB-coded depth into float valued depth. '''
        dep = (dep_[:, :, 0] * 2 ** 8 + dep_[:, :, 1]).astype('float32')
        dep /= float(2 ** 16 - 1)
        dep *= 5.0  ## depth in meter !

        return dep

    def _is_valid(self, img, index):
        valid_data = isinstance(img, (np.ndarray, PIL.Image.Image))
        if not valid_data:
            raise Exception("Encountered error processing rhd[{}]".format(index))
        return valid_data


def main():
    data_split = 'test'
    rhd = RHDDataset(
        data_root="/home/chen/datasets/RHD/RHD_published_v2",
        data_split=data_split,
        hand_side='right',
        njoints=21,
        use_cache=False,
        visual=True
    )
    print("len(rhd)=", len(rhd))

    for i in tqdm(range(len(rhd))):
        print("id=", id)
        data = rhd.get_sample(i)


if __name__ == "__main__":
    main()
