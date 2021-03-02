# Copyright (c) Hao Meng. All Rights Reserved.

r"""
DexterObjectDataset
Real-time Joint Tracking of a Hand Manipulating an Object from RGB-D Input, ECCV 2016
Link to dataset: https://handtracker.mpi-inf.mpg.de/projects/RealtimeHO/dexter+object.htm
"""

import os
import pickle
from builtins import print

import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL import Image
from termcolor import colored
from tqdm import tqdm

import config as cfg
from utils import handutils

CACHE_HOME = os.path.expanduser(cfg.DEFAULT_CACHE_DIR)


# all hands in DexterObjectDataset are left hands
class DexterObjectDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_root,
                 data_split='test',
                 hand_side='right',
                 njoints=21,
                 use_cache=True, vis=False):
        if not os.path.exists(data_root):
            raise ValueError("data_root: %s not exist" % data_root)

        self.name = 'do'
        self.data_root = data_root
        self.data_split = data_split
        self.hand_side = hand_side
        self.clr_paths = []
        self.dep_paths = []
        self.mask_paths = []
        self.joints = []
        self.anno_2d_depth = []
        self.centers = []
        self.my_scales = []
        self.sides = []
        self.intrs = []
        self.njoints = njoints
        self.reslu = [480, 640]
        self.vis = vis

        self.image_size = 128

        if data_split == 'test':
            self.sequence = ['Grasp1', 'Grasp2', 'Occlusion', 'Rigid', 'Pinch', 'Rotate']
        else:
            print("DexterObjectDataset here only for evaluation, no train set here !")
            return None

        # self.bboxes = pd.read_csv(
        #     os.path.join(data_root, 'bbox_dexter+object.csv'))

        color_intrisics = np.array([[587.45209, 0, 325],
                                    [0, 600.67456, 249],
                                    [0, 0, 1]])

        color_extrisics = np.array([[0.9999, 0.0034, 0.0161, 19.0473],
                                    [-0.0033, 1.0000, -0.0079, -1.8514],
                                    [-0.0162, 0.0079, 0.9998, -4.7501]])

        self.depth_intrisics = np.array([[224.502, 0, 160],
                                         [0, 230.494, 120],
                                         [0, 0, 1]])

        self.xmap = np.array([[j for i in range(320)] for j in range(240)])
        self.ymap = np.array([[i for i in range(320)] for j in range(240)])

        self.M_color = np.matmul(color_intrisics, color_extrisics)
        self.DO_PRED_2D = np.load(os.path.join(self.data_root, "DO_pred_2d.npy"))

        self.cache_folder = os.path.join(CACHE_HOME, "my-test", "DexterObjectDataset")
        os.makedirs(self.cache_folder, exist_ok=True)
        cache_path = os.path.join(
            self.cache_folder, "{}.pkl".format(self.data_split)
        )

        if os.path.exists(cache_path) and use_cache:
            with open(cache_path, "rb") as fid:
                annotations = pickle.load(fid)
                self.clr_paths = annotations["clr_paths"]
                self.dep_paths = annotations["dep_paths"]
                self.anno_2d_depth = annotations["2d_depth"]
                self.joints = annotations["joints"],
                self.joints = self.joints[0]
                self.centers = annotations["centers"]
                self.my_scales = annotations["my_scales"]
            print("DexterObjectDataset {} gt loaded from {}".format(self.data_split, cache_path))
            return

        print("init DexterObjectDataset {}, It may take a while at first time".format(data_split))

        for fd in self.sequence:
            clr_fd_path = os.path.join(self.data_root, 'data', fd, 'color')
            clr_files = os.listdir(clr_fd_path)
            clr_files = [os.path.join(fd, 'color', x) for x in clr_files]
            clr_files = np.sort(clr_files)
            self.clr_paths.extend(clr_files)

            dep_fd_path = os.path.join(self.data_root, 'data', fd, 'depth')
            dep_files = os.listdir(dep_fd_path)
            dep_files = [os.path.join(fd, 'depth', x) for x in dep_files]
            dep_files = np.sort(dep_files)
            self.dep_paths.extend(dep_files)

            fn_anno_2d = os.path.join(self.data_root, 'data', fd, 'annotations/', fd + '2D.txt')
            df_anno_2d = pd.read_table(fn_anno_2d, sep=';', header=None)
            cols = [0, 1, 2, 3, 4]
            df_anno_2d = df_anno_2d[cols]
            for col in cols:
                new_cols_2d = df_anno_2d[col].str.replace(' ', '').str.split(',', expand=True)
                df_anno_2d[[str(col) + '_u', str(col) + '_v']] = new_cols_2d
            df_anno_2d = df_anno_2d[df_anno_2d.columns[5:]]
            df_anno_2d = np.array(df_anno_2d, dtype='float32').reshape([df_anno_2d.shape[0], -1, 2])
            self.anno_2d_depth.extend(df_anno_2d)

            fn_anno_3d = os.path.join(self.data_root, 'data', fd, 'annotations', 'my_' + fd + '3D.txt')
            df_anno_3d = pd.read_table(fn_anno_3d, sep=';', header=None)
            cols = [0, 1, 2, 3, 4]
            df_anno_3d = df_anno_3d[cols]

            for col in cols:
                new_cols_3d = df_anno_3d[col].str.replace(' ', '').str.split(',', expand=True)
                df_anno_3d[[str(col) + '_x', str(col) + '_y', str(col) + '_z']] = new_cols_3d
            df_anno_3d = df_anno_3d[df_anno_3d.columns[5:]]
            df_anno_3d = np.array(df_anno_3d, dtype='float32').reshape([df_anno_3d.shape[0], -1, 3])  # N*5*3
            self.joints.extend(df_anno_3d)

        self.joints = np.array(self.joints)
        for i in range(len(self.joints)):
            b = np.where(self.joints[i][:, 2:].squeeze() == 32001)
            self.joints[i][b] = np.array([np.nan, np.nan, np.nan])

            center = handutils.get_annot_center(self.DO_PRED_2D[i])
            my_scale = handutils.get_ori_crop_scale(mask=False, mask_flag=False, side=None,
                                                    kp2d=self.DO_PRED_2D[i])

            center = center[np.newaxis, :]
            self.centers.append(center)

            my_scale = (np.atleast_1d(my_scale))[np.newaxis, :]
            self.my_scales.append(my_scale)

        self.joints = self.joints / 1000.0  # transfer mm to m
        self.joints = self.joints.tolist()

        self.centers = np.concatenate(self.centers, axis=0).astype(np.float32)  # (N, 1)
        self.my_scales = np.concatenate(self.my_scales, axis=0).astype(np.float32)  # (N, 1)

        if use_cache:
            full_info = {
                "clr_paths": self.clr_paths,
                "dep_paths": self.dep_paths,
                "joints": self.joints,
                "2d_depth": self.anno_2d_depth,
                "centers": self.centers,
                "my_scales": self.my_scales,
            }
            with open(cache_path, "wb") as fid:
                pickle.dump(full_info, fid)
                print("Wrote cache for dataset DexterObjectDataset {} to {}".format(
                    self.data_split, cache_path
                ))
        return

    def __len__(self):
        """for DexterObject Dataset total
        """
        return len(self.clr_paths)

    def __str__(self):
        info = "DexterObject {} set. lenth {}".format(
            self.data_split, len(self.clr_paths)
        )
        return colored(info, 'blue', attrs=['bold'])

    def _is_valid(self, clr, index):
        valid_data = isinstance(clr, (np.ndarray, PIL.Image.Image))

        if not valid_data:
            raise Exception("Encountered error processing DO[{}]".format(index))
        return valid_data

    def get_sample(self, index):

        flip = True if self.hand_side != 'left' else False

        clr_name = self.clr_paths[index]
        clr_path = os.path.join(self.data_root, 'data', clr_name)
        clr = Image.open(clr_path)
        self._is_valid(clr, index)

        ori_clr = clr.copy()

        dep_name = self.dep_paths[index]
        dep_path = os.path.join(self.data_root, 'data', dep_name)
        dep = Image.open(dep_path)
        self._is_valid(dep, index)

        # bbox = np.array(self.bboxes[self.bboxes['img_name'] == clr_name][['x', 'y', 'w', 'h']], dtype='int')[0]

        # keypoint order : thumb, index, middle, ring, little
        joint = self.joints[index].copy()  # 5*3
        joint = np.array(joint)

        joint_transform = joint.copy()
        joint_transform[:, 0] = -joint_transform[:, 0]
        joint_transform[:, 1] = -joint_transform[:, 1]

        joint_process = joint_transform.copy()

        anno_3d_h = np.vstack((joint_transform.T * 1000., np.ones((1, joint.copy().T.shape[1]))))
        anno_2d_h = np.matmul(self.M_color, anno_3d_h)
        anno_2d = anno_2d_h[:2, :] / anno_2d_h[2, :]
        anno_2d_clr = anno_2d.copy().T
        anno_2d_clr_ori = anno_2d_clr.copy()

        # cropped_clr = clr.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

        anno_2d_depth = self.anno_2d_depth[index]
        anno_2d_depth_flip = anno_2d_depth.copy()

        DO_pred_2d = self.DO_PRED_2D[index]
        DO_pred_2d_ori = DO_pred_2d.copy()

        center = self.centers[index].copy()
        my_scale = self.my_scales[index].copy()

        if flip:
            clr = clr.transpose(Image.FLIP_LEFT_RIGHT)
            # cropped_clr = cropped_clr.transpose(Image.FLIP_LEFT_RIGHT)
            dep = dep.transpose(Image.FLIP_LEFT_RIGHT)
            joint[:, 0] = -joint[:, 0]
            joint_process[:, 0] = -joint_process[:, 0]
            anno_2d_depth_flip[:, 0] = dep.size[0] - anno_2d_depth[:, 0]
            anno_2d_clr[:, 0] = clr.size[0] - anno_2d_clr[:, 0]
            DO_pred_2d[:, 0] = clr.size[0] - DO_pred_2d[:, 0]

            center[0] = clr.size[0] - center[0]

        ##########color and depth##################
        if self.vis:
            fig = plt.figure(figsize=[50, 50])

            plt.subplot(2, 5, 1)
            plt.imshow(ori_clr)
            plt.title('ori_Color+2D annotations')
            plt.plot(anno_2d_clr_ori[0, 0], anno_2d_clr_ori[0, 1], 'ro')
            plt.text(anno_2d_clr_ori[0][0], anno_2d_clr_ori[0][1], '0', color="w", fontsize=10)
            for p in range(1, anno_2d_clr_ori.shape[0]):
                plt.plot(anno_2d_clr_ori[p][0], anno_2d_clr_ori[p][1], 'bo')
                plt.text(anno_2d_clr_ori[p][0], anno_2d_clr_ori[p][1], '{0}'.format(p), color="w", fontsize=10)

            plt.subplot(2, 5, 2)
            plt.imshow(ori_clr)
            plt.title('ori_Color+predict 2D annotations')
            plt.plot(DO_pred_2d_ori[0, 0], DO_pred_2d_ori[0, 1], 'ro')
            plt.text(DO_pred_2d_ori[0][0], DO_pred_2d_ori[0][1], '0', color="w", fontsize=10)
            for p in range(1, DO_pred_2d_ori.shape[0]):
                plt.plot(DO_pred_2d_ori[p][0], DO_pred_2d_ori[p][1], 'bo')
                plt.text(DO_pred_2d_ori[p][0], DO_pred_2d_ori[p][1], '{0}'.format(p), color="w", fontsize=10)

            plt.subplot(2, 5, 3)
            ori_depth = cv2.cvtColor(cv2.imread(dep_path), cv2.COLOR_BGR2RGB)
            plt.imshow(ori_depth)
            plt.title('ori_Depth')
            plt.plot(anno_2d_depth[0, 0], anno_2d_depth[0, 1], 'ro')
            plt.text(anno_2d_depth[0][0], anno_2d_depth[0][1], '0', color="w", fontsize=10)
            for p in range(1, anno_2d_depth.shape[0]):
                plt.plot(anno_2d_depth[p][0], anno_2d_depth[p][1], 'bo')
                plt.text(anno_2d_depth[p][0], anno_2d_depth[p][1], '{0}'.format(p), color="w", fontsize=10)

            plt.subplot(2, 5, 4)
            plt.imshow(clr)
            plt.title('flip_Color')
            plt.plot(anno_2d_clr[:, :1], anno_2d_clr[:, 1:], 'ro')
            for p in range(anno_2d_clr.shape[0]):
                plt.plot(anno_2d_clr[p][0], anno_2d_clr[p][1], 'ro')
                plt.text(anno_2d_clr[p][0], anno_2d_clr[p][1], '{0}'.format(p), fontsize=2)

            for p in range(1, DO_pred_2d.shape[0]):
                plt.plot(DO_pred_2d[p][0], DO_pred_2d[p][1], 'bo')
                plt.text(DO_pred_2d[p][0], DO_pred_2d[p][1], '{0}'.format(p), color="w", fontsize=10)

            plt.subplot(2, 5, 5)
            dep_ = cv2.flip(cv2.cvtColor(cv2.imread(dep_path), cv2.COLOR_BGR2RGB), 1)
            plt.imshow(dep_)
            plt.title('flip_Depth')
            for p in range(anno_2d_depth_flip.shape[0]):
                plt.plot(anno_2d_depth_flip[p][0], anno_2d_depth_flip[p][1], 'ro')
                plt.text(anno_2d_depth_flip[p][0], anno_2d_depth_flip[p][1], '{0}'.format(p), color="y", fontsize=10)

            plt.subplot(2, 5, 6)
            plt.text(0.5, 1, 'index={}'.format(index), color="b", fontsize=10)
            plt.text(0, 0.5, '{}'.format(joint_process), color="b", fontsize=10)
            plt.text(0.1, 0.2, 'clr_name={}'.format(clr_name), color="b", fontsize=10)

            # plt.subplot(2, 5, 7)
            # plt.imshow(cropped_clr)
            # plt.title('Cropped_flip_clr')

            # plt.subplot(2, 5, 8)
            # transform_image_ = transforms.Compose([
            #     transforms.Resize((self.image_size, self.image_size), Image.ANTIALIAS)])
            # plt.imshow(transform_image_(cropped_clr.copy()))
            # plt.title('Cropped_flip_resize_clr')

            ax = fig.add_subplot(259, projection='3d')

            plt.plot(joint_process[0, 0], joint_process[0, 1], joint_process[0, 2], 'rD', label='thumb')
            plt.plot(joint_process[1, 0], joint_process[1, 1], joint_process[1, 2], 'y*', label='index')
            plt.plot(joint_process[2, 0], joint_process[2, 1], joint_process[2, 2], 'ys', label='middle')
            plt.plot(joint_process[3, 0], joint_process[3, 1], joint_process[3, 2], 'yo', label='ring')
            plt.plot(joint_process[4, 0], joint_process[4, 1], joint_process[4, 2], 'yv', label='little')

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.title('3D annotations')
            plt.legend()
            ax.view_init(-90, -90)

            plt.show()

        metas = {
            'index': index,
            'clr': clr,
            'tip': joint_process,
            "kp2d": DO_pred_2d,
            'center': center,
            'my_scale': my_scale,
        }

        return metas


def main():
    do = DexterObjectDataset(
        data_root="/home/chen/datasets/dexter+object/",
        data_split='test',
        hand_side='right',
        njoints=21,
        use_cache=False,
        vis=True
    )

    print("len(do)=", len(do))
    for idx in tqdm(range(len(do))):
        print("idx=", idx)
        data = do.get_sample(idx)


if __name__ == '__main__':
    main()
