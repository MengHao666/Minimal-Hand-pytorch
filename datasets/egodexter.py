# Copyright (c) Hao Meng. All Rights Reserved.
r"""
EgoDexterDataset
Real-time Hand Tracking under Occlusion from an Egocentric RGB-D Sensor, ICCV 2017
Link to dataset: http://handtracker.mpi-inf.mpg.de/projects/OccludedHands/EgoDexter.htm
"""

import os
import pickle

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL import Image
from termcolor import colored
from torchvision import transforms
from tqdm import tqdm

import config as cfg

CACHE_HOME = os.path.expanduser(cfg.DEFAULT_CACHE_DIR)


# all hands in EgoDexter are left hands
class EgoDexter(torch.utils.data.Dataset):
    def __init__(self,
                 data_root,
                 data_split='train',
                 hand_side='right',
                 njoints=21,
                 use_cache=True, vis=False):
        if not os.path.exists(data_root):
            raise ValueError("data_root: %s not exist" % data_root)

        self.name = 'eo'
        self.data_root = os.path.join(data_root, 'EgoDexter/data')
        self.data_split = data_split
        self.hand_side = hand_side
        self.clr_paths = []
        self.dep_paths = []
        self.color_on_dep_paths = []
        self.joints = []
        self.kp2ds = []
        self.njoints = njoints
        self.reslu = [480, 640]

        self.image_size = 128
        self.vis = vis

        if data_split != 'test':
            print("EgoDexterDataset here only for evaluation, no train set here !")
            return None

        self.color_intrisics = np.array([[617.173, 0, 315.453],
                                         [0, 617.173, 242.259],
                                         [0, 0, 1]])
        self.color_extrisics = np.array([[1.0000, 0.00090442, -0.0074, 20.2365],
                                         [-0.00071933, 0.9997, 0.0248, 1.2846],
                                         [0.0075, -0.0248, 0.9997, 5.7360]])

        self.M_color = np.matmul(self.color_intrisics, self.color_extrisics)

        self.transform_image = transforms.Compose([
            transforms.CenterCrop((480, 480)),
            transforms.Resize((128, 128), Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [1, 1, 1])
        ])

        self.cache_folder = os.path.join(CACHE_HOME, "my-test", "EgoDexter")
        os.makedirs(self.cache_folder, exist_ok=True)
        cache_path = os.path.join(
            self.cache_folder, "{}.pkl".format(self.data_split)
        )

        if os.path.exists(cache_path) and use_cache:
            with open(cache_path, "rb") as fid:
                annotations = pickle.load(fid)
                self.clr_paths = annotations["clr_paths"]
                self.dep_paths = annotations["dep_paths"]
                self.color_on_dep_paths = annotations["color_on_dep_paths"]
                self.anno_2d_depth = annotations["anno_2d_depth"]
                self.joints = annotations["joints"]
            print("EgoDexter Dataset {} gt loaded from {}".format(self.data_split, cache_path))
            return

        print("init EgoDexter Dataset {}, It may take a while at first time".format(data_split))
        for folder in sorted(os.listdir(self.data_root)):
            # print("folder=",folder)
            clr_images = os.listdir(os.path.join(self.data_root, folder, 'color'))
            clr_images = [os.path.join(self.data_root, folder, 'color', x) for x in clr_images]
            clr_images_sorted = sorted(clr_images)
            self.clr_paths.extend(clr_images_sorted)

            dep_images = os.listdir(os.path.join(self.data_root, folder, 'depth'))
            dep_images = [os.path.join(self.data_root, folder, 'depth', x) for x in dep_images]
            dep_images_sorted = sorted(dep_images)
            self.dep_paths.extend(dep_images_sorted)

            color_on_dep_images = os.listdir(os.path.join(self.data_root, folder, 'color_on_depth'))
            color_on_dep_images = [os.path.join(self.data_root, folder, 'color_on_depth', x) for x in
                                   color_on_dep_images]
            color_on_dep_images_sorted = sorted(color_on_dep_images)
            self.color_on_dep_paths.extend(color_on_dep_images_sorted)

        self.anno_2d_depth = get_2d_annotations(self.data_root)
        self.anno_2d_depth = np.array(self.anno_2d_depth)[:, :10].reshape(-1, 5, 2)

        self.anno_3d = get_3d_annotations(self.data_root)
        self.joints = np.array(self.anno_3d)[:, :15].reshape(-1, 5, 3) / 1000.0  # transfer mm to m

        if use_cache:
            full_info = {
                "clr_paths": self.clr_paths,
                "dep_paths": self.dep_paths,
                "color_on_dep_paths": self.color_on_dep_paths,
                "joints": self.joints,
                "anno_2d_depth": self.anno_2d_depth,
            }
            with open(cache_path, "wb") as fid:
                pickle.dump(full_info, fid)
                print("Wrote cache for dataset EgoDexter Dataset {} to {}".format(
                    self.data_split, cache_path
                ))
        return

    def __len__(self):
        """for EgoDexter Dataset total
        """
        return len(self.clr_paths)

    def __str__(self):
        info = "EgoDexter {} set. lenth {}".format(
            self.data_split, len(self.clr_paths)
        )
        return colored(info, 'blue', attrs=['bold'])

    def _is_valid(self, clr, index):
        valid_data = isinstance(clr, (np.ndarray, PIL.Image.Image))

        if not valid_data:
            raise Exception("Encountered error processing ED[{}]".format(index))
        return valid_data

    def __getitem__(self, index):

        flip = True if self.hand_side != 'left' else False

        clr = Image.open(self.clr_paths[index]).convert("RGB")
        self._is_valid(clr, index)
        ori_clr = clr.copy()

        dep = Image.open(self.dep_paths[index])
        self._is_valid(dep, index)
        ori_dep = dep.copy()

        clr_on_dep = Image.open(self.color_on_dep_paths[index])
        self._is_valid(clr_on_dep, index)
        ori_clr_on_dep = clr_on_dep.copy()

        anno_2d_depth = self.anno_2d_depth[index]
        ori_anno_2d_depth = anno_2d_depth.copy()

        joint = self.joints[index].copy()

        anno_3d_h = np.vstack((joint.copy().T * 1000., np.ones((1, joint.copy().T.shape[1]))))
        anno_2d_h = np.matmul(self.M_color, anno_3d_h)
        anno_2d = anno_2d_h[:2, :] / anno_2d_h[2, :]
        anno_2d_clr = anno_2d.copy().T

        anno_2d_clr_ori = anno_2d_clr.copy()
        if flip:
            clr = clr.transpose(Image.FLIP_LEFT_RIGHT)
            dep = dep.transpose(Image.FLIP_LEFT_RIGHT)
            clr_on_dep = clr_on_dep.transpose(Image.FLIP_LEFT_RIGHT)
            anno_2d_depth[:, 0] = clr.size[0] - anno_2d_depth[:, 0]
            anno_2d_clr[:, 0] = clr.size[0] - anno_2d_clr[:, 0]
            joint[:, 0] = -joint[:, 0]

        flip_clr = clr.copy()
        clr = self.transform_image(clr)

        if self.vis:
            fig = plt.figure(figsize=[50, 50])

            plt.subplot(2, 5, 1)
            plt.imshow(ori_clr)
            plt.title('ori_Color+2D annotations')
            plt.plot(anno_2d_clr_ori[0, 0], anno_2d_clr_ori[0, 1], 'ro', markersize=5)
            plt.text(anno_2d_clr_ori[0][0], anno_2d_clr_ori[0][1], '0', color="w", fontsize=7.5)
            for p in range(1, anno_2d_clr_ori.shape[0]):
                plt.plot(anno_2d_clr_ori[p][0], anno_2d_clr_ori[p][1], 'bo', markersize=5)
                plt.text(anno_2d_clr_ori[p][0], anno_2d_clr_ori[p][1], '{0}'.format(p), color="w", fontsize=7.5)

            plt.subplot(2, 5, 2)
            plt.imshow(ori_clr_on_dep)
            plt.title('ori_Color_on_Depth')
            plt.plot(ori_anno_2d_depth[0, 0], ori_anno_2d_depth[0, 1], 'ro')
            plt.text(ori_anno_2d_depth[0][0], ori_anno_2d_depth[0][1], '0', color="w", fontsize=7.5)
            for p in range(1, ori_anno_2d_depth.shape[0]):
                plt.plot(ori_anno_2d_depth[p][0], ori_anno_2d_depth[p][1], 'bo')
                plt.text(ori_anno_2d_depth[p][0], ori_anno_2d_depth[p][1], '{0}'.format(p), color="w", fontsize=7.5)

            plt.subplot(2, 5, 3)
            plt.imshow(ori_dep)
            plt.title('ori_Depth')
            plt.plot(ori_anno_2d_depth[0, 0], ori_anno_2d_depth[0, 1], 'ro')
            plt.text(ori_anno_2d_depth[0][0], ori_anno_2d_depth[0][1], '0', color="r", fontsize=7.5)
            for p in range(1, ori_anno_2d_depth.shape[0]):
                plt.plot(ori_anno_2d_depth[p][0], ori_anno_2d_depth[p][1], 'o')
                plt.text(ori_anno_2d_depth[p][0], ori_anno_2d_depth[p][1], '{0}'.format(p), color="y", fontsize=7.5)

            plt.subplot(2, 5, 4)
            plt.imshow(flip_clr)
            plt.plot(anno_2d_clr[0, 0], anno_2d_clr[0, 1], 'ro')
            plt.text(anno_2d_clr[0][0], anno_2d_clr[0][1], '0', color="w", fontsize=7.5)
            for p in range(1, anno_2d_clr.shape[0]):
                plt.plot(anno_2d_clr[p][0], anno_2d_clr[p][1], 'bo')
                plt.text(anno_2d_clr[p][0], anno_2d_clr[p][1], '{0}'.format(p), color="w", fontsize=7.5)
            plt.title('flip_clr+ 2D annotations')

            plt.subplot(2, 5, 5)
            plt.imshow(dep)
            plt.title('Depth')
            plt.plot(anno_2d_depth[0, 0], anno_2d_depth[0, 1], 'ro')
            plt.text(anno_2d_depth[0][0], anno_2d_depth[0][1], '0', color="r", fontsize=7.5)
            for p in range(1, anno_2d_depth.shape[0]):
                plt.plot(anno_2d_depth[p][0], anno_2d_depth[p][1], 'bo')
                plt.text(anno_2d_depth[p][0], anno_2d_depth[p][1], '{0}'.format(p), color="y", fontsize=7.5)

            plt.subplot(2, 5, 6)
            plt.text(0.5, 1, 'index={}'.format(index), color="b", fontsize=10)
            plt.text(0, 0.5, '{}'.format(joint), color="b", fontsize=10)
            plt.text(-0.2, 0.2, 'clr_name={}'.format(self.clr_paths[index].replace(self.data_root, "")), color="b",
                     fontsize=10)

            ax = fig.add_subplot(2, 5, 7, projection='3d')

            plt.plot(joint[0, 0], joint[0, 1], joint[0, 2], 'rD', label='thumb')
            plt.plot(joint[1, 0], joint[1, 1], joint[1, 2], 'y*', label='index')
            plt.plot(joint[2, 0], joint[2, 1], joint[2, 2], 'ys', label='middle')
            plt.plot(joint[3, 0], joint[3, 1], joint[3, 2], 'yo', label='ring')
            plt.plot(joint[4, 0], joint[4, 1], joint[4, 2], 'yv', label='little')

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.title('3D annotations')
            plt.legend()
            ax.view_init(-90, -90)

            plt.subplot(2, 5, 8)
            plt.imshow(clr_on_dep)
            plt.title('Color_on_Depth')
            plt.plot(anno_2d_depth[0, 0], anno_2d_depth[0, 1], 'ro')
            plt.text(anno_2d_depth[0][0], anno_2d_depth[0][1], '0', color="r", fontsize=7.5)
            for p in range(1, anno_2d_depth.shape[0]):
                plt.plot(anno_2d_depth[p][0], anno_2d_depth[p][1], 'bo')
                plt.text(anno_2d_depth[p][0], anno_2d_depth[p][1], '{0}'.format(p), color="w", fontsize=7.5)

            plt.subplot(2, 5, 9)
            transform_image_ = transforms.Compose([transforms.CenterCrop((480, 480))])
            plt.imshow(transform_image_(flip_clr.copy()))
            plt.title('Cropped_flip_clr')

            plt.subplot(2, 5, 10)
            transform_image_ = transforms.Compose([transforms.CenterCrop((480, 480)),
                                                   transforms.Resize((self.image_size, self.image_size),
                                                                     Image.ANTIALIAS)])
            plt.imshow(transform_image_(flip_clr.copy()))
            plt.title('Cropped_resized_flip_clr')

            plt.show()

        metas = {
            'index': index,
            'clr': clr,
            'tips': joint
        }
        return metas


def get_3d_annotations(root_dir):
    folders = sorted(os.listdir(root_dir))
    df_all = pd.DataFrame()
    for folder_name in folders:
        df = get_item_3d_annotations(root_dir, folder_name)
        df_all = pd.concat([df_all, df])

    df_all = df_all.reset_index(drop=True)
    for col in df_all.columns:
        df_all[col] = np.where(abs(df_all[col]) < 1e-6, np.nan, df_all[col])

    return df_all


def get_2d_annotations(root_dir):
    folders = sorted(os.listdir(root_dir))
    df_all = pd.DataFrame()
    for folder_name in folders:
        df = get_item_2d_annotations(root_dir, folder_name)
        df_all = pd.concat([df_all, df])

    df_all = df_all.reset_index(drop=True)
    for col in df_all.columns:
        df_all[col] = np.where(df_all[col] == -1, np.nan, df_all[col])

    df_all['not_nan'] = 1
    for col in df_all.columns[:10]:
        df_all['not_nan'] = df_all['not_nan'] * df_all[col].notnull()
    return df_all


def get_item_3d_annotations(root_dir, folder_name):
    df = os.path.join(root_dir, folder_name, 'my_annotation.txt_3D.txt')
    df = pd.read_table(df, sep=';', header=None)
    df = df[df.columns[:-1]]
    df.columns = ['thumb', 'index', 'middle', 'ring', 'little']

    for col in df.columns:
        df[col + '_x'] = df[col].str.split(',', expand=True)[0]
        df[col + '_y'] = df[col].str.split(',', expand=True)[1]
        df[col + '_z'] = df[col].str.split(',', expand=True)[2]
        new_cols = [x for x in df.columns if x.find('_') > 0]
    df = df[new_cols]

    for col in df.columns:
        df[col] = df[col].str.strip()
        df[col] = np.where(df[col] == '', np.nan, df[col])  # nan means incorrect or lacking annotation
        df[col] = df[col].astype(float)
    return df


def get_item_2d_annotations(root_dir, folder_name):
    df = os.path.join(root_dir, folder_name, 'annotation.txt')
    df = pd.read_table(df, sep=';', header=None)
    df = df[df.columns[:-1]]
    df.columns = ['thumb', 'index', 'middle', 'ring', 'little']

    for col in df.columns:
        df[col + '_x'] = df[col].str.split(',', expand=True)[0]
        df[col + '_y'] = df[col].str.split(',', expand=True)[1]
        new_cols = [x for x in df.columns if x.find('_') > 0]
    df = df[new_cols]

    for col in df.columns:
        df[col] = df[col].str.strip()
        df[col] = np.where(df[col] == '', np.nan, df[col])
        df[col] = df[col].astype(float)
    return df


def main():
    ed = EgoDexter(
        data_root="/home/chen/datasets/",
        data_split='test',
        hand_side='right',
        njoints=21,
        use_cache=False,
        vis=True
    )

    print("len(ed)=", len(ed))

    for id in tqdm(range(len(ed))):
        print("id=", id)
        data = ed[id]


if __name__ == '__main__':
    main()
