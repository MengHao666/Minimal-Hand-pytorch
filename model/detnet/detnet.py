'''
detnet  based on PyTorch
this is modified from https://github.com/lingtengqiu/Minimal-Hand
'''
import sys

import torch

sys.path.append("./")
from torch import nn
from einops import rearrange, repeat
from model.helper import resnet50, conv3x3
import numpy as np


# my modification
def get_pose_tile_torch(N):
    pos_tile = np.expand_dims(
        np.stack(
            [
                np.tile(np.linspace(-1, 1, 32).reshape([1, 32]), [32, 1]),
                np.tile(np.linspace(-1, 1, 32).reshape([32, 1]), [1, 32])
            ], -1
        ), 0
    )
    pos_tile = np.tile(pos_tile, (N, 1, 1, 1))
    retv = torch.from_numpy(pos_tile).float()
    return rearrange(retv, 'b h w c -> b c h w')


class net_2d(nn.Module):
    def __init__(self, input_features, output_features, stride, joints=21):
        super().__init__()
        self.project = nn.Sequential(conv3x3(input_features, output_features, stride), nn.BatchNorm2d(output_features),
                                     nn.ReLU())

        self.prediction = nn.Conv2d(output_features, joints, 1, 1, 0)

    def forward(self, x):
        x = self.project(x)
        x = self.prediction(x).sigmoid()
        return x


class net_3d(nn.Module):
    def __init__(self, input_features, output_features, stride, joints=21, need_norm=False):
        super().__init__()
        self.need_norm = need_norm
        self.project = nn.Sequential(conv3x3(input_features, output_features, stride), nn.BatchNorm2d(output_features),
                                     nn.ReLU())
        self.prediction = nn.Conv2d(output_features, joints * 3, 1, 1, 0)

    def forward(self, x):
        x = self.prediction(self.project(x))

        dmap = rearrange(x, 'b (j l) h w -> b j l h w', l=3)

        return dmap


class detnet(nn.Module):
    def __init__(self, stacks=1):
        super().__init__()
        self.resnet50 = resnet50()

        self.hmap_0 = net_2d(258, 256, 1)
        self.dmap_0 = net_3d(279, 256, 1)
        self.lmap_0 = net_3d(342, 256, 1)
        self.stacks = stacks

    def forward(self, x):
        features = self.resnet50(x)

        device = x.device
        pos_tile = get_pose_tile_torch(features.shape[0]).to(device)

        x = torch.cat([features, pos_tile], dim=1)

        hmaps = []
        dmaps = []
        lmaps = []

        for _ in range(self.stacks):
            heat_map = self.hmap_0(x)
            hmaps.append(heat_map)
            x = torch.cat([x, heat_map], dim=1)

            dmap = self.dmap_0(x)
            dmaps.append(dmap)

            x = torch.cat([x, rearrange(dmap, 'b j l h w -> b (j l) h w')], dim=1)

            lmap = self.lmap_0(x)
            lmaps.append(lmap)
        hmap, dmap, lmap = hmaps[-1], dmaps[-1], lmaps[-1]

        uv, argmax = self.map_to_uv(hmap)

        delta = self.dmap_to_delta(dmap, argmax)
        xyz = self.lmap_to_xyz(lmap, argmax)

        det_result = {
            "h_map": hmap,
            "d_map": dmap,
            "l_map": lmap,
            "delta": delta,
            "xyz": xyz,
            "uv": uv
        }

        return det_result

    @property
    def pos(self):
        return self.__pos_tile

    @staticmethod
    def map_to_uv(hmap):
        b, j, h, w = hmap.shape
        hmap = rearrange(hmap, 'b j h w -> b j (h w)')
        argmax = torch.argmax(hmap, -1, keepdim=True)
        u = argmax // w
        v = argmax % w
        uv = torch.cat([u, v], dim=-1)

        return uv, argmax

    @staticmethod
    def dmap_to_delta(dmap, argmax):
        return detnet.lmap_to_xyz(dmap, argmax)

    @staticmethod
    def lmap_to_xyz(lmap, argmax):
        lmap = rearrange(lmap, 'b j l h w -> b j (h w) l')
        index = repeat(argmax, 'b j i -> b j i c', c=3)
        xyz = torch.gather(lmap, dim=2, index=index).squeeze(2)
        return xyz


if __name__ == '__main__':
    mydet = detnet()
    img_crop = torch.randn(10, 3, 128, 128)
    res = mydet(img_crop)

    hmap = res["h_map"]
    dmap = res["d_map"]
    lmap = res["l_map"]
    delta = res["delta"]
    xyz = res["xyz"]
    uv = res["uv"]

    print("hmap.shape=", hmap.shape)
    print("dmap.shape=", dmap.shape)
    print("lmap.shape=", lmap.shape)
    print("delta.shape=", delta.shape)
    print("xyz.shape=", xyz.shape)
    print("uv.shape=", uv.shape)
