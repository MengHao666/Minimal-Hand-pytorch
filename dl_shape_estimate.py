import os

import torch

import create_data
from model import shape_net

import numpy as np



def align_bone_len(opt_, pre_):
    opt = opt_.copy()
    pre = pre_.copy()

    opt_align = opt.copy()
    for i in range(opt.shape[0]):
        ratio = pre[i][6] / opt[i][6]
        opt_align[i] = ratio * opt_align[i]

    err = np.abs(opt_align - pre).mean(0)

    return err

def fun(_shape, _label, data_loader):
    # 计算相对骨骼长度
    shape = _shape.clone().detach()
    label = _label.detach().clone()
    # 根据shape计算相对骨骼长度
    X = data_loader.new_cal_ref_bone(shape)
    err = align_bone_len(X.cpu().numpy(), label.cpu().numpy())
    return  err.sum()

checkpoint = 'checkpoints'

model = shape_net.ShapeNet()
shape_net.load_checkpoint(
    model, os.path.join(checkpoint, 'ckp_siknet_synth_41.pth.tar')
)
for params in model.parameters():
    params.requires_grad = False

data_set = ['rhd', 'stb', 'do', 'eo']
temp_data = create_data.DataSet(_mano_root='mano/models')
for data in data_set:
    print('*' * 20)
    print('加载' + data + '数据集')
    print('*' * 20)
    # 加载预测
    pre_path = os.path.join('out_testset/', data + '_pre_joints.npy')
    temp = np.load(pre_path)
    temp = torch.Tensor(temp)
    _x = temp_data.cal_ref_bone(temp)
    # 模型回归shape
    Y = model(_x)
    Y = Y['beta']
    np.save('out_testset/' + data + '_dl.npy', Y.clone().detach().cpu().numpy())
    dl_err = fun(Y, _x, temp_data)
    print('回归误差:{}'.format(dl_err))

