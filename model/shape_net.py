# Copyright (c) Lixin YANG. All Rights Reserved.
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from manopth.manolayer import ManoLayer


class ShapeNet(nn.Module):
    def __init__(
            self,
            dropout=0,
            _mano_root='mano/models'
    ):
        super(ShapeNet, self).__init__()

        ''' shape '''
        hidden_neurons = [128, 256, 512, 256, 128]
        in_neurons = 15
        out_neurons = 10
        neurons = [in_neurons] + hidden_neurons

        shapereg_layers = []
        for layer_idx, (inps, outs) in enumerate(
                zip(neurons[:-1], neurons[1:])
        ):
            if dropout:
                shapereg_layers.append(nn.Dropout(p=dropout))
            shapereg_layers.append(nn.Linear(inps, outs))
            shapereg_layers.append(nn.ReLU())

        shapereg_layers.append(nn.Linear(neurons[-1], out_neurons))
        self.shapereg_layers = nn.Sequential(*shapereg_layers)
        args = {'flat_hand_mean': True, 'root_rot_mode': 'axisang',
                'ncomps': 45, 'mano_root': _mano_root,
                'no_pca': True, 'joint_rot_mode': 'axisang', 'side': 'right'}
        self.mano_layer = ManoLayer(flat_hand_mean=args['flat_hand_mean'],
                                    side=args['side'],
                                    mano_root=args['mano_root'],
                                    ncomps=args['ncomps'],
                                    use_pca=not args['no_pca'],
                                    root_rot_mode=args['root_rot_mode'],
                                    joint_rot_mode=args['joint_rot_mode']
                                    )

    def new_cal_ref_bone(self, _shape):
        parent_index = [0,
                        0, 1, 2,
                        0, 4, 5,
                        0, 7, 8,
                        0, 10, 11,
                        0, 13, 14
                        ]
        index = [0,
                 1, 2, 3,  # index
                 4, 5, 6,  # middle
                 7, 8, 9,  # pinky
                 10, 11, 12,  # ring
                 13, 14, 15]  # thumb
        reoder_index = [
            13, 14, 15,
            1, 2, 3,
            4, 5, 6,
            10, 11, 12,
            7, 8, 9]
        shape = _shape
        th_v_shaped = torch.matmul(self.mano_layer.th_shapedirs,
                                   shape.transpose(1, 0)).permute(2, 0, 1) \
                      + self.mano_layer.th_v_template
        th_j = torch.matmul(self.mano_layer.th_J_regressor, th_v_shaped)
        temp1 = th_j
        temp2 = th_j[:, parent_index, :]
        result = temp1 - temp2
        ref_len = th_j[:, [4], :] - th_j[:, [0], :]
        ref_len = torch.norm(ref_len, dim=-1, keepdim=True)
        result = torch.norm(result, dim=-1, keepdim=True)
        result = result / ref_len
        return torch.squeeze(result, dim=-1)[:, reoder_index]

    def forward(self, bone_len):
        beta = self.shapereg_layers(bone_len)
        beta = torch.tanh(beta)
        bone_len_hat = self.new_cal_ref_bone(beta)

        results = {
            'beta': beta,
            'bone_len_hat': bone_len_hat
        }
        return results

def save_checkpoint(
        state,
        checkpoint='checkpoint',
        filename='checkpoint.pth.tar',
        snapshot=None,
        is_best=False
):
    # preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    fileprefix = filename.split('.')[0]
    torch.save(state, filepath)

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(
            filepath,
            os.path.join(
                checkpoint,
                '{}_{}.pth.tar'.format(fileprefix, state['epoch'])
            )
        )

    if is_best:
        shutil.copyfile(
            filepath,
            os.path.join(
                checkpoint,
                '{}_best.pth.tar'.format(fileprefix)
            )
        )

def load_checkpoint(model, checkpoint):
    name = checkpoint
    checkpoint = torch.load(name)
    pretrain_dict = clean_state_dict(checkpoint['state_dict'])
    model_state = model.state_dict()
    state = {}
    for k, v in pretrain_dict.items():
        if k in model_state:
            state[k] = v
        else:
            print(k, ' is NOT in current model')
    model_state.update(state)
    model.load_state_dict(model_state)

def clean_state_dict(state_dict):
    """save a cleaned version of model without dict and DataParallel

    Arguments:
        state_dict {collections.OrderedDict} -- [description]

    Returns:
        clean_model {collections.OrderedDict} -- [description]
    """

    clean_model = state_dict
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    clean_model = OrderedDict()
    if any(key.startswith('module') for key in state_dict):
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            clean_model[name] = v
    else:
        return state_dict

    return clean_model

if __name__ == '__main__':
    input = torch.rand((10, 15))
    model = ShapeNet()
    out_put = model(input)
    loss = torch.mean(out_put)
    loss.backward()
