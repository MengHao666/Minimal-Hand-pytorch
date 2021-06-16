import torch
from manopth import manolayer
import os


class DataSet:
    def __init__(self, device=torch.device('cpu'), _mano_root='mano/models'):
        args = {'flat_hand_mean': True, 'root_rot_mode': 'axisang',
                'ncomps': 45, 'mano_root': _mano_root,
                'no_pca': True, 'joint_rot_mode': 'axisang', 'side': 'right'}
        self.mano = manolayer.ManoLayer(flat_hand_mean=args['flat_hand_mean'],
                                        side=args['side'],
                                        mano_root=args['mano_root'],
                                        ncomps=args['ncomps'],
                                        use_pca=not args['no_pca'],
                                        root_rot_mode=args['root_rot_mode'],
                                        joint_rot_mode=args['joint_rot_mode']
                                        ).to(device)
        self.device = device

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
        shape = _shape.clone().detach()
        th_v_shaped = torch.matmul(self.mano.th_shapedirs,
                                   shape.transpose(1, 0)).permute(2, 0, 1) \
                      + self.mano.th_v_template
        th_j = torch.matmul(self.mano.th_J_regressor, th_v_shaped)
        temp1 = th_j.clone().detach()
        temp2 = th_j.clone().detach()[:, parent_index, :]
        result = temp1 - temp2
        ref_len = th_j[:, [4], :] - th_j[:, [0], :]
        ref_len = torch.norm(ref_len, dim=-1, keepdim=True)
        result = torch.norm(result, dim=-1, keepdim=True)
        result = result / ref_len
        return torch.squeeze(result, dim=-1)[:, reoder_index]

    def sample(self):
        shape = 3 * torch.randn((1, 10))
        result = self.new_cal_ref_bone(shape)
        return (result, shape)

    def batch_sample(self, batch_size):
        shape = 3 * torch.randn((batch_size, 10))
        result = self.new_cal_ref_bone(shape)
        return (result, shape)

    @staticmethod
    def cal_ref_bone(_Jtr):
        parent_index = [0,
                        0, 1, 2, 3,
                        0, 5, 6, 7,
                        0, 9, 10, 8,
                        0, 13, 14, 15,
                        0, 17, 18, 19
                        ]
        index = [1, 2, 3,
                 5, 6, 7,
                 9, 10, 11,
                 13, 14, 15,
                 17, 18, 19]
        temp1 = _Jtr.clone().detach()
        temp2 = _Jtr.clone().detach()[:, parent_index, :]
        result = temp1 - temp2
        result = result[:, index, :]
        ref_len = _Jtr[:, [9], :] - _Jtr[:, [0], :]
        ref_len = torch.norm(ref_len, dim=-1, keepdim=True)
        result = torch.norm(result, dim=-1, keepdim=True)
        # result = result / ref_len
        return torch.squeeze(result, dim=-1)


if __name__ == '__main__':
    dataset = DataSet()
    import numpy as np
    import tqdm

    Total_Num = 1000000
    NUM = 10000
    data_bone = np.zeros((Total_Num, 15))
    data_shape = np.zeros((Total_Num, 10))
    for i in tqdm.tqdm(range(Total_Num // NUM)):
        t1 = i * NUM
        t2 = t1 + NUM
        temp_1, temp_2 = dataset.batch_sample(NUM)
        data_bone[t1:t2] = temp_1
        data_shape[t1:t2] = temp_2
        print(t1, t2)

    save_dir = 'data'
    if os.path.exists(save_dir):
        pass
    else:
        os.mkdir(save_dir)
    np.save(os.path.join(save_dir, 'data_bone.npy'), data_bone)
    np.save(os.path.join(save_dir, 'data_shape.npy'), data_shape)
    print('*' * 10, 'test', '*' * 10)
    data_bone = np.load(os.path.join(save_dir, 'data_bone.npy'))
    data_shape = np.load(os.path.join(save_dir, 'data_shape.npy'))
    test_flag = 1
    for i in tqdm.tqdm(range(Total_Num // NUM)):
        t1 = i * NUM
        t2 = t1 + NUM
        test_shape = data_shape[t1:t2]
        test_shape = torch.tensor(test_shape, dtype=torch.float)
        test_bone = data_bone[t1:t2]
        temp_1 = dataset.new_cal_ref_bone(test_shape)
        flag = np.allclose(temp_1, test_bone)
        flag = int(flag)
        test_flag = test_flag * flag
    print(test_flag)
