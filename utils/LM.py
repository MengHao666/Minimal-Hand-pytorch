# Copyright (c) Hao Meng. All Rights Reserved.
# import time

import numpy as np
import torch
from manopth.manolayer import ManoLayer

from utils import bone


class LM_Solver():
    def __init__(self, num_Iter=500, th_beta=None, th_pose=None, lb_target=None,
                 weight=0.01):
        self.count = 0
        # self.time_start = time.time()
        # self.time_in_mano = 0
        self.minimal_loss = 9999
        self.best_beta = np.zeros([10, 1])
        self.num_Iter = num_Iter

        self.th_beta = th_beta
        self.th_pose = th_pose

        self.beta = th_beta.numpy()
        self.pose = th_pose.numpy()

        self.mano_layer = ManoLayer(side="right",
                                    mano_root='mano/models', use_pca=False, flat_hand_mean=True)

        self.threshold_stop = 10 ** -13
        self.weight = weight
        self.residual_memory = []

        self.lb = np.zeros(21)

        _, self.joints = self.mano_layer(self.th_pose, self.th_beta)
        self.joints = self.joints.numpy().reshape(21, 3)

        self.lb_target = lb_target.reshape(15, 1)
        # self.test_time = 0

    def update(self, beta_):
        beta = beta_.copy()
        self.count += 1
        # now = time.time()
        my_th_beta = torch.from_numpy(beta).float().reshape(1, 10)
        _, joints = self.mano_layer(self.th_pose, my_th_beta)
        # self.time_in_mano = time.time() - now

        useful_lb = bone.caculate_length(joints, label="useful")
        lb_ref = useful_lb[6]
        return useful_lb, lb_ref

    def new_cal_ref_bone(self, _shape):
        # now = time.time()
        parent_index = [0,
                        0, 1, 2,
                        0, 4, 5,
                        0, 7, 8,
                        0, 10, 11,
                        0, 13, 14
                        ]
        # index = [0,
        #          1, 2, 3,  # index
        #          4, 5, 6,  # middle
        #          7, 8, 9,  # pinky
        #          10, 11, 12,  # ring
        #          13, 14, 15]  # thumb
        reoder_index = [
            13, 14, 15,
            1, 2, 3,
            4, 5, 6,
            10, 11, 12,
            7, 8, 9]
        shape = torch.Tensor(_shape.reshape((-1, 10)))
        th_v_shaped = torch.matmul(self.mano_layer.th_shapedirs,
                                   shape.transpose(1, 0)).permute(2, 0, 1) \
                      + self.mano_layer.th_v_template
        th_j = torch.matmul(self.mano_layer.th_J_regressor, th_v_shaped)
        temp1 = th_j.clone().detach()
        temp2 = th_j.clone().detach()[:, parent_index, :]
        result = temp1 - temp2
        result = torch.norm(result, dim=-1, keepdim=True)
        ref_len = result[:, [4]]
        result = result / ref_len
        # self.time_in_mano = time.time() - now
        return torch.squeeze(result, dim=-1)[:, reoder_index].cpu().numpy()

    def get_residual(self, beta_):
        beta = beta_.copy()
        lb, lb_ref = self.update(beta)
        lb = lb.reshape(45, 1)
        return lb / lb_ref - self.lb_target

    def get_count(self):
        return self.count

    def get_bones(self, beta_):
        beta = beta_.copy()
        lb, _ = self.update(beta)
        lb = lb.reshape(15, 1)

        return lb

    # Vectorization implementation
    def batch_get_l2_loss(self, beta_):
        weight = 1e-5
        beta = beta_.copy()
        temp = self.new_cal_ref_bone(beta)
        loss = np.transpose(temp)
        loss = np.linalg.norm(loss - self.lb_target, axis=0) ** 2 + \
               weight * np.linalg.norm(beta, axis=-1)
        return loss

    def new_get_derivative(self, beta_):
        # params: beta_ 10*1
        # return: 1*10
        beta = beta_.copy().reshape((1, 10))
        temp_shape = np.zeros((20, beta.shape[1]))  # 20*10
        step = 0.01
        for t2 in range(10):  # 位置
            t3 = 10 + t2
            temp_shape[t2] = beta.copy()
            temp_shape[t3] = beta.copy()
            temp_shape[t2, t2] += step
            temp_shape[t3, t2] -= step

        res = self.batch_get_l2_loss(temp_shape)
        d = res[0:10] - res[10:20]  # 10*1
        d = d.reshape((1, 10)) / (2 * step)
        return d

    # LM algorithm
    def LM(self):
        u = 1e-2
        v = 1.5
        beta = self.beta.reshape(10, 1)

        out_n = 1
        # num_beta = np.shape(beta)[0]  # the number of beta
        # calculating the init Jocobian matrix
        Jacobian = np.zeros([out_n, beta.shape[0]])

        last_update = 0
        last_loss = 0
        # self.test_time = 0
        for i in range(self.num_Iter):
            # loss = self.new_get_loss(beta)
            loss = self.batch_get_l2_loss(beta)
            loss = loss[0]
            if loss < self.minimal_loss:
                self.minimal_loss = loss
                self.best_beta = beta

            if abs(loss - last_loss) < self.threshold_stop:
                # self.time_total = time.time() - self.time_start
                return beta

            # for k in range(num_beta):
            #     Jacobian[:, k] = self.get_derivative(beta, k)
            Jacobian = self.new_get_derivative(beta)
            jtj = np.matmul(Jacobian.T, Jacobian)
            jtj = jtj + u * np.eye(jtj.shape[0])

            update = last_loss - loss
            delta = (np.matmul(np.linalg.inv(jtj), Jacobian.T) * loss)

            beta -= delta

            if update > last_update and update > 0:
                u /= v
            else:
                u *= v

            last_update = update
            last_loss = loss
            self.residual_memory.append(loss)

        return beta

    def get_result(self):
        return self.residual_memory
