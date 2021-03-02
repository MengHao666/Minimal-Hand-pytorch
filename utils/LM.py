# Copyright (c) Hao Meng. All Rights Reserved.
import time

import numpy as np
import torch
from manopth.manolayer import ManoLayer

from utils import bone


class LM_Solver():
    def __init__(self, num_Iter=500, th_beta=None, th_pose=None, lb_target=None,
                 weight=0.01):
        self.count = 0
        self.time_start = time.time()
        self.time_in_mano = 0
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

    def update(self, beta_):
        beta = beta_.copy()
        self.count += 1
        now = time.time()
        my_th_beta = torch.from_numpy(beta).float().reshape(1, 10)
        _, joints = self.mano_layer(self.th_pose, my_th_beta)
        self.time_in_mano = time.time() - now

        useful_lb = bone.caculate_length(joints, label="useful")
        lb_ref = useful_lb[6]
        return useful_lb, lb_ref

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

    def get_loss(self, beta_):
        beta = beta_.copy()

        lb, lb_ref = self.update(beta)
        lb = lb.reshape(15, 1)

        loss = np.linalg.norm(lb / lb_ref - self.lb_target) ** 2 + \
               self.weight * np.linalg.norm(beta) ** 2

        return loss

    def get_derivative(self, beta_, n):

        beta = beta_.copy()
        params1 = np.array(beta)
        params2 = np.array(beta)
        step = 0.01
        params1[n] += step
        params2[n] -= step

        res1 = self.get_loss(params1)
        res2 = self.get_loss(params2)

        d = (res1 - res2) / (2 * step)

        return d.ravel()

    # LM algorithm
    def LM(self):
        u = 1e-2
        v = 1.5
        beta = self.beta.reshape(10, 1)

        out_n = 1
        num_beta = np.shape(beta)[0]  # the number of beta
        # calculating the init Jocobian matrix
        Jacobian = np.zeros([out_n, beta.shape[0]])

        last_update = 0
        last_loss = 0
        for i in range(self.num_Iter):
            loss = self.get_loss(beta)

            if loss < self.minimal_loss:
                self.minimal_loss = loss
                self.best_beta = beta

            if abs(loss - last_loss) < self.threshold_stop:
                self.time_total = time.time() - self.time_start
                return beta

            for k in range(num_beta):
                Jacobian[:, k] = self.get_derivative(beta, k)

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
