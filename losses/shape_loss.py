import numpy as np
import torch
from torch import nn
import torch.nn.functional as torch_f



class SIKLoss:
    def __init__(
            self,
            lambda_joint=1.0,
            lambda_shape=1.0
    ):
        self.lambda_joint = lambda_joint
        self.lambda_shape = lambda_shape

    def compute_loss(self, preds, targs):
        batch_size = targs['batch_size']
        final_loss = torch.Tensor([0]).cuda()
        invk_losses = {}

        if self.lambda_joint:
            joint_loss = torch_f.mse_loss(
                1000 * preds['jointRS'] * targs['joint_bone'].unsqueeze(1),
                1000 * targs['jointRS'] * targs['joint_bone'].unsqueeze(1)
            )
            final_loss += self.lambda_joint * joint_loss
        else:
            joint_loss = None
        invk_losses["joint"] = joint_loss

        if self.lambda_shape:
            # shape_reg_loss = 10.0 * torch_f.mse_loss(
            #     preds["beta"],
            #     torch.zeros_like(preds["beta"])
            # )
            shape_reg_loss = torch.norm(preds['beta'], dim=-1, keepdim=True)
            shape_reg_loss = torch.pow(shape_reg_loss, 2.0)
            shape_reg_loss = torch.mean(shape_reg_loss)

            pred_rel_len = preds['bone_len_hat']

            # kin_len_loss = torch_f.mse_loss(
            #     pred_rel_len.reshape(batch_size, -1),
            #     targs['rel_bone_len'].reshape(batch_size, -1)
            # )
            kin_len_loss = torch.norm(pred_rel_len -
                                      targs['rel_bone_len'].reshape(batch_size, -1),
                                      dim=-1, keepdim=True)
            kin_len_loss = torch.pow(kin_len_loss, 2.0)
            kin_len_loss = torch.mean(kin_len_loss)
            shape_total_loss = kin_len_loss + 1e-3 * shape_reg_loss
            final_loss += self.lambda_shape * shape_total_loss
        else:
            shape_reg_loss, kin_len_loss = None, None
        invk_losses['shape_reg'] = shape_reg_loss
        invk_losses['bone_len'] = kin_len_loss

        return final_loss, invk_losses
