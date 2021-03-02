import torch
import torch.nn.functional as torch_f


class DetLoss:
    def __init__(
            self,
            lambda_hm=100,
            lambda_dm=1.0,
            lambda_lm=1.0,

    ):
        self.lambda_hm = lambda_hm
        self.lambda_dm = lambda_dm
        self.lambda_lm = lambda_lm

    def compute_loss(self, preds, targs, infos):

        hm_veil = infos['hm_veil']
        batch_size = infos['batch_size']
        flag = targs['flag_3d']
        batch_3d_size = flag.sum()

        flag = flag.bool()

        final_loss = torch.Tensor([0]).cuda()
        det_losses = {}

        pred_hm = preds['h_map']
        pred_dm = preds['d_map'][flag]
        pred_lm = preds['l_map'][flag]

        targ_hm = targs['hm']  # B*21*32*32

        targ_hm_tile = \
            targ_hm.unsqueeze(2).expand(targ_hm.size(0), targ_hm.size(1), 3, targ_hm.size(2), targ_hm.size(3),
                                        )[flag]  # B'*21*3*32*32
        targ_dm = targs['dm'][flag]
        targ_lm = targs['lm'][flag]

        # compute hmloss anyway
        hm_loss = torch.Tensor([0]).cuda()
        if self.lambda_hm:
            hm_veil = hm_veil.unsqueeze(-1)
            njoints = pred_hm.size(1)
            pred_hm = pred_hm.reshape((batch_size, njoints, -1)).split(1, 1)
            targ_hm = targ_hm.reshape((batch_size, njoints, -1)).split(1, 1)
            for idx in range(njoints):
                pred_hmapi = pred_hm[idx].squeeze()  # (B, 1, 1024)->(B, 1024)
                targ_hmi = targ_hm[idx].squeeze()
                hm_loss += 0.5 * torch_f.mse_loss(
                    pred_hmapi.mul(hm_veil[:, idx]),  # (B, 1024) mul (B, 1)
                    targ_hmi.mul(hm_veil[:, idx])
                )  # mse calculate the loss of every sample  (in fact it calculate minbacth_loss/32*32 )
            final_loss += self.lambda_hm * hm_loss
        det_losses["det_hm"] = hm_loss

        # compute dm loss
        loss_dm = torch.Tensor([0]).cuda()
        if self.lambda_dm:
            loss_dm = torch.norm(
                (pred_dm - targ_dm) * targ_hm_tile) / batch_3d_size  # loss of every sample
            final_loss += self.lambda_dm * loss_dm
        det_losses["det_dm"] = loss_dm

        # compute lm loss
        loss_lm = torch.Tensor([0]).cuda()
        if self.lambda_lm:
            loss_lm = torch.norm(
                (pred_lm - targ_lm) * targ_hm_tile) / batch_3d_size  # loss of every sample
            final_loss += self.lambda_lm * loss_lm
        det_losses["det_lm"] = loss_lm

        det_losses["det_total"] = final_loss

        return final_loss, det_losses, batch_3d_size
