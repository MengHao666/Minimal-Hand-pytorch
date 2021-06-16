import argparse

import numpy as np
import torch
from manopth import demo
from manopth import manolayer
from tqdm import tqdm

from utils import AIK, align, vis
from utils.eval.zimeval import EvalUtil


def recon_eval(op_shapes, pre_j3ds, gt_j3ds, visual, key):
    pose0 = torch.eye(3).repeat(1, 16, 1, 1)
    mano = manolayer.ManoLayer(flat_hand_mean=True,
                               side="right",
                               mano_root='mano/models',
                               use_pca=False,
                               root_rot_mode='rotmat',
                               joint_rot_mode='rotmat')

    j3d_recons = []
    evaluator = EvalUtil()
    for i in tqdm(range(pre_j3ds.shape[0])):
        j3d_pre = pre_j3ds[i]

        op_shape = torch.tensor(op_shapes[i]).float().unsqueeze(0)
        _, j3d_p0_ops = mano(pose0, op_shape)
        template = j3d_p0_ops.cpu().numpy().squeeze() / 1000.0  # template, m

        ratio = np.linalg.norm(template[9] - template[0]) / np.linalg.norm(j3d_pre[9] - j3d_pre[0])
        j3d_pre_process = j3d_pre * ratio  # template, m
        j3d_pre_process = j3d_pre_process - j3d_pre_process[0] + template[0]

        pose_R = AIK.adaptive_IK(template, j3d_pre_process)
        pose_R = torch.from_numpy(pose_R).float()

        #  reconstruction
        hand_verts, j3d_recon = mano(pose_R, op_shape.float())

        # visualization
        if visual:
            demo.display_hand({
                'verts': hand_verts.cpu(),
                'joints': j3d_recon.cpu()
            },
                mano_faces=mano.th_faces)

        j3d_recon = j3d_recon.cpu().numpy().squeeze() / 1000.
        j3d_recons.append(j3d_recon)

        # visualization
        if visual:
            vis.multi_plot3d([j3d_recon, j3d_pre_process], title=["recon", "pre"])
    j3d_recons = np.array(j3d_recons)
    gt_joint, j3d_recon_align_gt = align.global_align(gt_j3ds, j3d_recons, key=key)

    for targj, predj_a in zip(gt_joint, j3d_recon_align_gt):
        evaluator.feed(targj * 1000.0, predj_a * 1000.0)

    (
        _1, _2, _3,
        auc_all,
        pck_curve_all,
        thresholds
    ) = evaluator.get_measures(
        20, 50, 15
    )
    print("Reconstruction AUC all of {}_test_set is : {}".format(key, auc_all))


def main(args):
    path = args.path
    for key_i in args.dataset:
        print("load {}'s joint 3D".format(key_i))
        _path = "{}/{}_dl.npy".format(path, key_i)
        print('load {}'.format(_path))
        op_shapes = np.load(_path)
        pre_j3ds = np.load("{}/{}_pre_joints.npy".format(path, key_i))
        gt_j3ds = np.load("{}/{}_gt_joints.npy".format(path, key_i))
        recon_eval(op_shapes, pre_j3ds, gt_j3ds, args.visualize, key_i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=' get pose params. of mano model ')

    parser.add_argument(
        '-ds',
        "--dataset",
        nargs="+",
        default=['rhd', 'stb', 'do', 'eo'],
        type=list,
        help="sub datasets, should be listed in: [stb|rhd|do|eo]"
    )

    parser.add_argument(
        '-p',
        "--path",
        default="out_testset",
        type=str,
        help="path"
    )

    parser.add_argument(
        '-vis',
        '--visualize',
        action='store_true',
        help='visualize reconstruction result',
        default=False
    )

    main(parser.parse_args())
