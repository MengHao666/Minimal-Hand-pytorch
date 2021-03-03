import argparse

import numpy as np
from tqdm import tqdm

from utils import func, bone
from utils.LM import LM_Solver


def align_bone_len(opt_, pre_):
    opt = opt_.copy()
    pre = pre_.copy()

    opt_align = opt.copy()
    for i in range(opt.shape[0]):
        ratio = pre[i][6] / opt[i][6]
        opt_align[i] = ratio * opt_align[i]

    err = np.abs(opt_align - pre).mean(0)

    return err


def main(args):
    path=args.path
    for dataset in args.dataset:
        # load predictions (N*21*3)
        print("load {}'s joint 3D".format(dataset))
        pred_j3d = np.load("{}/{}_pre_joints.npy".format(path, dataset),allow_pickle=True)

        opt_shapes = []
        opt_bone_lens = []
        pre_useful_bone_lens = []

        # loop
        for pred in tqdm(pred_j3d):
            # 0 initialization
            pose, shape = func.initiate("zero")

            pre_useful_bone_len = bone.caculate_length(pred, label="useful")
            pre_useful_bone_lens.append(pre_useful_bone_len)

            # optimize here!
            solver = LM_Solver(num_Iter=500, th_beta=shape, th_pose=pose, lb_target=pre_useful_bone_len,
                               weight=args.weight)
            opt_shape = solver.LM()
            opt_shapes.append(opt_shape)

            opt_bone_len = solver.get_bones(opt_shape)
            opt_bone_lens.append(opt_bone_len)

            # plt.plot(solver.get_result(), 'r')
            # plt.show()

            # break

        opt_shapes = np.array(opt_shapes).reshape(-1, 10)
        opt_bone_lens = np.array(opt_bone_lens).reshape(-1, 15)
        pre_useful_bone_lens = np.array(pre_useful_bone_lens).reshape(-1, 15)

        np.save("{}/{}_shapes.npy".format(path, dataset, args.weight), opt_shapes)

        error = align_bone_len(opt_bone_lens, pre_useful_bone_lens)

        print("dataset:{} weight:{} ERR sum: {}".format(dataset, args.weight, error.sum()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='optimize shape params. of mano model ')

    # Dataset setting
    parser.add_argument(
        '-ds',
        "--dataset",
        nargs="+",
        default=['rhd', 'stb', 'do', 'eo'],
        type=str,
        help="sub datasets, should be listed in: [stb|rhd|do|eo]"
    )
    parser.add_argument(
        '-wt', '--weight',
        default=1e-5,
        type=float,
        metavar='weight',
        help='weight of L2 regularizer '
    )

    parser.add_argument(
        '-p',
        '--path',
        default='out_testset',
        type=str,
        metavar='data_root',
        help='directory')

    main(parser.parse_args())
