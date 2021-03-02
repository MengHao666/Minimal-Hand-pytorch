import os
import argparse
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    path=args.out_path

    # losses in train
    lossD = np.load(os.path.join(path, "lossD.npy"))
    lossH = np.load(os.path.join(path, "lossH.npy"))
    lossL = np.load(os.path.join(path, "lossL.npy"))

    auc_all = np.load(os.path.join(path, "auc_all.npy"), allow_pickle=True).item()
    acc_hm_all = np.load(os.path.join(path, "acc_hm_all.npy"), allow_pickle=True).item()

    # rhd
    auc_all_rhd = np.array(auc_all['rhd'])
    acc_hm_rhd = np.array(acc_hm_all["rhd"])

    # stb
    auc_all_stb = np.array(auc_all['stb'])
    acc_hm_stb = np.array(acc_hm_all["stb"])

    # do
    auc_all_do = np.array(auc_all['do'])

    # eo
    auc_all_eo = np.array(auc_all['eo'])

    plt.figure(figsize=[50, 50])

    plt.subplot(2, 4, 1)
    plt.plot(lossH[:, :1], lossH[:, 1:], marker='o', label='lossH')
    plt.plot(lossD[:, :1], lossD[:, 1:], marker='*', label='lossD')
    plt.plot(lossL[:, :1], lossL[:, 1:], marker='h', label='lossL')
    plt.title("LOSSES")
    plt.legend(title='Losses Category:')

    # rhd
    plt.subplot(2, 4, 2)
    plt.plot(auc_all_rhd[:, :1], auc_all_rhd[:, 1:], marker='d')
    plt.title(
        "{}_test || (EPOCH={} , AUC={:0.4f})".format("RHD", np.argmax(auc_all_rhd[:, 1:]) + 1,
                                                     np.max(auc_all_rhd[:, 1:])))

    plt.subplot(2, 4, 3)
    plt.plot(acc_hm_rhd[:, :1], acc_hm_rhd[:, 1:], marker='d')
    plt.title(
        "{}_test || (EPOCH={} , ACC_HM={:0.4f})".format("RHD", np.argmax(acc_hm_rhd[:, 1:]) + 1,
                                                        np.max(acc_hm_rhd[:, 1:])))

    # stb
    plt.subplot(2, 4, 4)
    plt.plot(auc_all_stb[:, :1], auc_all_stb[:, 1:], marker='d')
    plt.title(
        "{}_test || (EPOCH={} , AUC={:0.4f})".format("STB", np.argmax(auc_all_stb[:, 1:]) + 1,
                                                     np.max(auc_all_stb[:, 1:])))

    plt.subplot(2, 4, 5)
    plt.plot(acc_hm_stb[:, :1], acc_hm_stb[:, 1:], marker='d')
    plt.title(
        "{}_test || (EPOCH={} , ACC_HM={:0.4f})".format("STB", np.argmax(acc_hm_stb[:, 1:]) + 1,
                                                        np.max(acc_hm_stb[:, 1:])))

    # do
    plt.subplot(2, 4, 6)
    plt.plot(auc_all_do[:, :1], auc_all_do[:, 1:], marker='d')
    plt.title(
        "{}_test || (EPOCH={} , AUC={:0.4f})".format("DO", np.argmax(auc_all_do[:, 1:] + 1), np.max(auc_all_do[:, 1:])))

    # eo
    plt.subplot(2, 4, 7)
    plt.plot(auc_all_eo[:, :1], auc_all_eo[:, 1:], marker='d')
    plt.title(
        "{}_test || (EPOCH={} , AUC={:0.4f})".format("EO", np.argmax(auc_all_eo[:, 1:]) + 1, np.max(auc_all_eo[:, 1:])))

    # plt.savefig("vis_train.png")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Result')
    parser.add_argument(
        '-p',
        '--out_path',
        type=str,
        default="out_loss_auc",
        help='ouput path'
    )
    main(parser.parse_args())

