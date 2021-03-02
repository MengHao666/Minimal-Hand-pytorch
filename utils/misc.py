import os
import shutil

import numpy as np
import scipy.io
import torch
from termcolor import colored, cprint

import utils.func as func
import copy


def print_args(args):
    opts = vars(args)
    cprint("{:>30}  Options  {}".format("=" * 15, "=" * 15), 'yellow')
    for k, v in sorted(opts.items()):
        print("{:>30}  :  {}".format(k, v))
    cprint("{:>30}  Options  {}".format("=" * 15, "=" * 15), 'yellow')


def param_count(net):
    return sum(p.numel() for p in net.parameters()) / 1e6




def out_loss_auc(
        loss_all_, auc_all_, acc_hm_all_, outpath
):
    loss_all = copy.deepcopy(loss_all_)
    acc_hm_all = copy.deepcopy(acc_hm_all_)
    auc_all = copy.deepcopy(auc_all_)

    for k, l in zip(loss_all.keys(), loss_all.values()):
        np.save(os.path.join(outpath, "{}.npy".format(k)), np.vstack((np.arange(1, len(l) + 1), np.array(l))).T)

    if len(acc_hm_all):
        for key ,value in acc_hm_all.items():
            acc_hm_all[key]=np.array(value)
        np.save(os.path.join(outpath, "acc_hm_all.npy"), acc_hm_all)


    if len(auc_all):
        for key ,value in auc_all.items():
            auc_all[key]=np.array(value)
        np.save(os.path.join(outpath, "auc_all.npy"), np.array(auc_all))


def saveloss(d):
    for k, v in zip(d.keys(), d.values()):
        mat = np.array(v)
        np.save(os.path.join("losses", "{}.npy".format(k)), mat)


def save_checkpoint(
        state,
        checkpoint='checkpoint',
        filename='checkpoint.pth',
        snapshot=None,
        # is_best=False
        is_best=None
):
    # preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    fileprefix = filename.split('.')[0]
    # torch.save(state, filepath)
    torch.save(state['model'].state_dict(), filepath)

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(
            filepath,
            os.path.join(
                checkpoint,
                '{}_{}.pth'.format(fileprefix, state['epoch'])
            )
        )

    [auc, best_acc] = is_best

    for key in auc.keys():
        if auc[key] > best_acc[key]:
            shutil.copyfile(
                filepath,
                os.path.join(
                    checkpoint,
                    '{}_{}best.pth'.format(fileprefix, key)
                )
            )


# def load_checkpoint(model, checkpoint):
#     name = checkpoint
#     checkpoint = torch.load(name)
#     pretrain_dict = clean_state_dict(checkpoint['state_dict'])
#     model_state = model.state_dict()
#     state = {}
#     for k, v in pretrain_dict.items():
#         if k in model_state:
#             state[k] = v
#         else:
#             print(k, ' is NOT in current model')
#     model_state.update(state)
#     model.load_state_dict(model_state)
#     print(colored('loaded {}'.format(name), 'cyan'))

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
    print(colored('loaded {}'.format(name), 'cyan'))


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


def save_pred(preds, checkpoint='checkpoint', filename='preds_valid.mat'):
    preds = func.to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    scipy.io.savemat(filepath, mdict={'preds': preds})


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        print("adjust learning rate to: %.3e" % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def adjust_learning_rate_in_group(optimizer, group_id, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        print("adjust learning rate of group %d to: %.3e" % (group_id, lr))
        optimizer.param_groups[group_id]['lr'] = lr
    return lr


def resume_learning_rate(optimizer, epoch, lr, schedule, gamma):
    for decay_id in schedule:
        if epoch > decay_id:
            lr *= gamma
    print("adjust learning rate to: %.3e" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def resume_learning_rate_in_group(optimizer, group_id, epoch, lr, schedule, gamma):
    for decay_id in schedule:
        if epoch > decay_id:
            lr *= gamma
    print("adjust learning rate of group %d to: %.3e" % (group_id, lr))
    optimizer.param_groups[group_id]['lr'] = lr
    return lr
