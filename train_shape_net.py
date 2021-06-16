import argparse
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from progress.bar import Bar
from tensorboardX.writer import SummaryWriter
from termcolor import cprint

from model import shape_net
from datasets import SIK1M
from losses import shape_loss
# select proper device to run
from utils import misc
from utils.eval.evalutils import AverageMeter
import numpy as np

writer = SummaryWriter('log')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
steps = 0



def print_args(args):
    opts = vars(args)
    cprint("{:>30}  Options  {}".format("=" * 15, "=" * 15), 'yellow')
    for k, v in sorted(opts.items()):
        print("{:>30}  :  {}".format(k, v))
    cprint("{:>30}  Options  {}".format("=" * 15, "=" * 15), 'yellow')


def main(args):
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    print_args(args)
    print("\nCREATE NETWORK")
    model = shape_net.ShapeNet(_mano_root='mano/models')
    model = model.to(device)

    criterion = shape_loss.SIKLoss(
        lambda_joint=0.0,
        lambda_shape=1.0
    )

    optimizer = torch.optim.Adam(
        [
            {
                'params': model.shapereg_layers.parameters(),
                'initial_lr': args.learning_rate
            },

        ],
        lr=args.learning_rate,
    )

    train_dataset = SIK1M.SIK1M(
        data_root=args.data_root,
        data_split="train"
    )

    val_dataset = SIK1M.SIK1M(
        data_root=args.data_root,
        data_split="test"
    )

    print("Total train dataset size: {}".format(len(train_dataset)))
    print("Total val dataset size: {}".format(len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    if args.evaluate or args.resume:
        shape_net.load_checkpoint(
            model, os.path.join(args.checkpoint, 'ckp_siknet_synth.pth.tar')
        )
        if args.evaluate:
            for params in model.invk_layers.parameters():
                params.requires_grad = False

    if args.evaluate:
        validate(val_loader, model, args=args)
        cprint('Eval All Done', 'yellow', attrs=['bold'])
        return 0

    model = torch.nn.DataParallel(model)
    print("\nUSING {} GPUs".format(torch.cuda.device_count()))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_decay_step, gamma=args.gamma,
        last_epoch=args.start_epoch
    )
    train_bone_len = []
    train_shape_l2 = []
    test_bone_len = []
    test_shape_l2 = []
    for epoch in range(args.start_epoch, args.epochs + 1):
        print('\nEpoch: %d' % (epoch + 1))
        for i in range(len(optimizer.param_groups)):
            print('group %d lr:' % i, optimizer.param_groups[i]['lr'])
        #############  trian for on epoch  ###############
        t1, t2 = train(
            train_loader,
            model,
            criterion,
            optimizer,
            args=args,
        )
        ##################################################
        shape_net.save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
            },
            checkpoint=args.checkpoint,
            filename='{}.pth.tar'.format(args.saved_prefix),
            snapshot=args.snapshot,
            is_best=False
        )
        t3, t4 = validate(val_loader, model, criterion, args)
        train_bone_len.append(t1)
        train_shape_l2.append(t2)
        test_bone_len.append(t3)
        test_shape_l2.append(t4)
        np.save('log/train_bone_len.npy', train_bone_len)
        np.save('log/test_bone_len.npy', test_bone_len)
        np.save('log/train_shape_l2.npy', train_shape_l2)
        np.save('log/test_shape_l2.npy', test_shape_l2)
        scheduler.step()
    cprint('All Done', 'yellow', attrs=['bold'])
    return 0  # end of main


def validate(val_loader, model, criterion, args):
    am_shape_l2 = AverageMeter()
    am_bone_len = AverageMeter()
    model.eval()
    bar = Bar('\033[33m Eval  \033[0m', max=len(val_loader))
    with torch.no_grad():
        for i, metas in enumerate(val_loader):
            results, targets, total_loss, losses = one_forward_pass(
                metas, model, criterion, args, train=True
            )
            am_shape_l2.update(losses['shape_reg'].item(), targets['batch_size'])
            am_bone_len.update(losses['bone_len'].item(), targets['batch_size'])
            bar.suffix = (
                '({batch}/{size}) '
                't: {total:}s | '
                'eta:{eta:}s | '
                'lN: {lossLen:.5f} | '
                'lL2: {lossL2:.5f} | '
            ).format(
                batch=i + 1,
                size=len(val_loader),
                total=bar.elapsed_td,
                eta=bar.eta_td,
                lossLen=am_bone_len.avg,
                lossL2=am_shape_l2.avg,
            )
            bar.next()
        bar.finish()
    return (am_bone_len.avg, am_shape_l2.avg)


def train(train_loader, model, criterion, optimizer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    am_shape_l2 = AverageMeter()
    am_bone_len = AverageMeter()

    last = time.time()
    # switch to trian
    model.train()
    bar = Bar('\033[31m Train \033[0m', max=len(train_loader))
    for i, metas in enumerate(train_loader):
        data_time.update(time.time() - last)
        results, targets, total_loss, losses = one_forward_pass(
            metas, model, criterion, args, train=True
        )
        global steps
        steps += 1
        writer.add_scalar('loss', total_loss.item(), steps)
        am_shape_l2.update(losses['shape_reg'].item(), targets['batch_size'])
        am_bone_len.update(losses['bone_len'].item(), targets['batch_size'])
        ''' backward and step '''
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        ''' progress '''
        batch_time.update(time.time() - last)
        last = time.time()
        bar.suffix = (
            '({batch}/{size}) '
            'd: {data:.2f}s | '
            'b: {bt:.2f}s | '
            't: {total:}s | '
            'eta:{eta:}s | '
            'lN: {lossLen:.5f} | '
            'lL2: {lossL2:.5f} | '
        ).format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            lossLen=am_bone_len.avg,
            lossL2=am_shape_l2.avg,
        )
        bar.next()
    bar.finish()
    return (am_bone_len.avg, am_shape_l2.avg)


def one_forward_pass(metas, model, criterion, args, train=True):
    ''' prepare targets '''
    rel_bone_len = metas['rel_bone_len'].float().to(device, non_blocking=True)
    targets = {
        'batch_size': rel_bone_len.shape[0],
        'rel_bone_len': rel_bone_len
    }
    ''' ----------------  Forward Pass  ---------------- '''
    results = model(rel_bone_len)
    ''' ----------------  Forward End   ---------------- '''

    total_loss = torch.Tensor([0]).cuda()
    losses = {}
    if not train:
        return results, targets, total_loss, losses

    ''' conpute losses '''
    total_loss, losses = criterion.compute_loss(results, targets)
    return results, targets, total_loss, losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch Train dl shape')
    # Miscs
    parser.add_argument(
        '-ckp',
        '--checkpoint',
        default='checkpoints',
        type=str,
        metavar='PATH',
        help='path to save checkpoint (default: checkpoint)'
    )

    parser.add_argument(
        '-dr',
        '--data_root',
        type=str,
        default='data',
        help='dataset root directory'
    )

    parser.add_argument(
        '-sp',
        '--saved_prefix',
        default='ckp_siknet_synth',
        type=str,
        metavar='PATH',
        help='path to save checkpoint (default: checkpoint)'
    )
    parser.add_argument(
        '--snapshot',
        default=1, type=int,
        help='save models for every #snapshot epochs (default: 1)'
    )

    parser.add_argument(
        '-e', '--evaluate',
        dest='evaluate',
        action='store_true',
        help='evaluate model on validation set'
    )

    parser.add_argument(
        '-r', '--resume',
        dest='resume',
        action='store_true',
        help='resume model on validation set'
    )

    # Training Parameters
    parser.add_argument(
        '-j', '--workers',
        default=8,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 8)'
    )
    parser.add_argument(
        '--epochs',
        default=150,
        type=int,
        metavar='N',
        help='number of total epochs to run'
    )
    parser.add_argument(
        '-se', '--start_epoch',
        default=0,
        type=int,
        metavar='N',
        help='manual epoch number (useful on restarts)'
    )
    parser.add_argument(
        '-b', '--train_batch',
        default=1024,
        type=int,
        metavar='N',
        help='train batchsize'
    )
    parser.add_argument(
        '-tb', '--test_batch',
        default=512,
        type=int,
        metavar='N',
        help='test batchsize'
    )

    parser.add_argument(
        '-lr', '--learning-rate',
        default=1.0e-4,
        type=float,
        metavar='LR',
        help='initial learning rate'
    )
    parser.add_argument(
        "--lr_decay_step",
        default=40,
        type=int,
        help="Epochs after which to decay learning rate",
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.1,
        help='LR is multiplied by gamma on schedule.'
    )
    main(parser.parse_args())
