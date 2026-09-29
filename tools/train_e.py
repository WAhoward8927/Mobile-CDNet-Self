import sys

from models.model import BaseNet
from models.bifa_mods import MobileCDNetBiFA
# [Opus 5.5] Phase-1 training script. Fixes vs. tools/train.py:
#  (1) pre/post channel order (ToTensorRGB), (2) resume keeps best F1 / EMA / best epoch,
#  (3) one data root for train/val/test + counts logged, (4) full seeding (torch/numpy/random/workers).
# Phase-1 recipe (architecture unchanged = author BaseNet): per-temporal colour jitter, pre-image shift +-4px,
#  rot90, poly LR, EMA of weights (model selection on EMA val F1), test evaluated once at the end.
import copy, json, random
import transforms_p1 as P1

sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel import gather
import torch.optim.lr_scheduler

import dataset as myDataLoader
import Transforms as myTransforms
from metric_tool import ConfuseMatrixMeter
import utils
import matplotlib.pyplot as plt

import os, time
import numpy as np
from argparse import ArgumentParser


def BCEDiceLoss(inputs, targets):
    # print(inputs.shape, targets.shape)
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    # print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
    return bce + 1 - dice


def BCE(inputs, targets):
    # print(inputs.shape, targets.shape)
    bce = F.binary_cross_entropy(inputs, targets)
    return bce


class EMA(object):
    def __init__(self, model, decay):
        self.decay = decay; self.updates = 0   # warm-up: d_t = min(decay, (1+t)/(10+t))
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1; d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        msd = model.state_dict()
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point: v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else: v.copy_(msd[k])


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s); random.seed(s)


@torch.no_grad()
def val(args, val_loader, model, epoch):
    model.eval()

    salEvalVal = ConfuseMatrixMeter(n_class=2)

    epoch_loss = []

    total_batches = len(val_loader)
    print(len(val_loader))
    for iter, batched_inputs in enumerate(val_loader):

        img, target = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]
        start_time = time.time()

        if args.onGPU == True:
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var = torch.autograd.Variable(target).float()

        # run the mdoel
        output = model(pre_img_var, post_img_var)
        loss = BCEDiceLoss(output, target_var)

        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

        # torch.cuda.synchronize()
        time_taken = time.time() - start_time

        epoch_loss.append(loss.data.item())

        # compute the confusion matrix
        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(pred, 0, dim=0)
        # salEvalVal.addBatch(pred, target_var)
        f1 = salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=target_var.cpu().numpy())
        if iter % 5 == 0:
            print('\r[%d/%d] F1: %3f loss: %.3f time: %.3f' % (iter, total_batches, f1, loss.data.item(), time_taken),
                  end='')

        if np.mod(iter, 200) == 1:
            vis_input = utils.make_numpy_grid(utils.de_norm(pre_img_var[0:8]))
            vis_input2 = utils.make_numpy_grid(utils.de_norm(post_img_var[0:8]))
            vis_pred = utils.make_numpy_grid(pred[0:8])
            vis_gt = utils.make_numpy_grid(target_var[0:8])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                args.vis_dir, 'val_' + str(epoch) + '_' + str(iter) + '.jpg')
            plt.imsave(file_name, vis)

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalVal.get_scores()

    return average_epoch_loss_val, scores


def train(args, train_loader, model, optimizer, epoch, max_batches, cur_iter=0, lr_factor=1., ema=None):
    # switch to train mode
    model.train()

    salEvalVal = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []

    total_batches = len(train_loader)

    for iter, batched_inputs in enumerate(train_loader):

        img, target = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        start_time = time.time()

        # adjust the learning rate
        lr = adjust_learning_rate(args, optimizer, epoch, iter + cur_iter, max_batches, lr_factor=lr_factor)

        if args.onGPU == True:
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var = torch.autograd.Variable(target).float()

        # run the model
        output = model(pre_img_var, post_img_var)
        loss = BCEDiceLoss(output, target_var)

        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema is not None: ema.update(model)

        epoch_loss.append(loss.data.item())
        time_taken = time.time() - start_time
        res_time = (max_batches * args.max_epochs - iter - cur_iter) * time_taken / 3600

        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(pred, 0, dim=0)

        with torch.no_grad():
            f1 = salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=target_var.cpu().numpy())

        if iter % 50 == 0:
            print('\riteration: [%d/%d] f1: %.3f lr: %.7f loss: %.3f time:%.3f h' % (
                iter + cur_iter, max_batches * args.max_epochs, f1, lr, loss.data.item(),
                res_time),
                  end='')

        if np.mod(iter, 200) == 1:
            vis_input = utils.make_numpy_grid(utils.de_norm(pre_img_var[0:8]))
            vis_input2 = utils.make_numpy_grid(utils.de_norm(post_img_var[0:8]))
            vis_pred = utils.make_numpy_grid(pred[0:8])
            vis_gt = utils.make_numpy_grid(target_var[0:8])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                args.vis_dir, 'train_' + str(epoch) + '_' + str(iter) + '.jpg')
            plt.imsave(file_name, vis)

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalVal.get_scores()

    return average_epoch_loss_train, scores, lr


def adjust_learning_rate(args, optimizer, epoch, iter, max_batches, lr_factor=1):
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step_loss))
    elif args.lr_mode == 'poly':
        cur_iter = iter
        max_iter = max_batches * args.max_epochs
        lr = args.lr * (1 - cur_iter * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))
    if epoch == 0 and iter < 200:
        lr = args.lr * 0.9 * (iter + 1) / 200 + 0.1 * args.lr  # warm_up
    lr *= lr_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def trainValidateSegmentation(args):
    torch.backends.cudnn.benchmark = True
    SEED = args.seed
    torch.manual_seed(SEED); torch.cuda.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model = BaseNet(3, 1) if args.arch == 'base' else MobileCDNetBiFA(args.arch)
    print('ARCH', args.arch, 'params', sum(p.numel() for p in model.parameters()), flush=True)

    args.savedir = args.savedir + '_' + args.file_root + '_iter_' + str(args.max_steps) + '_lr_' + str(args.lr) + '_seed' + str(SEED) + '/'
    args.vis_dir = args.savedir + '/Vis/'

    if os.environ.get('MOBILE_CDNET_DATA_ROOT'):
        args.file_root = os.environ['MOBILE_CDNET_DATA_ROOT']
    else:
      if args.file_root == 'LEVIR':
        args.file_root = 'H:\\penghaifeng\\LEVIR-CD'
      elif args.file_root == 'BCDD':
        args.file_root = 'H:\\penghaifeng\\BCDD'
      elif args.file_root == 'SYSU':
        args.file_root = 'H:\\penghaifeng\\SYSU-CD'
      elif args.file_root == 'CDD':
        args.file_root = '/home/guan/Documents/Datasets/ChangeDetection/CDD'
      elif args.file_root == 'quick_start':
        args.file_root = './samples'
      else:
        raise TypeError('%s has not defined' % args.file_root)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)

    if args.onGPU:
        model = model.cuda()

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters (excluding idr): ' + str(total_params))

    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
    # mean = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    # compose the data with transforms
    train_tf = []
    if args.color_jitter: train_tf.append(P1.PerTemporalColorJitter(p=0.8))
    if args.pre_shift > 0: train_tf.append(P1.RandomPreShift(max_shift=args.pre_shift, p=0.5))
    train_tf += [myTransforms.Normalize(mean=mean, std=std),
                 myTransforms.Scale(args.inWidth, args.inHeight),
                 myTransforms.RandomCropResize(int(7. / 224. * args.inWidth)),
                 myTransforms.RandomFlip()]
    if args.rot90: train_tf.append(P1.RandomRot90())
    train_tf += [myTransforms.RandomExchange(), P1.ToTensorRGB()]
    trainDataset_main = myTransforms.Compose(train_tf)
    print('Train transforms:', [type(t).__name__ for t in train_tf])

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        P1.ToTensorRGB()
    ])

    train_data = myDataLoader.Dataset("train", file_root=args.file_root, transform=trainDataset_main)

    trainLoader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(SEED), persistent_workers=args.num_workers > 0
    )

    val_data = myDataLoader.Dataset("val", file_root=args.file_root, transform=valDataset)
    valLoader = torch.utils.data.DataLoader(
        val_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    test_data = myDataLoader.Dataset("test", file_root=args.file_root, transform=valDataset)
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    # whether use multi-scale training

    max_batches = len(trainLoader)

    print('For each epoch, we have {} batches'.format(max_batches))
    run_config = dict(vars(args)); run_config.update(seed=SEED, data_root=args.file_root, n_train=len(train_data),
                      n_val=len(val_data), n_test=len(test_data), baseline_paper_test_F1=0.9451,
                      model_selection='best val F1 of ' + ('EMA' if args.ema_decay > 0 else 'raw') + ' weights; test evaluated once at end')
    json.dump(run_config, open(os.path.join(args.savedir, 'run_config.json'), 'w'), indent=2, default=str)
    print('RUN_CONFIG', json.dumps(run_config, default=str), flush=True)

    if args.onGPU:
        cudnn.benchmark = True

    args.max_epochs = int(np.ceil(args.max_steps / max_batches))
    start_epoch = 0
    cur_iter = 0
    max_F1_val = 0

    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    best_epoch = -1

    if args.resume:
        ck_path = os.path.join(args.savedir, 'checkpoint.pth.tar')
        if os.path.isfile(ck_path):
            checkpoint = torch.load(ck_path, map_location='cpu', weights_only=False)
            start_epoch = checkpoint['epoch']; cur_iter = start_epoch * len(trainLoader)
            model.load_state_dict(checkpoint['state_dict']); optimizer.load_state_dict(checkpoint['optimizer'])
            if ema is not None and checkpoint.get('ema_state_dict') is not None: ema.model.load_state_dict(checkpoint['ema_state_dict']); ema.updates = cur_iter
            max_F1_val = checkpoint.get('max_F1_val', 0); best_epoch = checkpoint.get('best_epoch', -1)   # FIX: keep best across resume
            print("=> resumed from epoch %d (best val F1 %.4f @ %d)" % (start_epoch, max_F1_val, best_epoch))
        else:
            print("=> no checkpoint found at '{}', starting fresh".format(ck_path))

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_params)))
        logger.write(
            "\n%s\t%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Kappa (val)', 'IoU (val)', 'F1 (val)', 'R (val)', 'P (val)'))
    logger.flush()

    for epoch in range(start_epoch, args.max_epochs):

        lossTr, score_tr, lr = \
            train(args, trainLoader, model, optimizer, epoch, max_batches, cur_iter, ema=ema)
        cur_iter += len(trainLoader)

        torch.cuda.empty_cache()

        # evaluate on validation set
        if epoch == 0:
            continue

        eval_model = ema.model if ema is not None else model
        lossVal, score_val = val(args, valLoader, eval_model, epoch)
        raw_F1 = val(args, valLoader, model, epoch)[1]['F1'] if (ema is not None and args.log_raw_val) else float('nan')
        torch.cuda.empty_cache()
        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (epoch, score_val['Kappa'], score_val['IoU'],
                                                                       score_val['F1'], score_val['recall'],
                                                                       score_val['precision']))
        logger.flush()
        model_file_name = os.path.join(args.savedir, 'best_model.pth')
        if max_F1_val <= score_val['F1']:
            max_F1_val = score_val['F1']; best_epoch = epoch
            torch.save(eval_model.state_dict(), model_file_name)
        with open(os.path.join(args.savedir, 'val_metrics.jsonl'), 'a') as f:
            f.write(json.dumps({'epoch': epoch, 'F1': float(score_val['F1']), 'P': float(score_val['precision']),
                                'R': float(score_val['recall']), 'IoU': float(score_val['IoU']), 'raw_F1': float(raw_F1),
                                'lr': lr, 'train_loss': float(lossTr)}) + '\n')
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema.model.state_dict() if ema is not None else None,
            'optimizer': optimizer.state_dict(),
            'max_F1_val': max_F1_val, 'best_epoch': best_epoch,
            'lossTr': lossTr, 'lossVal': lossVal, 'F_Tr': score_tr['F1'], 'F_val': score_val['F1'], 'lr': lr
        }, os.path.join(args.savedir, 'checkpoint.pth.tar'))
        print("\n[VAL] epoch %d | %s F1 %.4f | P %.4f | R %.4f | raw F1 %.4f | best %.4f @ %d | lr %.2e" % (
            epoch, 'EMA' if ema is not None else 'raw', score_val['F1'], score_val['precision'], score_val['recall'],
            raw_F1, max_F1_val, best_epoch, lr), flush=True)
        torch.cuda.empty_cache()
    model_file_name = os.path.join(args.savedir, 'best_model.pth')
    model.load_state_dict(torch.load(model_file_name, map_location='cpu', weights_only=True))

    loss_test, score_test = val(args, testLoader, model, 0)
    print("\nTest :\t Kappa (te) = %.4f\t IoU (te) = %.4f\t F1 (te) = %.4f\t R (te) = %.4f\t P (te) = %.4f" \
          % (score_test['Kappa'], score_test['IoU'], score_test['F1'], score_test['recall'], score_test['precision']))
    logger.write("\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % ('Test', score_test['Kappa'], score_test['IoU'],
                                                                   score_test['F1'], score_test['recall'],
                                                                   score_test['precision']))
    logger.flush()
    logger.close()
    res = {'test_F1': float(score_test['F1']), 'test_P': float(score_test['precision']), 'test_R': float(score_test['recall']),
           'test_IoU': float(score_test['IoU']), 'test_Kappa': float(score_test['Kappa']), 'best_val_F1': float(max_F1_val),
           'best_epoch': best_epoch, 'paper_test_F1': 0.9451, 'delta_vs_paper_pt': 100 * (float(score_test['F1']) - 0.9451)}
    json.dump(res, open(os.path.join(args.savedir, 'test_result.json'), 'w'), indent=2)
    print('TEST_RESULT', json.dumps(res), flush=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_root', default="LEVIR", help='Data directory | LEVIR | BCDD | SYSU ')
    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument('--arch', default='base', choices=['base', 'adff', 'bi', 'bi_adff'])
    parser.add_argument('--color_jitter', type=int, default=1, help='per-temporal colour jitter (0/1)')
    parser.add_argument('--pre_shift', type=int, default=4, help='max random shift of pre image in px (0 = off)')
    parser.add_argument('--rot90', type=int, default=1, help='random 90-degree rotation (0/1)')
    parser.add_argument('--ema_decay', type=float, default=0.9995, help='EMA decay (0 = off)')
    parser.add_argument('--log_raw_val', type=int, default=0, help='also evaluate raw (non-EMA) weights on val each epoch')
    parser.add_argument('--inWidth', type=int, default=256, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
    parser.add_argument('--max_steps', type=int, default=40000, help='Max. number of iterations')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy, step or poly')
    parser.add_argument('--savedir', default='H:\\penghaifeng\\A2Net-main2\\results', help='Directory to save the results')
    parser.add_argument('--resume', type=int, default=1, help='resume from <savedir>/checkpoint.pth.tar if it exists (0/1)')
    parser.add_argument('--logFile', default='trainValLog.txt',
                        help='File that stores the training and validation logs')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--weight', default='', type=str, help='pretrained weight, can be a non-strict copy')
    parser.add_argument('--ms', type=int, default=0, help='apply multi-scale training, default False')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    trainValidateSegmentation(args)
