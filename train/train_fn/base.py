from __future__ import print_function

import time
import numpy as np
import torch
import torch.nn as nn

from aug.cutmix import *

from utils.accuracy import AverageMeter
from utils.common import Bar

import copy, time

from datasets.cifar100 import test_CIFAR100
import random

def train_base(args, trainloader, model, optimizer, criterion, epoch, weighted_trainloader, teacher = None):
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    
    # cls_num need to get
    cls_num = trainloader.dataset.cls_num
    # correct_all = torch.zeros(cls_num).int()
    output = []
    targets = []

    bar = Bar('Training', max=len(trainloader))

    if args.cmo and 3 < epoch < (args.epochs - 3):
        inverse_iter = iter(weighted_trainloader)

    for batch_idx, data_tuple in enumerate(trainloader):
        inputs_b = data_tuple[0]
        targets_b = data_tuple[1]
        indexs = data_tuple[2]
        targets += targets_b.tolist()


        # Measure data loading
        data_time.update(time.time() - end)
        batch_size = targets_b.size(0)
        
        if args.cmo and 3 < epoch < (args.epochs - 3):
            try:
                data_tuple_f = next(inverse_iter)
            except:
                inverse_iter = iter(weighted_trainloader)
                data_tuple_f = next(inverse_iter)

            inputs_f = data_tuple_f[0]
            targets_f = data_tuple_f[1]
            inputs_f = inputs_f[:len(inputs_b)]
            targets_f = targets_f[:len(targets_b)]
            inputs_f = inputs_f.cuda(non_blocking=True)
            targets_f = targets_f.cuda(non_blocking=True)

        inputs_b = inputs_b.cuda(non_blocking=True)
        targets_b = targets_b.cuda(non_blocking=True)


        r = np.random.rand(1)
        if args.cmo and 3 < epoch < (args.epochs - 3) and r < 0.5:
            inputs_b, lam = cutmix(inputs_f, inputs_b)
            outputs = model(inputs_b, None)
            loss = criterion(outputs, targets_b, epoch) * lam + criterion(outputs, targets_f, epoch) * (1.-lam)
        else:
            outputs = model(inputs_b, None)
            loss = criterion(outputs, targets_b, epoch)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update
        if isinstance(outputs,torch.Tensor): 
            output += torch.argmax(outputs,dim=1).tolist()
        else:
            output += torch.argmax(outputs['output'],dim=1).tolist()
        # correct = (outputs.max(dim=1)[1] == targets_b).int().detach().cpu()
        
        # record
        losses.update(loss.item(), targets_b.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        # plot
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                      'Loss: {loss:.4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    )
        bar.next()
    bar.finish()

    ACCS = torch.zeros(cls_num).int()
    for cidx in range(cls_num):
        class_pos = torch.where(torch.tensor(targets) == cidx)[0]
        ACCS[cidx] = (torch.tensor(output) == torch.tensor(targets))[class_pos].sum()

    # History_ACCs
    ACCS = ACCS.numpy()
    trainloader.dataset.update_aug(ACCS,epoch)
    
    return losses.avg,ACCS
