import argparse, os, shutil, time, random, math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torch.nn.functional as F

from datasets.cifar100 import *

from train.train import *
from train.validate import *

from models.net import *
from losses.loss import *

from utils.config import *
from utils.plot import *
from utils.common import make_imb_data, save_checkpoint, hms_string

from utils.logger import logger


def is_an_augfunc(ael_get_aug_type_fun):
    info = np.load('./info.npy',allow_pickle=True).item()
    temps_thistimes = info['ACCs_save']
    temps = info['History_ACCs_save']
    choices =  info['choices_save']
    epoch = 50
    ACCs =  temps_thistimes[epoch]
    History_ACCs = temps[epoch]
    lats_chose_matix = choices[epoch]
    lats_chose_exts = np.random.rand(*lats_chose_matix.shape)

    try:
        for i in range(0,201):
            ael_get_aug_type_fun(np.ones((100,10)),ACCs, History_ACCs, lats_chose_matix, lats_chose_exts,epoch=i)
    except Exception as e:
        print('base_traing_test.py line40 : ael_get_aug_type_fun not accpted testing with {}'.format(e))
        raise 'base_traing_test.py line40 : ael_get_aug_type_fun not accpted testing with {}'.format(e)


def traing_test(args,ckp_epoch,ael_get_aug_type_fun):
    is_an_augfunc(ael_get_aug_type_fun) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!         assert
    base_tring_info = np.load('info.npy',allow_pickle=True).item()
    ckp =  torch.load(f'base_ckp/{ckp_epoch}/checkpoint.pth.tar')

    #-----------------------------------------------------------------------------------------------------------------
    try:
        assert args.num_max <= 50000. / args.num_class
    except AssertionError:
        args.num_max = int(50000 / args.num_class)
    
    print(f'==> Preparing imbalanced CIFAR-100')
    # N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, args.num_class, args.imb_ratio)
    trainset, testset = get_cifar100(os.path.join(args.data_dir, 'cifar100/'), args)
    N_SAMPLES_PER_CLASS = trainset.img_num_list
    print("img num list : " , trainset.img_num_list)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last= False, pin_memory=True, sampler=None)
    
    #-----------------------------------------------------------------------------------------------------------------
    
    trainset.get_aug_type_fun = ael_get_aug_type_fun
    #-----------------------------------------------------------------------------------------------------------------
    
    
    if args.cmo:
        cls_num_list = N_SAMPLES_PER_CLASS
        cls_weight = 1.0 / (np.array(cls_num_list))
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        labels = trainloader.dataset.targets
        samples_weight = np.array([cls_weight[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(labels), replacement=True)
        weighted_trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=sampler)
    else:
        weighted_trainloader = None
    
    # Model
    print ("==> creating {}".format(args.network))
    model = get_model(args, N_SAMPLES_PER_CLASS)
    model.load_state_dict(ckp['state_dict'])
    train_criterion = get_loss(args, N_SAMPLES_PER_CLASS)
    if hasattr(train_criterion, "_hook_before_epoch"):
        print('train_criterion have fun named \"_hook_before_epoch\" i\'ll run it.')

    optimizer = get_optimizer(args, model)
    optimizer.load_state_dict(ckp['optimizer'])
    scheduler = get_scheduler(args,optimizer)

    teacher = load_model(args)
    train = get_train_fn(args)
    
    trainloader.dataset.aug_weight = base_tring_info['aug_weight_save'][ckp_epoch]
    trainloader.dataset.chose_aug = base_tring_info['choices_save'][ckp_epoch]
    trainloader.dataset.History_ACCs = base_tring_info['History_ACCs_save'][ckp_epoch]
    trainloader.dataset.ext_matix = base_tring_info['exts_save'][ckp_epoch]
    
    
    for epoch in range(ckp_epoch + 1,ckp_epoch + 20 + 1):
        
        lr = adjust_learning_rate(optimizer, epoch, scheduler, args)
        if hasattr(train_criterion, "_hook_before_epoch"):
            train_criterion._hook_before_epoch(epoch)

        train_loss,ACCs = train(args, trainloader, model, optimizer,train_criterion, epoch, weighted_trainloader, teacher)
        acc = ACCs.sum()/sum(N_SAMPLES_PER_CLASS)
        gap = abs(base_tring_info['traing_losses'][epoch] - train_loss)
        print('epoch:[{}] ,acc:[{}]'.format(epoch,acc))
        print('epoch:[{}] ,gap:[{}]'.format(epoch,gap))
    
    return 1./gap



if __name__ == '__main__':
    ckp_train_set = {'dataset':'cifar100','imb_ratio':100,'num_max':500 ,'epochs':200,'batch-size': 256,'aug_prob': 0.5,'loss_fn': 'bs', 'aug_type': 'autoaug_cifar','seed': 0 ,'AutoLT':True,'cutout':True}

    args = parse_args()
    reproducibility(args.seed)
    args = dataset_argument(args)
    for k,v in ckp_train_set.items():
        setattr(args,k,v)
    traing_test(args,50)


