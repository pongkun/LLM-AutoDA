from __future__ import print_function

import argparse, os, shutil, time, random, math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torch.nn.functional as F
import json
import losses

from datasets.cifar100 import *

from train.train import *
from train.validate import *

from models.net import *
from losses.loss import *

from utils.config import *
from utils.plot import *
from utils.common import make_imb_data, save_checkpoint, hms_string

from utils.logger import logger

args = parse_args()
reproducibility(args.seed)
args = dataset_argument(args)
args.logger = logger(args)

best_acc = 0

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
        # print('base_traing_test.py line40 : ael_get_aug_type_fun not accpted testing with {}'.format(e))
        raise 'base_traing_test.py line40 : ael_get_aug_type_fun not accpted testing with {}'.format(e)


def get_ael_reinforcement(args):
    if not args.use_ael_reinforcement:
        return get_aug_type  
 
    with open(args.ael_reinforcement_url, 'r') as file:
        ael_reinforcement_url = json.load(file)
    code = ael_reinforcement_url['code']
    exec(code, globals())
    # 检查全局作用域中是否成功定义了新的get_aug_type函数
    if 'get_aug_type' in globals():
        # 返回新定义的get_aug_type函数引用
        return globals()['get_aug_type']
    else:
        raise ValueError("The code does not define a function named 'get_aug_type'")

# 确保在调用get_ael_reinforcement之前，args.use_ael_reinforcement, args.ael_reinforcement_url等已经被正确设置


def main():
    global best_acc

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
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True) 
    
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
    train_criterion = get_loss(args, N_SAMPLES_PER_CLASS)
    if hasattr(train_criterion, "_hook_before_epoch"):
        print('train_criterion have fun named \"_hook_before_epoch\" i\'ll run it.')
    criterion = nn.CrossEntropyLoss() # For test, validation 
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args,optimizer)

    teacher = load_model(args)

    train = get_train_fn(args)
    validate = get_valid_fn(args)
    trainloader.dataset.get_aug_type_fun=get_ael_reinforcement(args)
    start_time = time.time()
    
    test_accs,traing_losses = [],[]
    aug_weight_save = [copy.deepcopy(trainloader.dataset.aug_weight)]
    ACCs_save = []
    History_ACCs_save = [copy.deepcopy(trainloader.dataset.History_ACCs)]
    choices_save = [copy.deepcopy(trainloader.dataset.chose_aug)]
    exts_save = [copy.deepcopy(trainloader.dataset.ext_matix)]
    for epoch in range(args.epochs):
        
        lr = adjust_learning_rate(optimizer, epoch, scheduler, args)
        if hasattr(train_criterion, "_hook_before_epoch"):
            train_criterion._hook_before_epoch(epoch)

        train_loss,temp_thistime = train(args, trainloader, model, optimizer,train_criterion, epoch, weighted_trainloader, teacher) 
        test_loss, test_acc, test_cls = validate(args, testloader, model, criterion, N_SAMPLES_PER_CLASS,  num_class=args.num_class, mode='test Valid')
        
        if best_acc <= test_acc:
            best_acc = test_acc
            many_best = test_cls[0]
            med_best = test_cls[1]
            few_best = test_cls[2]
            # Save models
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, epoch + 1, args.out)

        if epoch in [50,60,70,100,110,150]:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, epoch + 1, f'base_ckp/{epoch}/')

        test_accs.append(test_acc)
        

        args.logger(f'Epoch: [{epoch+1} | {args.epochs}]', level=1)
        args.logger(f'[Train]\tLoss:\t{train_loss:.4f}', level=2)
        args.logger(f'[Test ]\tLoss:\t{test_loss:.4f}\tAcc:\t{test_acc:.4f}', level=2)
        args.logger(f'[Stats]\tMany:\t{test_cls[0]:.4f}\tMedium:\t{test_cls[1]:.4f}\tFew:\t{test_cls[2]:.4f}', level=2)
        args.logger(f'[Best ]\tAcc:\t{np.max(test_accs):.4f}\tMany:\t{100*many_best:.4f}\tMedium:\t{100*med_best:.4f}\tFew:\t{100*few_best:.4f}', level=2)
        args.logger(f'[Param]\tLR:\t{lr:.8f}', level=2)
        

        aug_weight_save.append(copy.deepcopy(trainloader.dataset.aug_weight))
        ACCs_save.append(temp_thistime)
        History_ACCs_save.append(copy.deepcopy(trainloader.dataset.History_ACCs))
        choices_save.append(copy.deepcopy(trainloader.dataset.chose_aug))
        exts_save.append(copy.deepcopy(trainloader.dataset.ext_matix))
        traing_losses.append(train_loss) 
             
    end_time = time.time()

    args.logger(f'Final performance...', level=1)
    args.logger(f'best bAcc (test):\t{np.max(test_accs)}', level=2)
    args.logger(f'best statistics:\tMany:\t{many_best}\tMed:\t{med_best}\tFew:\t{few_best}', level=2)
    args.logger(f'Training Time: {hms_string(end_time - start_time)}', level=1)

    # info = {
    #     'aug_weight_save':aug_weight_save,
    #     'ACCs_save':ACCs_save,
    #     'History_ACCs_save':History_ACCs_save,
    #     'choices_save':choices_save,
    #     'exts_save':exts_save,
    #     'traing_losses':traing_losses
    # }
    # np.save('info_base_bs_ir100.npy',info)

if __name__ == '__main__':
    main()