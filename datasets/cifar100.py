import numpy as np
from PIL import Image
import random

import torchvision
import torch

from torch.utils.data import Dataset

from torchvision.transforms import transforms
from aug.LLM_AutoLT import *
from aug.transforms import *
from aug.autoaug import *
from aug.randaug import *
from aug.others import *
import copy
def get_aug_type(aug_weight,ACCs, History_ACCs, lats_chose_matix, lats_chose_exts,epoch=0):
    cls_num,num_aug_type = History_ACCs.shape
    print(111)

    if epoch > 50 :
        for cidx in range(cls_num):
            indices = lats_chose_matix[cidx]
            assert indices.any() ,f'class index {cidx} has no chose_aug (num of aug must > 0)'
            upper_index = (ACCs[cidx] > History_ACCs[cidx][indices])
            aug_weight[cidx][indices][upper_index] += 1

            down_index = (ACCs[cidx] < History_ACCs[cidx][indices])
            aug_weight[cidx][indices][down_index] -= 1

        aug_weight = np.maximum(aug_weight, 1)

    chose_matix = np.zeros((cls_num ,num_aug_type)).astype(bool)
    chose_exts = np.random.rand(*lats_chose_exts.shape)
    aug_list = [i for i in range(num_aug_type)]
    for i in range(cls_num):
        indexes = random.choices(aug_list , weights = aug_weight[i , : ].tolist() , k = 1) #self.args.MAX_N
        for index in indexes:
            chose_matix[i][index] = True

    return chose_matix,chose_exts

    
def get_cifar100(root, args):
    transform_train, transform_val = get_transform(args.loss_fn, cutout = args.cutout)

    train_dataset = CIFAR100_train(root, args, imb_ratio = args.imb_ratio, train=True, transform = transform_train, aug_prob=args.aug_prob)
    test_dataset = CIFAR100_val(root, transform=transform_val)
    print (f"#Train: {len(train_dataset)}, #Test: {len(test_dataset)}")
    return train_dataset, test_dataset
    
def get_transforms(args):
    if 'autoaug_cifar' in args.aug_type:
        print('autoaug_cifar')
        return  transforms.Compose([CIFAR10Policy()])
    elif 'autoaug_svhn' in args.aug_type:
        print('autoaug_svhn')
        return transforms.Compose([SVHNPolicy()])
    elif 'autoaug_imagenet' in args.aug_type:
        print('autoaug_imagenet')
        return transforms.Compose([ImageNetPolicy()])
    elif 'dada_cifar' in args.aug_type:
        print('dada_cifar')
        return transforms.Compose([dada_cifar()])
    elif 'dada_imagenet' in args.aug_type:
        print('dada_imagenet')
        return transforms.Compose([dada_imagenet()])
    elif 'faa_cifar' in args.aug_type:
        print('faa_cifar')
        return transforms.Compose([faa_cifar()])
    elif 'faa_imagenet' in args.aug_type:
        print('faa_imagenet')
        return transforms.Compose([faa_imagenet()])
    elif 'randaug' in args.aug_type:
        print('randaug')
        return transforms.Compose([RandAugment(2, 14)])
    elif 'none' in args.aug_type:
        return transforms.Compose([])
    else:
        raise NotImplementedError

class test_CIFAR100(Dataset):
    def __init__(self, indices, state, cifar_dataset):
        self.indices = indices
        self.state = state
        self.dataset = cifar_dataset

    def __getitem__(self,idx):
        data, label, _ = self.dataset.get_item(self.indices[idx], self.state[idx], train=False)
        return data, label, self.indices[idx], self.state[idx]
    
    def __len__(self):
        return len(self.indices)

class CIFAR100_train(torchvision.datasets.CIFAR100):
    def __init__(self, root , args, aug_prob, imb_type='exp', imb_ratio=100, train=True, transform=None, target_transform=None, download=True):
        super(CIFAR100_train,self).__init__(root, train=train, transform=transform, target_transform = target_transform, download= download)

        np.random.seed(0)
        
        self.args = args
        self.cls_num = 100
        self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, 1./imb_ratio)
        self.transform_train = transform
        self.gen_imbalanced_data(self.img_num_list)
        self.acc = np.zeros(self.cls_num)
        self.num_aug_type = get_num_aug_type()
        self.aug_weight = np.ones((self.cls_num , self.num_aug_type))
        self.ext_matix = None
        self.History_ACCs = np.zeros_like(self.aug_weight)
        self.AutoLT = args.AutoLT
        self.get_aug_type_fun = get_aug_type
        self.aug_transform = get_transforms(args)
        chose_aug = np.zeros((self.cls_num ,self.num_aug_type)).astype(bool)
        ext_matix = np.random.rand(self.cls_num ,self.num_aug_type)
        aug_list = [i for i in range(self.num_aug_type)]
        for i in range(self.cls_num):
            indexes = random.choices(aug_list , weights = self.aug_weight[i , : ].tolist() , k = 1) #self.args.MAX_N
            for index in indexes:
                chose_aug[i][index] = True

        self.chose_aug,self.ext_matix = chose_aug,ext_matix
        self.aug_prob = aug_prob
    
    def update_aug(self,ACCs,epoch = 0):
        aug_chose =  copy.deepcopy(self.chose_aug)
        self.chose_aug,self.ext_matix = self.get_aug_type_fun(self.aug_weight,ACCs,self.History_ACCs, self.chose_aug, self.ext_matix,epoch)
        for cidx in range(ACCs.shape[0]):
            self.History_ACCs[cidx][aug_chose[cidx]] = ACCs[cidx]
        

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls


    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            # print(selec_idx)
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def get_item(self, index, state, train=True):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if train:
            if len(self.transform_train) == 1:
                img = self.transform_train[0][0](img)
                img = self.aug_transform(img)
                if state == 1 and self.AutoLT:
                    img = autolt(img , self.chose_aug , target,self.ext_matix)
                img = self.transform_train[0][1](img)
                return img, target, index

            elif len(self.transform_train) == 2:
                img1 = self.transform_train[0][0](img)
                img1 = self.aug_transform(img1)
                if state == 1 and self.AutoLT:
                    img1 = autolt(img1 , self.chose_aug , target,self.ext_matix)
                img1 = self.transform_train[0][1](img1)
                img2 = self.transform_train[1][0](img)
                img2 = self.transform_train[1][1](img2)
                
                return (img1, img2), target, index
                
            elif len(self.transform_train) == 3:
                img1 = self.transform_train[0][0](img)
                img1 = self.aug_transform(img1)
                if state == 1 and self.AutoLT:
                    img1 = autolt(img1 , self.chose_aug , target,self.ext_matix)
                img1 = self.transform_train[0][1](img1)

                img2 = self.transform_train[1][0](img)
                img2 = self.transform_train[1][1](img2)
                
                img3 = self.transform_train[2][0](img)
                img3 = self.transform_train[2][1](img3)
                return (img1, img2, img3), target, index

        else:
            img = self.transform_train[0][0](img)
            img = self.aug_transform(img)
            if state == 1 and self.AutoLT:
                img = autolt(img , self.chose_aug , target,self.ext_matix)
            img = self.transform_train[0][1](img)
            return img, target, index
        
    def __getitem__(self, index):
        state = 1 if torch.rand(1) < self.aug_prob else 0
        img, target, index = self.get_item(index, state, train=True)
        return img, target, index

class CIFAR100_val(torchvision.datasets.CIFAR100):
    def __init__(self, root, transform=None, indexs=None,
                 target_transform=None, download=True):
        super(CIFAR100_val, self).__init__(root, train=False, transform=transform, target_transform=target_transform,download=download)
        
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = [Image.fromarray(img) for img in self.data]
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index
