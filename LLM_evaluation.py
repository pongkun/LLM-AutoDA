import time
import numpy as np
from func_timeout import func_set_timeout
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from grakel.datasets.base import fetch_dataset
from functools import partial
from utils.config import *

import importlib
from joblib import Parallel, delayed
from base_traing_test import traing_test

class Evaluation():
    def __init__(self) -> None:
        # ----------------------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!--------------------------------
        args = parse_args()
        reproducibility(args.seed)
        args = dataset_argument(args)
        ckp_train_set = {'dataset':'cifar100','imb_ratio':100,'num_max':500 ,'epochs':200,'batch-size': 256,'aug_prob': 0.5,'loss_fn': 'bs', 'aug_type': 'autoaug_cifar','seed': 0 ,'AutoLT':True,'cutout':True}
        for k,v in ckp_train_set.items():
            setattr(args,k,v)
        # ----------------------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!--------------------------------
        self.args = args
        self.ckp_epoch = 50
        
    
    def evaluate(self):
        try:
            heuristic_module = importlib.import_module("LLM_alg")
            eva = importlib.reload(heuristic_module)   
            
            # ----------------------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!--------------------------------
            # ----------------------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!--------------------------------
            
            fitness = traing_test(self.args,self.ckp_epoch,eva.get_aug_type)
            return fitness
        except Exception as e:
            print("Error:",str(e))
            return None
