import numpy as np
from .augfuns import *




def autolt(img , chose_matix , target, ext_matix = None):# chose_list.shape ==  chose_matix_weight.shape
    # _augment_list = augment_list()
    # ext = np.random.rand()
    # val = float(ext) * float(_augment_list[chose_list[target]][2] - _augment_list[chose_list[target]][1]) + _augment_list[chose_list[target]][1]
    # img = _augment_list[chose_list[target]][0](img , val)
    # return img
    
    _augment_list = augment_list()
    # 从选择列表中获取目标行
    target_row = chose_matix[target]
    max_d = target_row.sum()
    if max_d < 1:
        return img
    if isinstance(ext_matix,np.ndarray)  : 
        exts = ext_matix[target]/max_d
    else:
        exts = np.random.rand(len(target_row))/max_d

    for index in np.nonzero(target_row)[0]:
        op, minval, maxval = _augment_list[index]
        val = exts[index]*float(maxval - minval) + minval
        img = op(img, val)
    
    return img

def augment_list():  
    return [
        (Flip, 0, 1),
        (Mirror, 0, 1),
        (EdgeEnhance, 0, 1),
        (Detail, 0, 1),
        (Smooth, 0, 1),
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (GaussianBlur, 0, 2),
        (Rotate, 0, 30),
    ]

# def augment_list():  
#     return [
#         (Flip, 0, 1),
#         (Mirror, 0, 1),
#         (EdgeEnhance, 0, 1),
#         (Detail, 0, 1),
#         (Smooth, 0, 1),
#         (AutoContrast, 0, 1),
#         (Equalize, 0, 1),
#         (Invert, 0, 1),
#         (GaussianBlur, 0, 2),
#         (ResizeCrop,1, 1.5),
#         (Rotate, 0, 30),
#         (Posterize, 0, 4),
#         (Solarize, 0, 256),
#         (SolarizeAdd, 0, 110),
#         (Color, 0.1, 1.9),
#         (Contrast, 0.1, 1.9),
#         (Brightness, 0.1, 1.9),
#         (Sharpness, 0.1, 1.9),
#         (ShearX, 0., 0.3),
#         (ShearY, 0., 0.3),
#         (TranslateXabs, 0., 100),
#         (TranslateYabs, 0., 100),
#     ]

def get_num_aug_type():
    return len(augment_list())