import numpy as np

def get_aug_type(aug_weight, ACCs, History_ACCs, lats_chose_matix, lats_chose_exts, epoch):
    cls_num, num_aug_type = History_ACCs.shape
    
    chosen_indices = np.random.choice(num_aug_type, int(num_aug_type/2), replace=False)
    chose_matrix = np.zeros((cls_num, num_aug_type), dtype=bool)
    chose_exts = np.zeros((cls_num, num_aug_type))
    
    for cidx in range(cls_num):
        best_aug_idx = np.argmax(np.abs(ACCs[cidx] - History_ACCs[cidx]) * aug_weight[cidx])
        if best_aug_idx in chosen_indices:
            chose_matrix[cidx, best_aug_idx] = True
            chose_exts[cidx, best_aug_idx] = lats_chose_exts[cidx, best_aug_idx]
    
    return chose_matrix,chose_exts