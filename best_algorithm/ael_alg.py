# import numpy as np

# def get_aug_type(aug_weight, ACCs, History_ACCs, lats_chose_matix, lats_chose_exts, epoch):
#     cls_num, num_aug_type = History_ACCs.shape
    
#     importance = ACCs - History_ACCs.mean(axis=1) * epoch + np.sum(aug_weight, axis=1)
#     chosen_indices = np.argsort(importance)[::-1]
    
#     chose_matrix = np.zeros((cls_num, num_aug_type), dtype=bool)
#     chose_exts = np.random.rand(*lats_chose_exts.shape)
    
#     for idx in chosen_indices:
#         for i in range(num_aug_type):
#             if lats_chose_matix[idx, i] and aug_weight[idx, i] > np.percentile(aug_weight[:, i], 50):
#                 chose_matrix[idx, i] = True
#                 chose_exts[idx, i] = lats_chose_exts[idx, i]
#                 break
    
#     return chose_matrix,chose_exts



# import numpy as np

# def get_aug_type(aug_weight, ACCs, History_ACCs, lats_chose_matix, lats_chose_exts, epoch):
#     cls_num, num_aug_type = History_ACCs.shape
    
#     perf_improvements = (ACCs - History_ACCs.mean(axis=1)) / History_ACCs.mean(axis=1)
#     chosen_indices = np.argsort(perf_improvements)[::-1]
    
#     chose_matrix = np.zeros((cls_num, num_aug_type), dtype=bool)
#     chose_exts = np.zeros((cls_num, num_aug_type))
    
#     for idx, cidx in enumerate(chosen_indices):
#         for i in range(num_aug_type):
#             if lats_chose_matix[cidx, i]:
#                 if np.random.rand() < aug_weight[cidx, i] * lats_chose_exts[cidx, i]:
#                     chose_matrix[cidx, i] = True
#                     chose_exts[cidx, i] = lats_chose_exts[cidx, i]
#                 break
    
#     return chose_matrix,chose_exts


# import numpy as np

# def get_aug_type(aug_weight, ACCs, History_ACCs, lats_chose_matix, lats_chose_exts, epoch):
#     cls_num, num_aug_type = History_ACCs.shape
    
#     chose_matrix = np.zeros((cls_num, num_aug_type), dtype=bool)
#     chose_exts = np.zeros((cls_num, num_aug_type))
    
#     for cidx in range(cls_num):
#         diff_acc = np.sum(History_ACCs[cidx] * aug_weight[cidx]) - ACCs[cidx]
#         if diff_acc < 0:
#             sorted_indices = np.argsort(History_ACCs[cidx])
#             for index in sorted_indices:
#                 if not lats_chose_matix[cidx, index]:
#                     chose_matrix[cidx, index] = True
#                     chose_exts[cidx, index] = lats_chose_exts[cidx, index]
#                     break
    
#     return chose_matrix,chose_exts