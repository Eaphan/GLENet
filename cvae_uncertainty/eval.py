import argparse
import os

from copy import deepcopy
import numpy as np
import pickle

# from visualize_utils import *
# import mayavi.mlab as mlab
# from pcdet.ops.iou3d_nms import iou3d_nms_utils
# from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from pcdet.utils import common_utils
import os
import torch


with open('kitti/kitti_dbinfos_val.pkl', 'rb') as f:
    db_infos = pickle.load(f)

# import pdb;pdb.set_trace()

# 'output/exp2/noshift/eval/epoch_80/val/default/final_result/data/result_val1.pkl',

import sys

exp=sys.argv[1]
# tag='large_ratio'
tag=sys.argv[2]
# tag='epoch_400_aug'
epoch=sys.argv[3]


result_data_list = []

file_list = os.listdir(f'output/{exp}/{tag}/eval/epoch_{epoch}/val/default/final_result/data/')
file_list = [f'output/{exp}/{tag}/eval/epoch_{epoch}/val/default/final_result/data/' + x for x in file_list if 'result_val' in x]

# for file in [
#     f'output/{exp}/{tag}/eval/epoch_{epoch}/val/default/final_result/data/result_val1.pkl',
#     f'output/{exp}/{tag}/eval/epoch_{epoch}/val/default/final_result/data/result_val2.pkl',
#     f'output/{exp}/{tag}/eval/epoch_{epoch}/val/default/final_result/data/result_val3.pkl',
#     f'output/{exp}/{tag}/eval/epoch_{epoch}/val/default/final_result/data/result_val4.pkl',
#     f'output/{exp}/{tag}/eval/epoch_{epoch}/val/default/final_result/data/result_val5.pkl',
#     f'output/{exp}/{tag}/eval/epoch_{epoch}/val/default/final_result/data/result_val6.pkl',
#     f'output/{exp}/{tag}/eval/epoch_{epoch}/val/default/final_result/data/result_val7.pkl',
#     f'output/{exp}/{tag}/eval/epoch_{epoch}/val/default/final_result/data/result_val8.pkl',
#     f'output/{exp}/{tag}/eval/epoch_{epoch}/val/default/final_result/data/result_val9.pkl',
#     f'output/{exp}/{tag}/eval/epoch_{epoch}/val/default/final_result/data/result_val10.pkl'    
# ]:
for file in file_list:
    with open(file, 'rb') as f:
        result_data = pickle.load(f)
        result_data_list.append(result_data)

overlap_l = []
variance_l = []
pointnum_l = []

variance7_l = []
pred_box_l = []

# car_info = db_infos['Car'] + db_infos['Van']
car_info = db_infos['Car']
# car_info = db_infos['Van']
from tqdm import tqdm
# np.random.seed(2021)
# choices = np.random.choice(len(car_info), 1500)
# for index in tqdm(choices):

for index in tqdm(range(len(car_info))):
# for index in tqdm(range(10)):
    info = car_info[index]

    frame_id = info['image_idx']
    gt_idx = info['gt_idx']
    key = f'{frame_id}_{gt_idx}'

    pc_path = 'kitti/' + info['path']
    pc_data = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
    pc_data_num = len(pc_data)

    try:
        overlap_list = [r[key]['overlap'] for r in result_data_list]
        over_lap_mean = np.mean(overlap_list)
    except:
        print(f"not found key={key} data.shape={pc_data_num}")
        continue

    pred_box_list = [r[key]['pred_box'] for r in result_data_list]
    pred_boxes = np.array(pred_box_list) # n * 9
    
    gt_box = result_data_list[0][key]['gt_box']
    gt_box_angle = gt_box[6]

    pred_boxes[:, :6] = pred_boxes[:, :6] - np.array(gt_box[:6])
    pred_boxes[:, 6] = common_utils.limit_period(pred_boxes[:, 6] - gt_box_angle, 0, 2 * np.pi)

    # # according to the coordinates
    pred_boxes[:, 6] = np.sin(pred_boxes[:, 6])
    pred_box_l.append(pred_boxes)
    variance_list = np.var(pred_boxes[:, :7], axis=0)

    variance7_l.append(deepcopy(variance_list))

    # variance_list[-1] *= 0.1
    variance_mean = np.mean(variance_list)

    # variance_mean = variance_list[1]
    
    # #according to the overlap to gt
    # variance_mean = np.var(overlap_list)

    # #according to the iou
    # pred_boxes_tensor = torch.tensor(pred_boxes[:, :7], dtype=torch.float32).cuda()
    # iou_matrix = boxes_iou3d_gpu(pred_boxes_tensor, pred_boxes_tensor)
    # some=iou_matrix.cpu().numpy()
    # variance_mean=some.var()

    pointnum_l.append(pc_data_num)
    overlap_l.append(over_lap_mean)
    variance_l.append(variance_mean)

# variance7_l
from pcdet.utils.loss_utils import WeightedSmoothL1Loss
loss_func = WeightedSmoothL1Loss()


# import pdb;pdb.set_trace()
pred_box_l=np.array(pred_box_l)
pred_box_l = torch.tensor(pred_box_l[:, :, :7])

variance_l=np.array(variance7_l)
variance_l=variance_l[:, np.newaxis, :]
# ad hoc
variance_l=np.tile(variance_l, [1, pred_box_l.shape[1], 1])

# import pdb;pdb.set_trace()
variance_l = variance_l + 1e-6
loss = 0.5 * loss_func(pred_box_l, torch.zeros_like(pred_box_l))/variance_l + 0.5 * np.log(variance_l)
loss = loss.sum()/pred_box_l.shape[0]/pred_box_l.shape[1]
print("### loss = ", loss)
print(f"### exp = {exp}, tag={tag}, epoch={epoch}, file num={pred_box_l.shape[1]}, loss = {loss:.3f}")
exit(0)
