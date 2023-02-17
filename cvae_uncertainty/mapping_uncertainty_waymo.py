import os
import argparse

from copy import deepcopy
import numpy as np
import pickle

from numpy.random import choice
# from visualize_utils import *
# import mayavi.mlab as mlab
# from pcdet.ops.iou3d_nms import iou3d_nms_utils
# from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from pcdet.utils import common_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from tqdm import tqdm
# np.random.seed(2021)


with open('waymo/waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl', 'rb') as f:
    db_infos = pickle.load(f)

exp='waymo_exp20_gen'

def single_fold_data(fold_idx):
    print(f"############################# {fold_idx} ###############################")
    # tag=f'gen_fold{fold_idx}'
    tag=f'fold_{fold_idx}'
    epoch='40'
    result_data_list = []

    file_list = os.listdir(f'output/{exp}/{tag}/eval/epoch_{epoch}/val/default/final_result/data/')
    file_list = [f'output/{exp}/{tag}/eval/epoch_{epoch}/val/default/final_result/data/' + x for x in file_list if 'result_val' in x]

    print(f"For fold={fold_idx}, len of file_list= ", len(file_list))

    for file in tqdm(file_list):
        with open(file, 'rb') as f:
            result_data = pickle.load(f)
            result_data_list.append(result_data)

    key_l = []
    overlap_l = []
    variance_l = []
    pointnum_l = []

    # car_info = db_infos['Car']

    splits=KFold(n_splits=5,shuffle=True,random_state=42)
    train_idx, val_idx = [x for x in splits.split(np.arange(len(db_infos['Vehicle'])))][fold_idx]

    car_info = [db_infos['Vehicle'][idx] for idx in val_idx]    

    # import pdb;pdb.set_trace()
    # choices = np.random.choice(len(car_info), 1500)

    for index in tqdm(range(len(car_info))):
        info = car_info[index]

        # frame_id = info['image_idx']
        frame_id = info['sequence_name'] + "#" + str(info["sample_idx"])
        gt_idx = info['gt_idx']
        key = f'{frame_id}_{gt_idx}'

        # pc_path = 'waymo/' + info['path']
        # pc_data = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
        # pc_data_num = len(pc_data)

        try:
            # overlap_list = [r[key]['overlap'] for r in result_data_list]
            # over_lap_mean = np.mean(overlap_list)

            pred_box_list = [r[key]['pred_box'] for r in result_data_list]
            pred_boxes = np.array(pred_box_list) # n * 9
        except:
            print(f"not found key={key}")
            import pdb;pdb.set_trace()
            continue

        
        gt_box = result_data_list[0][key]['gt_box']
        gt_box_angle = gt_box[6]
        # import pdb;pdb.set_trace()
        pred_boxes[:, 6] = common_utils.limit_period(pred_boxes[:, 6] - gt_box_angle, 0, 2 * np.pi)

        # # according to the coordinates
        pred_boxes[:, 6] = np.sin(pred_boxes[:, 6])
        variance_list = np.var(pred_boxes[:, :7], axis=0)
        # variance_list[-1] *= 0.1
        # variance_mean = np.mean(variance_list)
        
        # #according to the overlap to gt
        # variance_mean = np.var(overlap_list)

        # #according to the iou
        # pred_boxes_tensor = torch.tensor(pred_boxes[:, :7], dtype=torch.float32).cuda()
        # iou_matrix = boxes_iou3d_gpu(pred_boxes_tensor, pred_boxes_tensor)
        # some=iou_matrix.cpu().numpy()
        # variance_mean=some.var()

        key_l.append(key)
        # pointnum_l.append(pc_data_num) # removed
        # overlap_l.append(over_lap_mean)
        # variance_l.append(variance_mean)
        variance_l.append(variance_list)
    return key_l, pointnum_l, overlap_l, variance_l

# key_l_all = []
# pointnum_l_all = []
# overlap_l_all = []
# variance_l_all = []

result_json = {}

for fold_idx in range(5):
    key_l, pointnum_l, overlap_l, variance_l = single_fold_data(fold_idx)

    for i in range(len(key_l)):
        result_json[key_l[i]] = variance_l[i]

        # pointnum_l_all.extend(pointnum_l)
        # variance_l_all.extend(variance_l)


output_file = 'output/uncertainty_dump/waymo_un_v3.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(result_json, f)
