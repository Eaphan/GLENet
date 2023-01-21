import argparse

from copy import deepcopy
import numpy as np
import pickle

from numpy.random import choice
from visualize_utils import *
import mayavi.mlab as mlab
# from pcdet.ops.iou3d_nms import iou3d_nms_utils
# from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from pcdet.utils import common_utils
import matplotlib.pyplot as plt

from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

with open('kitti/kitti_dbinfos_val.pkl', 'rb') as f:
    db_infos = pickle.load(f)

# '/home/yifanzhang/workspace/cvae_uncertainty/output/exp2/noshift/eval/epoch_80/val/default/final_result/data/result_val1.pkl',

result_data_list = []
exp='exp20'
# tag='epoch_400'
# epoch='400'
tag='fsr_e1600'
# tag='epoch_400'
epoch='1600'
for file in [
    f'/home/yifanzhang/workspace/cvae_uncertainty/output/{exp}/{tag}/eval/epoch_{epoch}/val/default/final_result/data/result_val1.pkl',
    f'/home/yifanzhang/workspace/cvae_uncertainty/output/{exp}/{tag}/eval/epoch_{epoch}/val/default/final_result/data/result_val2.pkl',
    f'/home/yifanzhang/workspace/cvae_uncertainty/output/{exp}/{tag}/eval/epoch_{epoch}/val/default/final_result/data/result_val3.pkl',
    f'/home/yifanzhang/workspace/cvae_uncertainty/output/{exp}/{tag}/eval/epoch_{epoch}/val/default/final_result/data/result_val4.pkl',
    f'/home/yifanzhang/workspace/cvae_uncertainty/output/{exp}/{tag}/eval/epoch_{epoch}/val/default/final_result/data/result_val5.pkl'
]:

    with open(file, 'rb') as f:
        result_data = pickle.load(f)
        result_data_list.append(result_data)

overlap_l = []
variance_l = []
pointnum_l = []

variance7_l = []

car_info = db_infos['Car']
from tqdm import tqdm
np.random.seed(2021)
choices = np.random.choice(len(car_info), 1500)

for index in tqdm(choices):
    info = db_infos['Car'][index]


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
    # import pdb;pdb.set_trace()
    pred_boxes[:, 6] = common_utils.limit_period(pred_boxes[:, 6] - gt_box_angle, 0, 2 * np.pi)

    # # according to the coordinates
    pred_boxes[:, 6] = np.sin(pred_boxes[:, 6])
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



# pointnum_l = [200 if x>200 else x for x in pointnum_l]

overlap_l = [o for i,o in enumerate(overlap_l) if pointnum_l[i]<500]
variance_l = [o for i,o in enumerate(variance_l) if pointnum_l[i]<500]
variance7_l = [o for i,o in enumerate(variance7_l) if pointnum_l[i]<500]
pointnum_l = [o for i,o in enumerate(pointnum_l) if pointnum_l[i]<500]
variance7_l = np.array(variance7_l)


variance_l = [x if x<0.06 else 0.06 for x in variance_l]

print("###", min(variance_l), max(variance_l))

# variance_l = [np.exp(30*x) for x in variance_l]

# figure
# import pdb;pdb.set_trace()

# plt.figure()
# plt.subplot(1, 3, 1)

# plt.plot(overlap_l, variance_l, 'o')
# plt.xlabel('Overlap', fontsize=12)
# plt.ylabel('Variance', fontsize=12)
# # 绘制第二个图
# plt.subplot(1, 3, 2)
# plt.plot(pointnum_l, variance_l, 'o')
# plt.xlabel('PointNum', fontsize=12)
# plt.ylabel('Variance', fontsize=12)
# # 绘制第三个图
# plt.subplot(1, 3, 3)
# plt.plot(pointnum_l, overlap_l, 'o')
# plt.xlabel('PointNum', fontsize=12)
# plt.ylabel('Overlap', fontsize=12)
# plt.show()


# plt.figure()
# plt.subplot(2, 4, 1)
# plt.plot(pointnum_l, variance7_l[:, 0], 'o')
# plt.subplot(2, 4, 2)
# plt.plot(pointnum_l, variance7_l[:, 1], 'o')
# plt.subplot(2, 4, 3)
# plt.plot(pointnum_l, variance7_l[:, 2], 'o')
# plt.subplot(2, 4, 4)
# plt.plot(pointnum_l, variance7_l[:, 3], 'o')
# plt.subplot(2, 4, 5)
# plt.plot(pointnum_l, variance7_l[:, 4], 'o')
# plt.subplot(2, 4, 6)
# plt.plot(pointnum_l, variance7_l[:, 5], 'o')
# plt.subplot(2, 4, 7)
# plt.plot(pointnum_l, variance7_l[:, 6], 'o')
# plt.show()


# plt.figure()
# plt.subplot(2, 4, 1)
# plt.boxplot(variance7_l[:, 0], 'o')
# plt.subplot(2, 4, 2)
# plt.boxplot(variance7_l[:, 1], 'o')
# plt.subplot(2, 4, 3)
# plt.boxplot(variance7_l[:, 2], 'o')
# plt.subplot(2, 4, 4)
# plt.boxplot(variance7_l[:, 3], 'o')
# plt.subplot(2, 4, 5)
# plt.boxplot(variance7_l[:, 4], 'o')
# plt.subplot(2, 4, 6)
# plt.boxplot(variance7_l[:, 5], 'o')
# plt.subplot(2, 4, 7)
# plt.boxplot(variance7_l[:, 6], 'o')
# plt.show()

# import pdb;pdb.set_trace()
plt.figure()
variance7_l = np.array(variance7_l)
# variance7_l[variance7_l>0.15] = 0.15
plt.boxplot(variance7_l)
plt.xlabel('Dimensions', fontsize=27)
plt.ylabel('Variance', fontsize=27)
# plt.xticks(np.arange(1,8), ['x', 'y', 'z', 'l', 'w', 'h', r'$\sigma$'], fontsize=24)
plt.xticks(np.arange(1,8), [' ', ' ', ' ', ' ', ' ', ' ', ' '], fontsize=27)
plt.yticks(fontsize=21)
# plt.xlim([1, 7])

plt.show()
# variance mean [4.14986877e-03, 4.65869738e-03, 2.62521377e-04, 1.99590403e-03, 5.71257650e-05, 2.40119909e-04, 2.03212238e-02])



