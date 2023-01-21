###

# visualize the predicted result of kitti val/test set
import argparse

from copy import deepcopy
import numpy as np
import pickle
from visualize_utils import *
import mayavi.mlab as mlab
# from pcdet.ops.iou3d_nms import iou3d_nms_utils
# from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None, gt_labels=None, output_path=None):
    if not isinstance(points, np.ndarray):
        points = points.cpu().numpy()
    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = ref_boxes.cpu().numpy()
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes.cpu().numpy()
    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        ref_scores = ref_scores.cpu().numpy()
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.cpu().numpy()

    fig = visualize_pts(points)
    # fig = draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))

    if gt_boxes is not None and len(gt_boxes) > 0:
        # hit
        mask = (gt_labels == 1)
        corners3d = boxes_to_corners_3d(gt_boxes[mask, :])
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)#blue
        # miss
        mask = (gt_labels == 0)
        corners3d = boxes_to_corners_3d(gt_boxes[mask, :])
        fig = draw_corners3d(corners3d, fig=fig, color=(1, 0, 0), max_num=100)#red

    if ref_boxes is not None and len(ref_boxes) > 0:
        # hit
        mask = (ref_labels == 1)
        corners3d = boxes_to_corners_3d(ref_boxes[mask, :])
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 1, 0), max_num=100)#green
        # fa
        mask = (ref_labels == 0)
        corners3d = boxes_to_corners_3d(ref_boxes[mask, :])
        fig = draw_corners3d(corners3d, fig=fig, color=(1, 1, 0), max_num=100)#yellow
    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    # fig.scene.disable_render = False
    if output_path:
        mlab.savefig(output_path)
    return fig

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--result', type=str, default=None, help='specify the config for training')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # main()
    args = parse_config()

    # 读取单个数据
    # kitti_infos_train
    with open('kitti/kitti_dbinfos_val.pkl', 'rb') as f:
        db_infos = pickle.load(f)

    with open(args.result, 'rb') as f:
        result_data = pickle.load(f)



    # for index in range(10):
    #     db_infos['Car'][index]

    #     # show the points
    #     pc_path = 'kitti/' + db_infos['Car'][index]['path']

    #     import numpy as np
    #     pc_data = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
    #     print(pc_data.shape)

    #     box3d_lidar_ori = db_infos['Car'][index]['box3d_lidar']
    #     print(f"box3d_lidar_ori = {box3d_lidar_ori}")
    #     box3d_lidar = deepcopy(box3d_lidar_ori)
    #     box3d_lidar[:3] = 0

    #     draw_scenes(pc_data, gt_boxes=np.array([box3d_lidar]), gt_labels=np.array([1]))
    #     mlab.show()


    choices = np.random.choice(len(db_infos['Car']), 512)
    for index in choices:
        info = db_infos['Car'][index]

        frame_id = info['image_idx']
        gt_idx = info['gt_idx']
        key = f'{frame_id}_{gt_idx}'

        overlap = result_data[key]['overlap']
        pred_box = result_data[key]['pred_box']
        gt_box = result_data[key]['gt_box']

        # show the points
        pc_path = 'kitti/' + info['path']
        import numpy as np
        pc_data = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)

        if pc_data.shape[0] == 0:
            continue
        x_mean, y_mean, z_mean, _ = list(pc_data.mean(axis=0))
        pc_data[:, 0] = pc_data[:, 0] - x_mean
        pc_data[:, 1] = pc_data[:, 1] - y_mean
        pc_data[:, 2] = pc_data[:, 2] - z_mean

        # box3d_lidar_ori = info['box3d_lidar']
        # box3d_lidar_ori[0] = -x_mean
        # box3d_lidar_ori[1] = -y_mean
        # box3d_lidar_ori[2] = -z_mean

        # print(pc_data.shape)
        print(f' pc_data.shape = {pc_data.shape}, overlap={overlap},  ')
        print(f' gt_box = {gt_box}, pred_box={pred_box}')

        if overlap > 0.7:
            continue
            gt_label = 1
            ref_label = 1
        else:
            gt_label = 0
            ref_label = 0
        
        if pc_data.shape[0] < 100:
            continue

        # box3d_lidar_ori = info['box3d_lidar']
        # print(f"box3d_lidar_ori = {box3d_lidar_ori}")
        # box3d_lidar = deepcopy(box3d_lidar_ori)
        # box3d_lidar[:3] = 0


        draw_scenes(pc_data, gt_boxes=np.array([gt_box]), gt_labels=np.array([gt_label]), 
                                ref_boxes=np.array([pred_box]), ref_labels=np.array([ref_label]))
        mlab.show()


# /home/yifanzhang/workspace/cvae_uncertainty/output/exp1/default/eval/epoch_80/val/default/final_result/data/result.pkl