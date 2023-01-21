###

# visualize the predicted result of kitti val/test set
import argparse
import SharedArray
import copy

from copy import deepcopy
import numpy as np
import pickle
from visualize_utils import *
import mayavi.mlab as mlab
# from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (N, 3 + C)
        angle: float, angle along z-axis, angle increases x ==> y
    Returns:
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    rot_matrix = np.array([
        [cosa,  sina, 0],
        [-sina, cosa, 0],
        [0, 0, 1]
    ]).astype(np.float32)
    points_rot = np.matmul(points[:, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, 3:]), axis=-1)
    return points_rot

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
    # with open('waymo/waymo_processed_data_v0_5_0_waymo_dbinfos_val_sampled_1.pkl', 'rb') as f:
    with open('waymo/waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl', 'rb') as f:
        db_infos = pickle.load(f)

    with open(args.result, 'rb') as f:
        result_data = pickle.load(f)

    # gt_database_data_key = sa_key = 'waymo_processed_data_v0_5_0_gt_database_val_sampled_1_global.npy'
    gt_database_data_key = sa_key = 'waymo_processed_data_v0_5_0_gt_database_train_sampled_1_global.npy'
    # choices = np.random.choice(len(db_infos['Vehicle']), 51200)
    # import pdb;pdb.set_trace()
    for index in range(len(db_infos['Vehicle'])):
        info = db_infos['Vehicle'][index]

        frame_id = info['sequence_name'] + "#" + str(info["sample_idx"])
        gt_id = info['gt_idx']
        key = f'{frame_id}_{gt_id}'

        if key not in result_data: # 'segment-13940755514149579648_821_157_841_157_with_camera_labels#10_21'
        # if key != 'segment-10017090168044687777_6380_000_6400_000_with_camera_labels#6_4':
            continue
        # print("### key=", key)

        box3d_lidar_ori = info['box3d_lidar']
        box3d_lidar = copy.deepcopy(box3d_lidar_ori)
        # box3d_lidar[6] = common_utils.limit_period(torch.tensor(box3d_lidar[6]), offset=0.5, period=np.pi/2).item()
        # trans_angle = box3d_lidar[6] - box3d_lidar_ori[6]
        azimuth = np.arctan2(box3d_lidar[0], box3d_lidar[1])
        new_azimuth = common_utils.limit_period(torch.tensor(azimuth), offset=0.5, period=np.pi/2).item()
        trans_angle = new_azimuth - azimuth


        # overlap = result_data[key]['overlap']
        # pred_box = result_data[key]['pred_box']
        # gt_box = result_data[key]['gt_box']
        box_pred = np.array(result_data[key]['pred_box'][:7]).reshape(-1, 7)
        gt_boxes = np.array(result_data[key]['gt_box']).reshape(-1, 7)

        if np.abs(trans_angle/(np.pi/2)%2-1) < 0.01:
            t_x, t_y, t_z, t_l, t_w, t_h, t_theta  = box_pred[0]
            box_pred[0] = [t_y, t_x, t_z, t_l, t_w, t_h, t_theta]
            t_x, t_y, t_z, t_l, t_w, t_h, t_theta  = gt_boxes[0]
            gt_boxes[0] = [t_y, t_x, t_z, t_l, t_w, t_h, t_theta]


        point_anchor_size = [4.7, 2.1, 1.7]
        dxa, dya, dza = point_anchor_size
        diagonal = np.sqrt(dxa ** 2 + dya ** 2)

        box_pred[:, 0] *= diagonal
        box_pred[:, 1] *= diagonal
        box_pred[:, 2] *= dza
        box_pred[:, 3] = np.exp(box_pred[:, 3]) * dxa
        box_pred[:, 4] = np.exp(box_pred[:, 4]) * dya
        box_pred[:, 5] = np.exp(box_pred[:, 5]) * dza

        gt_boxes[:, 0] *= diagonal
        gt_boxes[:, 1] *= diagonal
        gt_boxes[:, 2] *= dza
        gt_boxes[:, 3] = np.exp(gt_boxes[:, 3]) * dxa
        gt_boxes[:, 4] = np.exp(gt_boxes[:, 4]) * dya
        gt_boxes[:, 5] = np.exp(gt_boxes[:, 5]) * dza


        # show the points
        # pc_path = 'kitti/' + info['path']
        # pc_data = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
        gt_database_data = SharedArray.attach(f"shm://{gt_database_data_key}")
        gt_database_data.setflags(write=0)
        start_offset, end_offset = info['global_data_offset']
        pc_data = copy.deepcopy(gt_database_data[start_offset:end_offset])

        # 这里根据框的角度旋转

        pc_data[:, :3] += box3d_lidar[:3]
        pc_data = rotate_points_along_z(pc_data, trans_angle)

        # import pdb;pdb.set_trace()

        if pc_data.shape[0] == 0:
            continue
        x_mean, y_mean, z_mean, _, _ = list(pc_data.mean(axis=0))
        pc_data[:, 0] = pc_data[:, 0] - x_mean
        pc_data[:, 1] = pc_data[:, 1] - y_mean
        pc_data[:, 2] = pc_data[:, 2] - z_mean

        # box3d_lidar_ori = info['box3d_lidar']
        # box3d_lidar_ori[0] = -x_mean
        # box3d_lidar_ori[1] = -y_mean
        # box3d_lidar_ori[2] = -z_mean

        gt_boxes_restore = copy.deepcopy(gt_boxes)
        gt_boxes_restore[:, 0] += x_mean
        gt_boxes_restore[:, 1] += y_mean
        gt_boxes_restore[:, 2] += z_mean
        box_center = np.array(gt_boxes_restore[:, :3])
        box_center = rotate_points_along_z(box_center, -trans_angle)
        gt_boxes_restore[:, :3] = box_center
        gt_boxes_restore[:, 6] -= trans_angle


        # print(pc_data.shape)
        # print(f' pc_data.shape = {pc_data.shape}, overlap={overlap},  ')
        print(f' pc_data.shape = {pc_data.shape},')
        print(f' gt_boxes = {gt_boxes}, box_pred={box_pred} z_mean = {z_mean}')

        gt_label = 1
        ref_label = 1
        
        # box3d_lidar_ori = info['box3d_lidar']
        print(f"box3d_lidar={box3d_lidar} gt_boxes_restore={gt_boxes_restore}")
        # box3d_lidar = deepcopy(box3d_lidar_ori)
        # box3d_lidar[:3] = 0

        

        draw_scenes(pc_data, gt_boxes=gt_boxes, gt_labels=np.array([gt_label]), 
                                ref_boxes=box_pred, ref_labels=np.array([ref_label]))
        mlab.show()
        # import pdb;pdb.set_trace()


# /home/yifanzhang/workspace/cvae_uncertainty/output/exp1/default/eval/epoch_80/val/default/final_result/data/result.pkl
