from pathlib import Path
from collections import defaultdict

from copy import deepcopy
import cv2
import os
import torch
import pickle
import copy
import SharedArray

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler
import torch.distributed as dist

from pcdet.utils import common_utils
from sklearn.model_selection import KFold
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti

np.seterr(divide='ignore',invalid='ignore')


def scan_to_rv(scan):
    rv_width = 512
    rv_height = 48
    fov_up = 3 / 180.0 * np.pi
    fov_down = -25.0 / 180.0 * np.pi
    r = np.sqrt(scan[:, 0]**2 + scan[:, 1]**2 + scan[:, 2]**2)
    u = 0.5 * (1 - np.arctan(scan[:, 1]/scan[:, 0])/np.pi) * rv_width
    v = (1 - (np.arcsin(scan[:, 2]/ r) + abs(fov_down))/ (fov_up + abs(fov_down)) ) * rv_height

    u = np.floor(u)
    u = np.minimum(rv_width - 1, u)
    u = np.maximum(u, 0)

    v = np.floor(v)
    v = np.minimum(rv_height - 1, v)
    v = np.maximum(v, 0)
    return np.stack([u, v, r]).T

def scan_to_rv_waymo(scan):
    rv_width = 2650
    rv_height = 64
    fov_up = 30 / 180.0 * np.pi
    fov_down = -90.0 / 180.0 * np.pi
    r = np.sqrt(scan[:, 0]**2 + scan[:, 1]**2 + scan[:, 2]**2)
    u = 0.5 * (1 - np.arctan(scan[:, 1]/scan[:, 0])/np.pi) * rv_width
    v = (1 - (np.arcsin(scan[:, 2]/ r) + abs(fov_down))/ (fov_up + abs(fov_down)) ) * rv_height

    u = np.floor(u)
    u = np.minimum(rv_width - 1, u)
    u = np.maximum(u, 0)

    v = np.floor(v)
    v = np.minimum(rv_height - 1, v)
    v = np.maximum(v, 0)
    return np.stack([u, v, r]).T

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

def get_road_plane(plane_file):
    with open(plane_file, 'r') as f:
        lines = f.readlines()
    lines = [float(i) for i in lines[3].split()]
    plane = np.asarray(lines)

    # Ensure normal is always facing up, this is in the rectified camera coordinate
    if plane[1] > 0:
        plane = -plane

    norm = np.linalg.norm(plane[0:3])
    plane = plane / norm
    return plane

def get_calib(calib_file):
    calib=calibration_kitti.Calibration(calib_file)
    return calib

def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
    """
    Only validate in KITTIDataset
    Args:
        gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
        road_planes: [a, b, c, d]
        calib:

    Returns:
    """
    a, b, c, d = road_planes
    center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
    cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
    center_cam[:, 1] = cur_height_cam
    cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
    mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
    gt_boxes[:, 2] -= mv_height  # lidar view
    return gt_boxes, mv_height




class KittiGtDataset():
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)

        self.enable_similar_type = self.dataset_cfg.get("ENABLE_SIMILAR_TYPE", False)

        if 'FOLD_IDX' in self.dataset_cfg:
            # db_infos_path = self.root_path / 'kitti_dbinfos_train.pkl'
            db_infos_path_list = [self.root_path / 'kitti_dbinfos_train.pkl']
            # db_infos_path_list = [self.root_path / 'kitti_dbinfos_trainval.pkl']
            # db_infos_path_list = [self.root_path / 'kitti_dbinfos_train.pkl', self.root_path / 'kitti_dbinfos_val.pkl']
            infos_list = []
            for db_infos_path in db_infos_path_list:
                with open(db_infos_path, 'rb') as f:
                    infos = pickle.load(f)
                    infos_list.append(infos)
            splits=KFold(n_splits=10,shuffle=True,random_state=42) # random_state=42
            fold_idx = self.dataset_cfg.FOLD_IDX
            used_infos = []
            if self.enable_similar_type:
                for infos in infos_list:
                    used_infos.extend(infos['Car'])
                    used_infos.extend(infos['Van'])
            else:
                used_infos.extend(infos['Car'])
            train_idx,val_idx = [x for x in splits.split(np.arange(len(used_infos)))][fold_idx]
            if self.training:
                self.kitti_infos = [used_infos[idx] for idx in train_idx]
            else:
                self.kitti_infos = [used_infos[idx] for idx in val_idx]     
        else:
            if self.training:
                db_infos_path = self.root_path / 'kitti_dbinfos_train.pkl'
            else:
                db_infos_path = self.root_path / 'kitti_dbinfos_val.pkl'

            with open(db_infos_path, 'rb') as f:
                infos = pickle.load(f)
            if self.enable_similar_type:
                self.kitti_infos = infos['Car'] + infos['Van']
            else:
                self.kitti_infos = infos['Car']

        

        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        # dense gt infos
        self.dense_gt_infos = [x for x in self.kitti_infos if x['num_points_in_gt']>1000]
        logger.info(f'Length of dense_gt_infos is {len(self.dense_gt_infos)}')

        self.linear_anneal = 0
        self.force_ratio = self.dataset_cfg.FORCE_RATIO
        self.force_num = self.dataset_cfg.FORCE_NUM

        self.enable_flip = self.dataset_cfg.get('ENABLE_FLIP', False)
        self.scale_range = self.dataset_cfg.get('RANDOM_SCALE_RANGE', [1.0, 1.0])
        self.angle_rot_max = self.dataset_cfg.get('ANGLE_ROT_MAX', 0) # 0.78539816
        self.pos_shift_max = self.dataset_cfg.get('POS_SHIFT_MAX', 0)
        logger.info(f"### Aug params: flip={self.enable_flip}, scale={self.scale_range}, rot={self.angle_rot_max},shift={self.pos_shift_max}, enable_similar_type={self.enable_similar_type}")

        self.rv_width = 512
        self.rv_height = 48

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def prepare_data(self, data_dict):
        if self.training:
            pass # aug


        # data_dict.pop('gt_names', None)
        return data_dict

    # self.occlude_aug(info, points, max_num=9999, min_num=1, max_try_time=5)
    def occlude_aug(self, sample_car_info, points, max_num=9999, min_num=1, max_try_time=5):
        
        frame_id = sample_car_info['image_idx']

        calib_path = f'{self.root_path}/training/calib/{frame_id}.txt'
        road_plane_path = f'{self.root_path}/training/planes/{frame_id}.txt'

        road_plane = get_road_plane(road_plane_path)
        calib = get_calib(calib_path)


        # step1: get dense object
        dense_gt_info = self.dense_gt_infos[np.random.choice(len(self.dense_gt_infos))]
        dense_pc_path = f'{self.root_path}/{dense_gt_info["path"]}'
        dense_points = np.fromfile(dense_pc_path, dtype=np.float32).reshape(-1, 4)
        dense_gt_box = dense_gt_info['box3d_lidar']


        # step2: change the points to scan

        scan = deepcopy(points)
        scan[:, :3] += sample_car_info['box3d_lidar'][:3]

        # random put on the road plane
        scale=np.random.random(1) * 0.4 + 0.5
        # scale=np.random.random(1) * 0.15 + 0.75
        new_c_x = sample_car_info['box3d_lidar'][0] * scale

        if new_c_x + dense_gt_box[3]/2 > sample_car_info['box3d_lidar'][0] - sample_car_info['box3d_lidar'][3]/2:
            new_c_x = sample_car_info['box3d_lidar'][0] - sample_car_info['box3d_lidar'][3]/2 - dense_gt_box[3]/2
            scale = new_c_x / sample_car_info['box3d_lidar'][0]
        new_c_y = sample_car_info['box3d_lidar'][1] * scale

        # ad hoc
        dense_gt_box[0] = new_c_x
        dense_gt_box[1] = new_c_y
        dense_gt_box = np.expand_dims(dense_gt_box, 0)
        new_dense_gt_box, mv_height = put_boxes_on_road_planes(dense_gt_box, road_plane, calib)
        dense_scan = dense_points
        dense_scan[:, 0] += new_c_x
        dense_scan[:, 1] += new_c_y
        # no enough, need to put on the road plane
        dense_scan[:, 2] += dense_gt_info['box3d_lidar'][2] - mv_height

        # todo points' coordinates in the range view

        # size of range view: (5x48x512)
        rv_sample = scan_to_rv(scan)
        rv_dense = scan_to_rv(dense_scan)

        # random move 
        sample_x_min = min(rv_sample[:, 0])
        sample_x_max = max(rv_sample[:, 0])
        sample_y_min = min(rv_sample[:, 1])
        sample_y_max = max(rv_sample[:, 1])


        dense_x_min = min(rv_dense[:, 0])
        dense_x_max = max(rv_dense[:, 0])
        dense_y_min = min(rv_dense[:, 1])
        # dense_y_max = max(rv_dense[:, 1])

        x_move_min = 0.7 * sample_x_min + 0.3 * sample_x_max - dense_x_max
        x_move_max = 0.3 * sample_x_min + 0.7 * sample_x_max - dense_x_min
        y_move_min = 0.9 * sample_y_min + 0.1 * sample_y_max - dense_y_min
        y_move_max = 0.5 * sample_y_min + 0.5 * sample_y_max - dense_y_min

        try_num=0
        while True:

            x_move_value = np.random.rand() * (x_move_max-x_move_min) + x_move_min
            y_move_value = np.random.rand() * (y_move_max-y_move_min) + y_move_min

            rv_dense[:, 0] += x_move_value
            rv_dense[:, 1] += y_move_value

            # print('#### min max', min(rv_dense[:, 0]), max(rv_dense[:, 0]), min(rv_dense[:, 1]), max(rv_dense[:, 1]))
            dot_list = rv_dense[:, :2].astype(np.int64)
            hull = cv2.convexHull(dot_list,clockwise=True,returnPoints=True)
            hull = hull.squeeze(1)

            img = np.ones([self.rv_height, self.rv_width])
            cv2.fillConvexPoly(img, hull, (-1))
            cols = rv_sample[:, 1].astype(np.int32)
            rows = rv_sample[:, 0].astype(np.int32)

            reserved_points=points[img[cols, rows]!=-1]

            if  len(reserved_points)>=min_num and len(reserved_points)<=max_num:
                break
            elif try_num>max_try_time:
                reserved_points = points
                break

            try_num += 1
        return reserved_points

    def __getitem__(self, index):
        data_dict = {}

        info = copy.deepcopy(self.kitti_infos[index])

        # print(f"### info = {info.keys()}")

        pc_path = self.root_path / info['path']
        points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)

        #################################  occlude data augmentation start  ##########################################

        # if self.training and points.shape[0] > 5:
        #     try_num=0
        #     while True:
        #         try_num += 1
        #         points = self.occlude_aug(info, points)
        #         if points.shape!=0:
        #             break
        #         else:
        #             points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
        #             if try_num == 5:
        #                 break

        # print("###self.force_ratio * self.linear_anneal", self.force_ratio, self.linear_anneal)
        if self.training:
            random_v = np.random.rand()
            if self.force_ratio * self.linear_anneal > random_v and points.shape[0] > self.force_num:
                # 以一定比例(force_ratio)强制采样到比较稀疏的数量
                points = self.occlude_aug(info, points, max_num=self.force_num, min_num=1, max_try_time=20)
            elif points.shape[0] > 10:
                points = self.occlude_aug(info, points, max_num=99999, min_num=1, max_try_time=5)
            else:
                pass
                

        #################################  occlude data augmentation end ##########################################


        ################################## random flip, scaling #######################################

        flip_mark = False
        noise_scale = 1.0
        if self.training:
            if self.enable_flip:
                flip_mark = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
                if flip_mark:
                    points[:, 1] = -points[:, 1]

            noise_scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            points[:, :3] *= noise_scale


        # rotate points along the z axis
        if points.shape[0] != 0:
            x_mean, y_mean, z_mean = list(points[:, :3].mean(axis=0))
        else:
            x_mean, y_mean, z_mean = [0,0,0]

        point_anchor_size = [3.9, 1.6, 1.56]
        dxa, dya, dza = point_anchor_size
        diagonal = np.sqrt(dxa ** 2 + dya ** 2)

        # 增加点云到每个平面的距离

        pos_shift = [0, 0]
        angle_rot = 0

        if self.training:
        #     # rotate + movement
        #     # range: pi/4, 1m

            angle_rot = (np.random.rand(1)[0] - 0.5) / 0.5 * self.angle_rot_max
            pos_shift = ((np.random.rand(2) - 0.5) / 0.5) * self.pos_shift_max
            
            points = rotate_points_along_z(points, angle_rot)


        points[:, 0] = (points[:, 0] - x_mean + pos_shift[0]) / diagonal
        points[:, 1] = (points[:, 1] - y_mean + pos_shift[1]) / diagonal
        points[:, 2] = (points[:, 2] - z_mean) / dza

        keep_num = 512
        if points.shape[0] != 0:
            choice = np.random.choice(points.shape[0], keep_num, replace=True)
            points = points[choice, :]
        else:
            points = np.full((keep_num, 4), 0)

        # import pdb;pdb.set_trace()
        # print('info gt_idx = ', info['gt_idx'], type(info['gt_idx']))

        data_dict['points'] = points.transpose()
        data_dict['frame_id'] = info['image_idx']
        data_dict['gt_id'] = info['gt_idx']

        if 'box3d_lidar' not in info:
            return data_dict


        box3d_lidar_ori = info['box3d_lidar']
        box3d_lidar = copy.deepcopy(box3d_lidar_ori)
        if flip_mark:
            # import pdb;pdb.set_trace()
            box3d_lidar[6] = -box3d_lidar[6]
        box3d_lidar[:6] *= noise_scale

        box3d_lidar[0] = (- x_mean + pos_shift[0]) / diagonal
        box3d_lidar[1] = (- y_mean + pos_shift[1]) / diagonal
        box3d_lidar[2] = (- z_mean) / dza
        box3d_lidar[3] = np.log(box3d_lidar[3]/dxa)
        box3d_lidar[4] = np.log(box3d_lidar[4]/dya)
        box3d_lidar[5] = np.log(box3d_lidar[5]/dza)
        box3d_lidar[6] += angle_rot

        box3d_lidar_dim7 = copy.deepcopy(box3d_lidar)

        angle = box3d_lidar[6]
        box3d_lidar[6] = np.sin(angle)
        box3d_lidar = np.append(box3d_lidar, np.cos(angle))

        # points sample
        # print(f"### points.shape[0] = {points.shape[0]}")

        # import pdb;pdb.set_trace()

        data_dict['gt_boxes_input'] = box3d_lidar
        data_dict['gt_boxes'] = box3d_lidar_dim7

        return data_dict

    # @staticmethod
    # def generate_prediction_dicts(batch_dict, box_pred, class_names, output_path=None):
    #     """
    #     Args:
    #         batch_dict:
    #             points (32, 4, 512)
    #             frame_id (32,)
    #                 gt_id (32,)
    #                 gt_boxes_input (32, 8)
    #                 gt_boxes (32, 7)
    #             batch_size 32
    #         box_pred: np.array(32, 9)
    #         class_names:
    #         output_path:

    #     Returns:

    #     """

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0) # concat [(512, 4), (512, 4)] -> [1024, 4]
                elif key in ['voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in []: 
                    max_gt = max([len(x) for x in val]) 
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        # Pad with nan, to be replaced later in the pipeline.
                        pad_value = np.nan

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret

class WaymoGtDataset():
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)

        self.enable_similar_type = self.dataset_cfg.get("ENABLE_SIMILAR_TYPE", False)
        self.use_shared_memory = self.dataset_cfg.get("USE_SHARED_MEMORY", False)
        self.gt_database_data_key = self.load_db_to_shared_memory() if self.use_shared_memory else None

        # if self.use_shared_memory:
        #     self.gt_database_data = SharedArray.attach(f"shm://{self.gt_database_data_key}")
        #     self.gt_database_data.setflags(write=0)
        # else:
        #     self.gt_database_data = None

        if 'FOLD_IDX' in self.dataset_cfg:
            db_infos_path_list = [self.root_path / 'waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl']
            # db_infos_path_list = ['/home/yifanzhang/git/OpenPCDet/tools/tmp.pkl']
            infos_list = []
            for db_infos_path in db_infos_path_list:
                with open(db_infos_path, 'rb') as f:
                    infos = pickle.load(f)
                    infos_list.append(infos)
            splits=KFold(n_splits=5,shuffle=True,random_state=42) # random_state=42
            fold_idx = self.dataset_cfg.FOLD_IDX
            used_infos = []
            if self.enable_similar_type:
                raise NotImplementedError()
            else:
                used_infos.extend(infos['Vehicle'])

            train_idx, val_idx = [x for x in splits.split(np.arange(len(used_infos)))][fold_idx]
            if self.training:
                self.waymo_infos = [used_infos[idx] for idx in train_idx]
            else:
                self.waymo_infos = [used_infos[idx] for idx in val_idx]     
        else:

            if self.training:
                db_infos_path = self.root_path / 'waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl'
            else:
                db_infos_path = self.root_path / 'waymo_processed_data_v0_5_0_waymo_dbinfos_val_sampled_1.pkl'

            with open(db_infos_path, 'rb') as f:
                infos = pickle.load(f)
            self.waymo_infos = infos['Vehicle']
            # if self.enable_similar_type:
            #     self.kitti_infos = infos['Car'] + infos['Van']
            # else:
            #     self.kitti_infos = infos['Car']

        

        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        # dense gt infos
        self.dense_gt_infos = [x for x in self.waymo_infos if x['num_points_in_gt']>1000]
        logger.info(f'Length of dense_gt_infos is {len(self.dense_gt_infos)}')

        self.linear_anneal = 0
        self.force_ratio = self.dataset_cfg.FORCE_RATIO
        self.force_num = self.dataset_cfg.FORCE_NUM

        self.enable_flip = self.dataset_cfg.get('ENABLE_FLIP', False)
        self.scale_range = self.dataset_cfg.get('RANDOM_SCALE_RANGE', [1.0, 1.0])
        self.angle_rot_max = self.dataset_cfg.get('ANGLE_ROT_MAX', 0) # 0.78539816
        # self.pos_shift_max = self.dataset_cfg.get('POS_SHIFT_MAX', 0)
        self.pos_shift_max = 0
        logger.info(f"### Aug params: flip={self.enable_flip}, scale={self.scale_range}, rot={self.angle_rot_max},shift={self.pos_shift_max}, enable_similar_type={self.enable_similar_type}")

        self.rv_width = 2650
        self.rv_height = 64

    def load_db_to_shared_memory(self):
        self.logger.info('Loading GT database to shared memory')
        cur_rank, world_size, num_gpus = common_utils.get_dist_info(return_gpu_per_machine=True)

        if 'FOLD_IDX' in self.dataset_cfg or self.training:
            db_data_path = self.root_path.resolve() / 'waymo_processed_data_v0_5_0_gt_database_train_sampled_1_global.npy'
            sa_key = 'waymo_processed_data_v0_5_0_gt_database_train_sampled_1_global.npy'
        else:
            db_data_path = self.root_path.resolve() / 'waymo_processed_data_v0_5_0_gt_database_val_sampled_1_global.npy'
            sa_key = 'waymo_processed_data_v0_5_0_gt_database_val_sampled_1_global.npy'

        if cur_rank % num_gpus == 0 and not os.path.exists(f"/dev/shm/{sa_key}"):
            gt_database_data = np.load(db_data_path)
            common_utils.sa_create(f"shm://{sa_key}", gt_database_data)
            
        if num_gpus > 1:
            dist.barrier()
        self.logger.info('GT database has been saved to shared memory')
        return sa_key

    # def __del__(self):
    #     if self.use_shared_memory:
    #         self.logger.info('Deleting GT database from shared memory')
    #         cur_rank, num_gpus = common_utils.get_dist_info()
    #         sa_key = self.gt_database_data_key
    #         if cur_rank % num_gpus == 0 and os.path.exists(f"/dev/shm/{sa_key}"):
    #             SharedArray.delete(f"shm://{sa_key}")

    #         if num_gpus > 1:
    #             dist.barrier()
    #         self.logger.info('GT database has been removed from shared memory')

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def prepare_data(self, data_dict):
        if self.training:
            pass # aug

        # data_dict.pop('gt_names', None)
        return data_dict

    # self.occlude_aug(info, points, max_num=9999, min_num=1, max_try_time=5)
    def occlude_aug(self, sample_car_info, points, max_num=9999, min_num=1, max_try_time=5):

        frame_id = sample_car_info['sequence_name'] + "#" + str(sample_car_info["sample_idx"])

        # step1: get dense object
        dense_gt_info = self.dense_gt_infos[np.random.choice(len(self.dense_gt_infos))]
        
        if self.use_shared_memory:
            gt_database_data = SharedArray.attach(f"shm://{self.gt_database_data_key}")
            gt_database_data.setflags(write=0)
            start_offset, end_offset = dense_gt_info['global_data_offset']
            dense_points = copy.deepcopy(gt_database_data[start_offset:end_offset])
            
        else:
            dense_pc_path = f'{self.root_path}/{dense_gt_info["path"]}'
            dense_points = np.fromfile(dense_pc_path, dtype=np.float32).reshape(-1, 5)

        dense_gt_box = copy.deepcopy(dense_gt_info['box3d_lidar'])


        # step2: change the points to scan

        scan = deepcopy(points)
        scan[:, :3] += sample_car_info['box3d_lidar'][:3]

        # random put on the road plane
        scale=np.random.random(1) * 0.4 + 0.5
        # scale=np.random.random(1) * 0.15 + 0.75
        new_c_x = sample_car_info['box3d_lidar'][0] * scale

        if new_c_x + dense_gt_box[3]/2 > sample_car_info['box3d_lidar'][0] - sample_car_info['box3d_lidar'][3]/2:
            new_c_x = sample_car_info['box3d_lidar'][0] - sample_car_info['box3d_lidar'][3]/2 - dense_gt_box[3]/2
            scale = new_c_x / sample_car_info['box3d_lidar'][0]
        new_c_y = sample_car_info['box3d_lidar'][1] * scale
        new_c_z = sample_car_info['box3d_lidar'][2]

        # ad hoc
        dense_gt_box[0] = new_c_x
        dense_gt_box[1] = new_c_y
        dense_gt_box = np.expand_dims(dense_gt_box, 0)
        dense_scan = dense_points
        dense_scan[:, 0] += new_c_x
        dense_scan[:, 1] += new_c_y
        dense_scan[:, 2] += new_c_z

        # todo points' coordinates in the range view

        # size of range view: (5x48x512)
        rv_sample = scan_to_rv_waymo(scan)
        rv_dense = scan_to_rv_waymo(dense_scan)

        # random move 
        sample_x_min = min(rv_sample[:, 0])
        sample_x_max = max(rv_sample[:, 0])
        sample_y_min = min(rv_sample[:, 1])
        sample_y_max = max(rv_sample[:, 1])

        try:
            dense_x_min = min(rv_dense[:, 0])
            dense_x_max = max(rv_dense[:, 0])
            dense_y_min = min(rv_dense[:, 1])
        except:
            print("### len(rv_dense)", len(rv_dense))
            return points
        # dense_y_max = max(rv_dense[:, 1])

        x_move_min = 0.7 * sample_x_min + 0.3 * sample_x_max - dense_x_max
        x_move_max = 0.3 * sample_x_min + 0.7 * sample_x_max - dense_x_min
        y_move_min = 0.9 * sample_y_min + 0.1 * sample_y_max - dense_y_min
        y_move_max = 0.5 * sample_y_min + 0.5 * sample_y_max - dense_y_min

        try_num=0
        while True:

            x_move_value = np.random.rand() * (x_move_max-x_move_min) + x_move_min
            y_move_value = np.random.rand() * (y_move_max-y_move_min) + y_move_min

            rv_dense[:, 0] += x_move_value
            rv_dense[:, 1] += y_move_value

            # print('#### min max', min(rv_dense[:, 0]), max(rv_dense[:, 0]), min(rv_dense[:, 1]), max(rv_dense[:, 1]))
            dot_list = rv_dense[:, :2].astype(np.int64)
            hull = cv2.convexHull(dot_list,clockwise=True,returnPoints=True)
            hull = hull.squeeze(1)

            img = np.ones([self.rv_height, self.rv_width])
            cv2.fillConvexPoly(img, hull, (-1))
            cols = rv_sample[:, 1].astype(np.int32)
            rows = rv_sample[:, 0].astype(np.int32)

            reserved_points=points[img[cols, rows]!=-1]

            if len(reserved_points)>=min_num and len(reserved_points)<=max_num:
                break
            elif try_num>max_try_time:
                reserved_points = points
                break

            try_num += 1
        return reserved_points

    def __getitem__(self, index):
        data_dict = {}
        while True:
            info = copy.deepcopy(self.waymo_infos[index])
            if self.use_shared_memory:
                gt_database_data = SharedArray.attach(f"shm://{self.gt_database_data_key}")
                gt_database_data.setflags(write=0)
                start_offset, end_offset = info['global_data_offset']
                points = copy.deepcopy(gt_database_data[start_offset:end_offset])
            else:
                pc_path = self.root_path / info['path']
                points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 5)

            if len(points)>0 or (not self.training):
                break
            else:
                p = np.random.rand()
                if p>0.99:
                    break
                else:
                    index = np.random.randint(len(self.waymo_infos))

        #################################  occlude data augmentation start  ######################################
        # print("### original points.shape", points.shape)
        data_dict['raw_points'] = copy.deepcopy(points)
        if self.training:
            random_v = np.random.rand()
            if self.force_ratio * self.linear_anneal > random_v and points.shape[0] > self.force_num:
                # 以一定比例(force_ratio)强制采样到比较稀疏的数量
                points = self.occlude_aug(info, points, max_num=self.force_num, min_num=1, max_try_time=20)
            # ad hoc
            elif points.shape[0] > 10:
                points = self.occlude_aug(info, points, max_num=99999, min_num=1, max_try_time=5)
            else:
                pass
        # print("### after occlusion points.shape", points.shape)
        #################################  occlude data augmentation end ##########################################

        box3d_lidar_ori = info['box3d_lidar']
        data_dict['gt_boxes_ori'] = box3d_lidar_ori
        box3d_lidar = copy.deepcopy(box3d_lidar_ori)
        # limit the angle of box and points
        # box3d_lidar[6] = common_utils.limit_period(torch.tensor(box3d_lidar[6]), offset=0.5, period=np.pi/2).item()
        # trans_angle = box3d_lidar[6] - box3d_lidar_ori[6]
        azimuth = np.arctan2(box3d_lidar[0], box3d_lidar[1])
        new_azimuth = common_utils.limit_period(torch.tensor(azimuth), offset=0.5, period=np.pi/2).item()
        trans_angle = new_azimuth - azimuth

        ## rotate the box and points
        points[:, :3] += box3d_lidar[:3]
        points = rotate_points_along_z(points, trans_angle)

        box3d_lidar[6] += trans_angle
        box_center = np.array([box3d_lidar[:3]])
        box_center = rotate_points_along_z(box_center, trans_angle)
        box3d_lidar[:3] = box_center
        data_dict['trans_angle'] = trans_angle
        ################################## random flip, scaling #######################################
        # flip_mark_x = False
        # flip_mark_y = False
        # noise_scale = 1.0
        # if self.training:
        #     if self.enable_flip:
        #         flip_mark_x = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
        #         if flip_mark_x:
        #             points[:, 1] = -points[:, 1]
        #         flip_mark_y = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
        #         if flip_mark_y:
        #             points[:, 0] = -points[:, 0]

        #     noise_scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        #     points[:, :3] *= noise_scale

        flip_mark = False
        noise_scale = 1.0
        if self.training:
            if self.enable_flip:
                flip_mark = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
                if flip_mark:
                    points[:, 1] = -points[:, 1]

            noise_scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            points[:, :3] *= noise_scale

        ############################## rotate points along the z axis
        point_anchor_size = [4.7, 2.1, 1.7]
        dxa, dya, dza = point_anchor_size
        diagonal = np.sqrt(dxa ** 2 + dya ** 2)

        pos_shift = [0, 0]
        angle_rot = 0

        if self.training:
        #     # rotate + movement
        #     # range: pi/4, 1m

            angle_rot = (np.random.rand(1)[0] - 0.5) / 0.5 * self.angle_rot_max
            pos_shift = ((np.random.rand(2) - 0.5) / 0.5) * self.pos_shift_max
            
            points = rotate_points_along_z(points, angle_rot)


        if points.shape[0] != 0:
            x_mean, y_mean, z_mean = list(points[:, :3].mean(axis=0))
        else:
            x_mean, y_mean, z_mean = [0,0,0]

        points[:, 0] = (points[:, 0] - x_mean + pos_shift[0]) / diagonal
        points[:, 1] = (points[:, 1] - y_mean + pos_shift[1]) / diagonal
        points[:, 2] = (points[:, 2] - z_mean) / dza

        
        keep_num = 2048
        if points.shape[0] != 0:
            choice = np.random.choice(points.shape[0], keep_num, replace=True)
            points = points[choice, :]
        else:
            points = np.full((keep_num, 5), 0).astype(np.float32)

        # import pdb;pdb.set_trace()
        # print('info gt_idx = ', info['gt_idx'], type(info['gt_idx']))

        data_dict['points'] = points.transpose()
        data_dict['frame_id'] = info['sequence_name'] + "#" + str(info["sample_idx"])
        data_dict['gt_id'] = info['gt_idx']

        if 'box3d_lidar' not in info:
            raise ValueError
            # return data_dict


        # if flip_mark_x:
        #     # import pdb;pdb.set_trace()
        #     box3d_lidar[6] = -box3d_lidar[6]
        # if flip_mark_y:
        #     box3d_lidar[6] = np.pi - box3d_lidar[6]

        if flip_mark:
            box3d_lidar[1] = -box3d_lidar[1]
            box3d_lidar[6] = -box3d_lidar[6]

        box3d_lidar[:6] *= noise_scale
        box_center = np.array([box3d_lidar[:3]])
        box_center = rotate_points_along_z(box_center, angle_rot)
        box3d_lidar[:3] = box_center
        box3d_lidar[6] += angle_rot

        box3d_lidar[0] = (box3d_lidar[0] - x_mean + pos_shift[0]) / diagonal
        box3d_lidar[1] = (box3d_lidar[1] - y_mean + pos_shift[1]) / diagonal
        box3d_lidar[2] = (box3d_lidar[2] - z_mean) / dza
        box3d_lidar[3] = np.log(box3d_lidar[3]/dxa)
        box3d_lidar[4] = np.log(box3d_lidar[4]/dya)
        box3d_lidar[5] = np.log(box3d_lidar[5]/dza)

        box3d_lidar_dim7 = copy.deepcopy(box3d_lidar)

        angle = box3d_lidar[6]
        box3d_lidar[6] = np.sin(angle)
        box3d_lidar = np.append(box3d_lidar, np.cos(angle))

        # points sample
        # print(f"### points.shape[0] = {points.shape[0]}")

        # import pdb;pdb.set_trace()

        data_dict['gt_boxes_input'] = box3d_lidar
        data_dict['gt_boxes'] = box3d_lidar_dim7

        return data_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.waymo_infos) * self.total_epochs

        return len(self.waymo_infos)

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            if key in ['raw_points']:
                continue
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0) # concat [(512, 4), (512, 4)] -> [1024, 4]
                elif key in ['voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in []: 
                    max_gt = max([len(x) for x in val]) 
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        # Pad with nan, to be replaced later in the pipeline.
                        pad_value = np.nan

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret


class DistributedSampler(_DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset_map = {
        'KittiGtDataset': KittiGtDataset,
        'WaymoGtDataset': WaymoGtDataset
    }
    dataset_class = dataset_map[dataset_cfg.get('DATASET', 'KittiGtDataset')]
    dataset = dataset_class(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset) # By default, shuffle = True
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler



if __name__ == '__main__':
    # main()
    from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
    import mayavi.mlab as mlab


    cfg_file = 'cfgs/waymo_exp20.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    cfg.TAG = Path(cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    # if args.set_cfgs is not None:
    #     cfg_from_list(args.set_cfgs, cfg)

    from pcdet.utils import common_utils
    import datetime

    log_file = '/home/yifanzhang/logs_output/' + 'log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=4,
        dist=False, workers=4,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=False,
        total_epochs=10
    )

    from tqdm import tqdm
    for i in tqdm(range(0, len(train_set))):
        item = train_set[i]
        # print(item)
        from vis_data import draw_scenes
        points = item['points'].transpose()
        raw_points = item['raw_points']
        # print('### item.keys()', item.keys())
        # print('### points.shape', points.shape)

        point_anchor_size = [4.7, 2.1, 1.7] # Waymo
        # point_anchor_size = [3.9, 1.6, 1.56] # kitti
        dxa, dya, dza = point_anchor_size
        diagonal = np.sqrt(dxa ** 2 + dya ** 2)
        points[:, 0] *= diagonal
        points[:, 1] *= diagonal
        points[:, 2] *= dza

        x,y,z,w,h,d,angle =  item['gt_boxes']

        x *= diagonal
        y *= diagonal
        z *= dza
        w = np.exp(w) * point_anchor_size[0]
        h = np.exp(h) * point_anchor_size[1]
        d = np.exp(d) * point_anchor_size[2]

        # angle = np.arcsin(_sin)
        # if _cos < 0:
        #     angle += np.pi
        box3d_lidar = np.array([x,y,z,w,h,d,angle])
        print(f'box3d_lidar = {box3d_lidar}', item.keys())
        
        # import pdb;pdb.set_trace()
        raw_points[:, 0] += item['gt_boxes_ori'][0]
        raw_points[:, 1] += item['gt_boxes_ori'][1]
        raw_points[:, 2] += item['gt_boxes_ori'][2]
        draw_scenes(raw_points, gt_boxes=item['gt_boxes_ori'].reshape(-1, 7))
        draw_scenes(points, gt_boxes=np.array([box3d_lidar]))

        # if i==3000000-1:
        #     draw_scenes(points, gt_boxes=np.array([box3d_lidar]))
        #     mlab.show()
