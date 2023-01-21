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

def visualize_pts(pts, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
                  show_intensity=False, size=(600, 600), draw_origin=True, point_scale_factor=1):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    if show_intensity:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3],
                          colormap='gnuplot', scale_factor=point_scale_factor, figure=fig)
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2],
                          colormap='gnuplot', scale_factor=point_scale_factor, figure=fig)
    G.glyph.scale_mode = 'scale_by_vector'
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)

    return fig

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

    fig = visualize_pts(points, bgcolor=(1,1,1), fgcolor=(0,0,0), draw_origin=False, point_scale_factor=0.025)
    # fig = draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))

    if gt_boxes is not None and len(gt_boxes) > 0:
        # hit
        mask = (gt_labels == 1)
        corners3d = boxes_to_corners_3d(gt_boxes[mask, :])
        # fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)#blue 
        fig = draw_corners3d(corners3d, fig=fig, color=(0.5294, 0.80784, 0.92156), max_num=100, line_width=4) # (0.5294, 0.80784, 0.92156)

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



if __name__ == '__main__':
    # main()
    print("##################### only for OpenPCDet #############################")


    # 读取单个数据
    # kitti_infos_train
    with open('kitti/kitti_dbinfos_train.pkl', 'rb') as f:
    # with open('kitti/dbinfos_train.pkl', 'rb') as f:
        db_infos = pickle.load(f)


# {'name': 'Car', 'path': 'gt_database/000007_Car_0.bin', 'image_idx': '000007', 'gt_idx': 0,
#  'box3d_lidar': array([ 2.52990341e+01,  7.08958089e-01, -6.88423336e-01,  3.20000000e+00, 1.66000000e+00,  1.61000000e+00,  1.92036732e-02]),
#  'num_points_in_gt': 183, 'difficulty': 0,
#  'bbox': array([564.62, 174.59, 616.43, 224.74], dtype=float32),
#  'score': -1.0}



    for index in range(len(db_infos['Car'])):
        # print(db_infos['Car'][index])

        # show the points
        pc_path = 'kitti/' + db_infos['Car'][index]['path']

        import numpy as np
        pc_data = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
        # print(pc_data.shape)

        box3d_lidar_ori = db_infos['Car'][index]['box3d_lidar']
        print(f"box3d_lidar_ori = {box3d_lidar_ori}")
        box3d_lidar = deepcopy(box3d_lidar_ori)
        box3d_lidar[:3] = 0
        # box3d_lidar[3:6] *= 1.1
        box3d_lidar[3:6] += 0.02

        from pcdet.utils import box_utils
        corners_lidar = box_utils.boxes_to_corners_3d(box3d_lidar.reshape(1,7))
        # import pdb;pdb.set_trace()
        draw_scenes(pc_data, gt_boxes=np.array([box3d_lidar]), gt_labels=np.array([1]))
        mlab.show()
        # flag = box_utils.in_hull(pc_data[:, :3], corners_lidar.squeeze())
        # if len(pc_data)-flag.sum() > 1:
        #     print("### not in box points num = ", len(pc_data)-flag.sum())
            # draw_scenes(pc_data, gt_boxes=np.array([box3d_lidar]), gt_labels=np.array([1]))
            # mlab.show()
            # draw_scenes(pc_data[~flag], gt_boxes=np.array([box3d_lidar]), gt_labels=np.array([1]))
            # mlab.show()

            # print((np.abs(pc_data[~flag, :3]) - box3d_lidar[3:6]/2).max())
            # if (np.abs(pc_data[~flag, :3]) - box3d_lidar[3:6]/2).max()>0.5:
            #     draw_scenes(pc_data, gt_boxes=np.array([box3d_lidar]), gt_labels=np.array([1]))
            #     mlab.show()
            #     draw_scenes(pc_data[~flag], gt_boxes=np.array([box3d_lidar]), gt_labels=np.array([1]))
            #     mlab.show()

    # import pdb;pdb.set_trace()
    # 读取 annotation
    


    # 预处理，1) 移动点云坐标 2)改变框的位置








