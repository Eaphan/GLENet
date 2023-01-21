from pathlib import Path

import numba
import numpy as np
from .geometry import (
    points_count_convex_polygon_3d_jit,
    points_in_convex_polygon_3d_jit,
)
from spconv.utils import rbbox_iou


def points_count_rbbox(points, rbbox, z_axis=2, origin=(0.5, 0.5, 0.5)):
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, -1], origin=origin, axis=z_axis
    )
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    return points_count_convex_polygon_3d_jit(points[:, :3], surfaces)


def riou_cc(rbboxes, qrbboxes, standup_thresh=0.0):
    # less than 50ms when used in second one thread. 10x slower than gpu
    boxes_corners = center_to_corner_box2d(
        rbboxes[:, :2], rbboxes[:, 2:4], rbboxes[:, 4]
    )
    boxes_standup = corner_to_standup_nd(boxes_corners)
    qboxes_corners = center_to_corner_box2d(
        qrbboxes[:, :2], qrbboxes[:, 2:4], qrbboxes[:, 4]
    )
    qboxes_standup = corner_to_standup_nd(qboxes_corners)
    # if standup box not overlapped, rbbox not overlapped too.
    standup_iou = iou_jit(boxes_standup, qboxes_standup, eps=0.0)
    return rbbox_iou(boxes_corners, qboxes_corners, standup_iou, standup_thresh)


# def rinter_cc(rbboxes, qrbboxes, standup_thresh=0.0):
#     # less than 50ms when used in second one thread. 10x slower than gpu
#     boxes_corners = center_to_corner_box2d(
#         rbboxes[:, :2], rbboxes[:, 2:4], rbboxes[:, 4]
#     )
#     boxes_standup = corner_to_standup_nd(boxes_corners)
#     qboxes_corners = center_to_corner_box2d(
#         qrbboxes[:, :2], qrbboxes[:, 2:4], qrbboxes[:, 4]
#     )
#     qboxes_standup = corner_to_standup_nd(qboxes_corners)
#     # if standup box not overlapped, rbbox not overlapped too.
#     standup_iou = iou_jit(boxes_standup, qboxes_standup, eps=0.0)
#     return rbbox_intersection(
#         boxes_corners, qboxes_corners, standup_iou, standup_thresh
#     )


def second_box_encode(
    boxes,            # [num_pos_anchor, 7]
    anchors,          # [num_pos_anchor, 7]
    encode_angle_to_vector=False,
    smooth_dim=False,
    cylindrical=False,
    norm_velo=False,
):
    """box encode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """
    # need to convert boxes to z-center format
    box_ndim = anchors.shape[-1]         # 7

    if box_ndim == 7:
        xa, ya, za, wa, la, ha, ra = np.split(anchors, box_ndim, axis=1)   # [num_pos_anchor, 1], [num_pos_anchor, 1] ...
        xg, yg, zg, wg, lg, hg, rg = np.split(boxes, box_ndim, axis=1)     # [num_pos_anchor, 1], [num_pos_anchor, 1] ...
    else:
        xa, ya, za, wa, la, ha, vxa, vya, ra = np.split(anchors, box_ndim, axis=1)
        xg, yg, zg, wg, lg, hg, vxg, vyg, rg = np.split(boxes, box_ndim, axis=1)

    diagonal = np.sqrt(la ** 2 + wa ** 2)  # [num_pos_anchor, 1], 4.215
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / ha

    if smooth_dim:  # False
        lt = lg / la - 1
        wt = wg / wa - 1
        ht = hg / ha - 1
    else:
        lt = np.log(lg / la)
        wt = np.log(wg / wa)
        ht = np.log(hg / ha)

    ret = [xt, yt, zt, wt, lt, ht]

    if box_ndim > 7:  # False
        if norm_velo:
            vxt = (vxg - vxa) / diagonal
            vyt = (vyg - vya) / diagonal
        else:
            vxt = vxg - vxa
            vyt = vyg - vya
        ret.extend([vxt, vyt])

    if encode_angle_to_vector:  # False
        rgx = np.cos(rg)
        rgy = np.sin(rg)
        rax = np.cos(ra)
        ray = np.sin(ra)
        rtx = rgx - rax
        rty = rgy - ray

        ret.extend([rtx, rty])
    else:
        rt = rg - ra
        ret.append(rt)

    return np.concatenate(ret, axis=1)   # [num_pos_anchor, 7]


def second_box_decode(
    box_encodings,
    anchors,
    encode_angle_to_vector=False,
    smooth_dim=False,
    cylindrical=False,
    norm_velo=False,
):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 9] Tensor): normal boxes: x, y, z, w, l, h, vx, vy, r
        anchors ([N, 9] Tensor): anchors
    """
    # need to convert box_encodings to z-bottom format
    box_ndim = anchors.shape[-1]

    if box_ndim > 7:
        xa, ya, za, wa, la, ha, vxa, vya, ra = np.split(anchors, box_ndim, axis=1)
        if encode_angle_to_vector:
            xt, yt, zt, wt, lt, ht, vxt, vyt, rtx, rty = np.split(
                box_encodings, box_ndim + 1, axis=-1
            )
        else:
            xt, yt, zt, wt, lt, ht, vxt, vyt, rt = np.split(
                box_encodings, box_ndim, axis=-1
            )
    else:
        xa, ya, za, wa, la, ha, ra = np.split(anchors, box_ndim, axis=-1)
        if encode_angle_to_vector:
            xt, yt, zt, wt, lt, ht, rtx, rty = np.split(
                box_encodings, box_ndim + 1, axis=-1
            )
        else:
            xt, yt, zt, wt, lt, ht, rt = np.split(box_encodings, box_ndim, axis=-1)

    # if cylindrical:
    #     diagonal = np.sqrt(la**2 + wa**2)
    #     xg = xt * diagonal + xa
    #     yg = yt * diagonal + ya
    # else:
    #     diagonal = np.sqrt(la**2 + wa**2)
    #     xg = xt * diagonal + xa
    #     yg = yt * diagonal + ya

    diagonal = np.sqrt(la ** 2 + wa ** 2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za

    ret = [xg, yg, zg]

    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
        hg = (ht + 1) * ha
    else:
        lg = np.exp(lt) * la
        wg = np.exp(wt) * wa
        hg = np.exp(ht) * ha
    ret.extend([wg, lg, hg])

    if encode_angle_to_vector:
        rax = np.cos(ra)
        ray = np.sin(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = np.arctan2(rgy, rgx)
    else:
        rg = rt + ra

    if box_ndim > 7:
        if norm_velo:
            vxg = vxt * diagonal + vxa
            vyg = vyt * diagonal + vya
        else:
            vxg = vxt + vxa
            vyg = vyt + vya
        ret.extend([vxg, vyg])

    ret.append(rg)

    return np.concatenate(ret, axis=1)


def bev_box_encode(boxes, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box encode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
        encode_angle_to_vector: bool. increase aos performance,
            decrease other performance.
    """
    # need to convert boxes to z-center format
    xa, ya, wa, la, ra = np.split(anchors, 5, axis=-1)
    xg, yg, wg, lg, rg = np.split(boxes, 5, axis=-1)
    diagonal = np.sqrt(la ** 2 + wa ** 2)  # 4.3
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    if smooth_dim:
        lt = lg / la - 1
        wt = wg / wa - 1
    else:
        lt = np.log(lg / la)
        wt = np.log(wg / wa)
    if encode_angle_to_vector:
        rgx = np.cos(rg)
        rgy = np.sin(rg)
        rax = np.cos(ra)
        ray = np.sin(ra)
        rtx = rgx - rax
        rty = rgy - ray
        return np.concatenate([xt, yt, wt, lt, rtx, rty], axis=-1)
    else:
        rt = rg - ra
        return np.concatenate([xt, yt, wt, lt, rt], axis=-1)


def bev_box_decode(
    box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False
):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """
    # need to convert box_encodings to z-bottom format
    xa, ya, wa, la, ra = np.split(anchors, 5, axis=-1)
    if encode_angle_to_vector:
        xt, yt, wt, lt, rtx, rty = np.split(box_encodings, 6, axis=-1)
    else:
        xt, yt, wt, lt, rt = np.split(box_encodings, 5, axis=-1)
    diagonal = np.sqrt(la ** 2 + wa ** 2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
    else:
        lg = np.exp(lt) * la
        wg = np.exp(wt) * wa
    if encode_angle_to_vector:
        rax = np.cos(ra)
        ray = np.sin(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = np.arctan2(rgy, rgx)
    else:
        rg = rt + ra
    return np.concatenate([xg, yg, wg, lg, rg], axis=-1)


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:  # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2 ** ndim, ndim])
    return corners


@numba.njit
def corners_2d_jit(dims, origin=0.5):
    ndim = 2
    corners_norm = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=dims.dtype)
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape((-1, 1, ndim)) * corners_norm.reshape((1, 2 ** ndim, ndim))
    return corners


@numba.njit
def corners_3d_jit(dims, origin=0.5):
    ndim = 3
    corners_norm = np.array(
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        dtype=dims.dtype,
    ).reshape((8, 3))
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape((-1, 1, ndim)) * corners_norm.reshape((1, 2 ** ndim, ndim))
    return corners


@numba.njit
def corner_to_standup_nd_jit(boxes_corner):
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result


@numba.njit
def far_points_first(points, dist_thres, max_voxel_num, shuffle=False):
    indices = np.arange(points.shape[0])
    points_dist = np.sqrt((points[:, :3]**2).sum(1))
    far_points_mask = points_dist > dist_thres
    near_points_mask = np.logical_not(far_points_mask)
    far_points_indices = indices[far_points_mask]
    near_points_indices = indices[near_points_mask]
    points_indices = np.concatenate((far_points_indices, near_points_indices), axis=0)
    if shuffle:
        np.random.shuffle(points_indices[:max_voxel_num])
    points = points[points_indices]
    return points

def corner_to_standup_nd(boxes_corner):
    assert len(boxes_corner.shape) == 3
    standup_boxes = []
    standup_boxes.append(np.min(boxes_corner, axis=1))
    standup_boxes.append(np.max(boxes_corner, axis=1))
    return np.concatenate(standup_boxes, -1)


def rbbox2d_to_near_bbox(rbboxes):
    """convert rotated bbox to nearest 'standing' or 'lying' bbox.
    Args:
        rbboxes: [N, 5(x, y, xdim, ydim, rad)] rotated bboxes
    Returns:
        bboxes: [N, 4(xmin, ymin, xmax, ymax)] bboxes
    """
    rots = rbboxes[..., -1]
    rots_0_pi_div_2 = np.abs(limit_period(rots, 0.5, np.pi))   # limit ry in range [-np.pi/2., np.pi/2.]
    cond = (rots_0_pi_div_2 > np.pi / 4)[..., np.newaxis]
    bboxes_center = np.where(cond, rbboxes[:, [0, 1, 3, 2]], rbboxes[:, :4])  # if True, change w and l; otherwise keep the same;
    bboxes = center_to_minmax_2d(bboxes_center[:, :2], bboxes_center[:, 2:])
    return bboxes


def rotation_3d_in_axis(points, angles, axis=0):
    '''
        points: [num_point, 3] / [1, 8, 3]
        angles: [num_gt_boxes,]
    '''
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack(
            [
                [rot_cos, zeros, -rot_sin],
                [zeros, ones, zeros],
                [rot_sin, zeros, rot_cos],
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack(
            [
                [rot_cos, -rot_sin, zeros],
                [rot_sin, rot_cos, zeros],
                [zeros, zeros, ones],
            ]
        )
    elif axis == 0:
        rot_mat_T = np.stack(
            [
                [zeros, rot_cos, -rot_sin],
                [zeros, rot_sin, rot_cos],
                [ones, zeros, zeros],
            ]
        )
    else:
        raise ValueError("axis should in range")

    return np.einsum("aij,jka->aik", points, rot_mat_T)   # (N, 8, 3) x (3, 3, num_angles) -> (N, 8, 3)


def rotation_points_single_angle(points, angle, axis=0):
    # points: [N, 3]
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    if axis == 1:
        rot_mat_T = np.array(
            [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
            dtype=points.dtype,
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.array(
            [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
            dtype=points.dtype,
        )
    elif axis == 0:
        rot_mat_T = np.array(
            [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
            dtype=points.dtype,
        )
    else:
        raise ValueError("axis should in range")

    return points @ rot_mat_T


def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def rotation_box(box_corners, angle):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angle (float): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T = np.array(
        [[rot_cos, -rot_sin], [rot_sin, rot_cos]], dtype=box_corners.dtype
    )
    return box_corners @ rot_mat_T


def center_to_corner_box3d(centers, dims, angles=None, origin=(0.5, 0.5, 0.5), axis=2):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    '''
    corners (in velo coord): [N, 8, 3]
            6 -------- 5
           /|         /|
          2 -------- 1 .
          | |        | |
          . 7 -------- 4
          |/         |/
          3 -------- 0
    corners_norm:
      [[-0.5, -0.5, -0.5],
       [-0.5, -0.5,  0.5],
       [-0.5,  0.5,  0.5],
       [-0.5,  0.5, -0.5],
       [ 0.5, -0.5, -0.5],
       [ 0.5, -0.5,  0.5],
       [ 0.5,  0.5,  0.5],
       [ 0.5,  0.5, -0.5]]
       
    corners = dims.reshape([-1, 1, 3]) * corners_norm  # dims: wlh (xyz), corners_norm.shape = [1, 8, 3]
    '''
    corners = corners_nd(dims, origin=origin)   # [w, l, h] * corners_norm
    if angles is not None:  # True
        # if ry > 0, then rotate clockwisely; otherwise, rotate anti-clockwisely; remember: initial is 0 degree.
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners



def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(clockwise when positive)

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.

    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners


@numba.jit(nopython=True)
def box2d_to_corner_jit(boxes):
    '''
        Transform the 2d box [x,y,w,l,ry] to the bev 2d-corners,
        with rotation of ry-angle clockwise and translation of real-center.

            1 ------ 2
           /        /
          0 ------ 3
        0: [-0.5, -0.5]
        1: [-0.5, 0.5]
        2: [0.5, 0.5]
        3: [0.5, -0.5]
    '''
    num_box = boxes.shape[0]
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners = boxes.reshape(num_box, 1, 5)[:, :, 2:4] * corners_norm.reshape(1, 4, 2)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    box_corners = np.zeros((num_box, 4, 2), dtype=boxes.dtype)
    for i in range(num_box):
        rot_sin = np.sin(boxes[i, -1])
        rot_cos = np.cos(boxes[i, -1])
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = rot_sin
        rot_mat_T[1, 0] = -rot_sin
        rot_mat_T[1, 1] = rot_cos
        box_corners[i] = corners[i] @ rot_mat_T + boxes[i, :2]   # rotate clockwisely.
    return box_corners


def rbbox3d_to_corners(rbboxes, origin=[0.5, 0.5, 0.5], axis=2):
    return center_to_corner_box3d(
        rbboxes[..., :3], rbboxes[..., 3:6], rbboxes[..., 6], origin, axis=axis
    )


def rbbox3d_to_bev_corners(rbboxes, origin=0.5):
    return center_to_corner_box2d(
        rbboxes[..., :2], rbboxes[..., 3:5], rbboxes[..., 6], origin
    )


def minmax_to_corner_2d(minmax_box):
    ndim = minmax_box.shape[-1] // 2
    center = minmax_box[..., :ndim]
    dims = minmax_box[..., ndim:] - center
    return center_to_corner_box2d(center, dims, origin=0.0)


def minmax_to_corner_2d_v2(minmax_box):
    # N, 4 -> N 4 2
    return minmax_box[..., [0, 1, 0, 3, 2, 3, 2, 1]].reshape(-1, 4, 2)


def minmax_to_corner_3d(minmax_box):
    ndim = minmax_box.shape[-1] // 2
    center = minmax_box[..., :ndim]
    dims = minmax_box[..., ndim:] - center
    return center_to_corner_box3d(center, dims, origin=0.0)


def minmax_to_center_2d(minmax_box):
    ndim = minmax_box.shape[-1] // 2
    center_min = minmax_box[..., :ndim]
    dims = minmax_box[..., ndim:] - center_min
    center = center_min + 0.5 * dims
    return np.concatenate([center, dims], axis=-1)


def center_to_minmax_2d_0_5(centers, dims):
    return np.concatenate([centers - dims / 2, centers + dims / 2], axis=-1)


def center_to_minmax_2d(centers, dims, origin=0.5):
    if origin == 0.5:
        return center_to_minmax_2d_0_5(centers, dims)
    corners = center_to_corner_box2d(centers, dims, origin=origin)
    return corners[:, [0, 2]].reshape([-1, 4])


def limit_period(val, offset=0.5, period=2 * np.pi):
    return val - np.floor(val / period + offset) * period


def projection_matrix_to_CRT_kitti(proj):
    # P = C @ [R|T]
    # C is upper triangular matrix, so we need to inverse CR and use QR
    # stable for all kitti camera projection matrix
    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    return C, R, T


def get_frustum(bbox_image, C, near_clip=0.001, far_clip=100):
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    z_points = np.array([near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[:, np.newaxis]
    b = bbox_image
    box_corners = np.array(
        [[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]], dtype=C.dtype
    )
    near_box_corners = (box_corners - u0v0) / np.array(
        [fku / near_clip, -fkv / near_clip], dtype=C.dtype
    )
    far_box_corners = (box_corners - u0v0) / np.array(
        [fku / far_clip, -fkv / far_clip], dtype=C.dtype
    )
    ret_xy = np.concatenate([near_box_corners, far_box_corners], axis=0)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=1)
    return ret_xyz


def get_frustum_v2(bboxes, C, near_clip=0.001, far_clip=100):
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    num_box = bboxes.shape[0]
    z_points = np.array([near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[
        np.newaxis, :, np.newaxis
    ]
    z_points = np.tile(z_points, [num_box, 1, 1])
    box_corners = minmax_to_corner_2d_v2(bboxes)
    near_box_corners = (box_corners - u0v0) / np.array(
        [fku / near_clip, -fkv / near_clip], dtype=C.dtype
    )
    far_box_corners = (box_corners - u0v0) / np.array(
        [fku / far_clip, -fkv / far_clip], dtype=C.dtype
    )
    ret_xy = np.concatenate([near_box_corners, far_box_corners], axis=1)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=-1)
    return ret_xyz


def create_anchors_3d_stride(
    feature_size,
    sizes=[1.6, 3.9, 1.56],
    anchor_strides=[0.4, 0.4, 0.0],
    anchor_offsets=[0.2, -39.8, -1.78],
    rotations=[0, np.pi / 2],
    velocities=[],
    dtype=np.float32,
):
    """
    Args:
        feature_size: list [D, H, W](zyx)
        sizes: [N, 3] list of list or array, size of anchors, xyz

    Returns:
        anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
    """
    # almost 2x faster than v1
    x_stride, y_stride, z_stride = anchor_strides
    x_offset, y_offset, z_offset = anchor_offsets
    z_centers = np.arange(feature_size[0], dtype=dtype)
    y_centers = np.arange(feature_size[1], dtype=dtype)
    x_centers = np.arange(feature_size[2], dtype=dtype)
    z_centers = z_centers * z_stride + z_offset
    y_centers = y_centers * y_stride + y_offset
    x_centers = x_centers * x_stride + x_offset
    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 3])
    rotations = np.array(rotations, dtype=dtype)
    velocities = np.array(velocities, dtype=dtype).reshape([-1, 2])

    combines = np.hstack([sizes, velocities]).reshape([-1, 5])

    rets = np.meshgrid(x_centers, y_centers, z_centers, rotations, indexing="ij")
    tile_shape = [1] * 5
    tile_shape[-2] = int(sizes.shape[0])
    for i in range(len(rets)):
        rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
        rets[i] = rets[i][..., np.newaxis]  # for concat
    # sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    combines = np.reshape(combines, [1, 1, 1, -1, 1, 5])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    # sizes = np.tile(sizes, tile_size_shape)
    combines = np.tile(combines, tile_size_shape)

    # rets.insert(3, sizes)
    rets.insert(3, combines)

    ret = np.concatenate(rets, axis=-1)
    return np.transpose(ret, [2, 1, 0, 3, 4, 5])


def create_anchors_bev_stride(
    feature_size,
    sizes=[1.6, 3.9],
    anchor_strides=[0.4, 0.4],
    anchor_offsets=[0.2, -39.8],
    rotations=[0, np.pi / 2],
    velocities=[],
    dtype=np.float32,
):
    """
    Args:
        feature_size: list [D, H, W](zyx)
        sizes: [N, 3] list of list or array, size of anchors, xyz

    Returns:
        anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
    """
    # almost 2x faster than v1
    x_stride, y_stride = anchor_strides
    x_offset, y_offset = anchor_offsets
    y_centers = np.arange(feature_size[0], dtype=dtype)
    x_centers = np.arange(feature_size[1], dtype=dtype)
    y_centers = y_centers * y_stride + y_offset
    x_centers = x_centers * x_stride + x_offset
    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 2])
    rotations = np.array(rotations, dtype=dtype)
    velocities = np.array(velocities, dtype=dtype).reshape([-1, 2])

    combines = np.hstack([sizes, velocities]).reshape([-1, 4])

    rets = np.meshgrid(x_centers, y_centers, rotations, indexing="ij")
    tile_shape = [1] * 4
    tile_shape[-2] = int(sizes.shape[0])
    for i in range(len(rets)):
        rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
        rets[i] = rets[i][..., np.newaxis]  # for concat
    # sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    combines = np.reshape(combines, [1, 1, 1, -1, 1, 4])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    # sizes = np.tile(sizes, tile_size_shape)
    combines = np.tile(combines, tile_size_shape)

    # rets.insert(3, sizes)
    rets.insert(3, combines)

    ret = np.concatenate(rets, axis=-1)
    return np.transpose(ret, [2, 1, 0, 3, 4, 5])


def create_anchors_3d_range(
    feature_size,             # [1, 200, 176]
    anchor_range,             # [0, -40.0, -1.0, 70.4, 40.0, -1.0]
    sizes=[1.6, 3.9, 1.56],   # w, l, h
    rotations=[0, np.pi / 2],
    velocities=None,
    dtype=np.float32,
):
    """
    Args:
        feature_size: list [D, H, W](zyx).
        sizes: [N, 3] list of list or array, size of anchors, xyz.
        rotations: len(stride) num Reference.
        velocities: ref velo along x and y axis.

    Returns:
        anchors: [*feature_size, num_sizes, num_rots, 7] -> [1, 200, 176, 1, 2, 7]

    """
    # create mesh of anchor centers
    anchor_range = np.array(anchor_range, dtype)
    stride = (anchor_range[3] - anchor_range[0]) / feature_size[2]    # (70.4 - 0) / 176 = 0.4
    z_centers =  np.linspace(anchor_range[2], anchor_range[5], feature_size[0], dtype=dtype)   # [-1, -1]
    y_centers = (np.linspace(anchor_range[1], anchor_range[4], feature_size[1], endpoint=False, dtype=dtype,) + stride / 2)  # [-40, 40], move to center
    x_centers = (np.linspace(anchor_range[0], anchor_range[3], feature_size[2], endpoint=False, dtype=dtype,) + stride / 2)  # [0, 70.4]
    rotations = np.array(rotations, dtype=dtype)                      # [0. 1.57]
    rets = np.meshgrid(x_centers, y_centers, z_centers, rotations, indexing="ij")  # rets[i]: (176, 200, 1, 2)

    # unknown operations
    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 3])   # [1.6, 3.9, 1.56]
    tile_shape = [1] * 5
    tile_shape[-2] = int(sizes.shape[0])     # [1, 1, 1, 1, 1]

    for i in range(len(rets)):  # len: 4
        rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
        rets[i] = rets[i][..., np.newaxis]   # rets[i]: (176, 200, 1, 2) -> (176, 200, 1, +1, 2, +1)

    # prepare anchor size (w, l, h)
    if velocities is not None: # False
        velocities = np.array(velocities, dtype=dtype).reshape([-1, 2])
        combines = np.hstack([sizes, velocities]).reshape([-1, 5])
    else:
        combines = sizes        # [1, 3]: [1.6, 3.9, 1.56]

    combines = np.reshape(combines, [1, 1, 1, -1, 1, combines.shape[-1]])   # [1, 1, 1, 1, 1, 3]
    tile_size_shape = list(rets[0].shape)  # [176, 200, 1, 1, 2, 1]
    tile_size_shape[3] = 1                 # [176, 200, 1, 1, 2, 1]
    combines = np.tile(combines, tile_size_shape)  # (176, 200, 1, 1, 2, 3):  [1.6, 3.9, 1.56]

    # get anchors <- [x, y, z, inserted: [w, l, h], r]
    rets.insert(3, combines)
    ret = np.concatenate(rets, axis=-1)  # (176, 200, 1, 1, 2, 7)

    return np.transpose(ret, [2, 1, 0, 3, 4, 5])    #  [*feature_size, num_sizes, num_rots, 7]: (1, 200, 176, 1, 2, 7)


def create_anchors_bev_range(
    feature_size,
    anchor_range,
    sizes=[1.6, 3.9],
    rotations=[0, np.pi / 2],
    velocities=None,
    dtype=np.float32,
):
    """
    Args:
        feature_size: list [D, H, W](zyx)
        sizes: [N, 3] list of list or array, size of anchors, xyz
        rotations: len(stride) num Reference
        velocities: ref velo along x and y axis.

    Returns:
        anchors: [*feature_size, num_sizes, num_rots, 9] tensor.
    """
    anchor_range = np.array(anchor_range, dtype)
    stride = (anchor_range[2] - anchor_range[0]) / feature_size[1]

    y_centers = (
        np.linspace(
            anchor_range[1],
            anchor_range[3],
            feature_size[0],
            endpoint=False,
            dtype=dtype,
        )
        + stride / 2
    )
    x_centers = (
        np.linspace(
            anchor_range[0],
            anchor_range[2],
            feature_size[1],
            endpoint=False,
            dtype=dtype,
        )
        + stride / 2
    )
    rotations = np.array(rotations, dtype=dtype)
    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 2])

    if velocities is not None:
        velocities = np.array(velocities, dtype=dtype).reshape([-1, 2])
        combines = np.hstack([sizes, velocities]).reshape([-1, 4])
    else:
        combines = sizes

    rets = np.meshgrid(x_centers, y_centers, rotations, indexing="ij")

    tile_shape = [1] * 4
    # tile_shape[-2] = int(sizes.shape[0])
    for i in range(len(rets)):
        rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
        rets[i] = rets[i][..., np.newaxis]  # for concat
    # sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    combines = np.reshape(combines, [1, 1, -1, 1, combines.shape[-1]])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[2] = 1
    # sizes = np.tile(sizes, tile_size_shape)
    combines = np.tile(combines, tile_size_shape)

    rets.insert(2, combines)

    ret = np.concatenate(rets, axis=-1)
    return np.transpose(ret, [1, 0, 2, 3, 4])


@numba.njit
def _add_rgb_to_points_kernel(points_2d, image, points_rgb):
    num_points = points_2d.shape[0]
    image_h, image_w = image.shape[:2]
    for i in range(num_points):
        img_pos = np.floor(points_2d[i]).astype(np.int32)
        if img_pos[0] >= 0 and img_pos[0] < image_w:
            if img_pos[1] >= 0 and img_pos[1] < image_h:
                points_rgb[i, :] = image[img_pos[1], img_pos[0], :]
                # image[img_pos[1], img_pos[0]] = 0


def add_rgb_to_points(points, image, rect, Trv2c, P2, mean_size=[5, 5]):
    kernel = np.ones(mean_size, np.float32) / np.prod(mean_size)
    # image = cv2.filter2D(image, -1, kernel)
    points_cam = lidar_to_camera(points[:, :3], rect, Trv2c)
    points_2d = project_to_image(points_cam, P2)
    points_rgb = np.zeros([points_cam.shape[0], 3], dtype=points.dtype)
    _add_rgb_to_points_kernel(points_2d, image, points_rgb)
    return points_rgb


def project_to_image(points_3d, proj_mat):
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1
    points_4 = np.concatenate([points_3d, np.zeros(points_shape)], axis=-1)
    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res


def camera_to_lidar(points, r_rect, velo2cam):
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
    return lidar_points[..., :3]


def lidar_to_camera(points, r_rect, velo2cam):
    '''
       r_rect[4,4]:
                [[ 0.9999239 ,  0.00983776, -0.00744505,  0.        ],
                 [-0.0098698 ,  0.9999421 , -0.00427846,  0.        ],
                 [ 0.00740253,  0.00435161,  0.9999631 ,  0.        ],
                 [ 0.        ,  0.        ,  0.        ,  1.        ]]
       velo2cam[4,4]:
               [[ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
               [ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
               [ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],
               [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]]
    '''
    points_shape = list(points.shape[:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    camera_points = points @ (r_rect @ velo2cam).T
    return camera_points[..., :3]


def box_camera_to_lidar(data, r_rect, velo2cam):
    xyz = data[:, 0:3]
    l, h, w = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    r = data[:, 6:7]
    xyz_lidar = camera_to_lidar(xyz, r_rect, velo2cam)
    return np.concatenate([xyz_lidar, w, l, h, r], axis=1)


def box_lidar_to_camera(data, r_rect, velo2cam):
    xyz_lidar = data[:, 0:3]
    w, l, h = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    r = data[:, 6:7]
    xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    return np.concatenate([xyz, l, h, w, r], axis=1)


def remove_outside_points(points, rect, Trv2c, P2, image_shape):
    # 5x faster than remove_outside_points_v1(2ms vs 10ms)
    C, R, T = projection_matrix_to_CRT_kitti(P2)
    image_bbox = [0, 0, image_shape[1], image_shape[0]]
    frustum = get_frustum(image_bbox, C)
    frustum -= T
    frustum = np.linalg.inv(R) @ frustum.T
    frustum = camera_to_lidar(frustum.T, rect, Trv2c)
    frustum_surfaces = corner_to_surfaces_3d_jit(frustum[np.newaxis, ...])
    indices = points_in_convex_polygon_3d_jit(points[:, :3], frustum_surfaces)
    points = points[indices.reshape([-1])]
    return points


def get_valid_frustum(rect, Trv2c, P2, image_shape):
    C, R, T = projection_matrix_to_CRT_kitti(P2)
    image_bbox = [0, 0, image_shape[1], image_shape[0]]
    frustum = get_frustum(image_bbox, C)
    frustum -= T
    frustum = np.linalg.inv(R) @ frustum.T
    frustum = camera_to_lidar(frustum.T, rect, Trv2c)
    frustum_surfaces = corner_to_surfaces_3d_jit(frustum[np.newaxis, ...])
    return frustum_surfaces



@numba.jit(nopython=True)
def iou_jit(boxes, query_boxes, eps=1.0):
    """calculate box iou. note that jit version runs 2x faster than cython in
    my machine!
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        box_area = (query_boxes[k, 2] - query_boxes[k, 0] + eps) * (
            query_boxes[k, 3] - query_boxes[k, 1] + eps
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2])
                - max(boxes[n, 0], query_boxes[k, 0])
                + eps
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3])
                    - max(boxes[n, 1], query_boxes[k, 1])
                    + eps
                )
                if ih > 0:
                    ua = (
                        (boxes[n, 2] - boxes[n, 0] + eps)
                        * (boxes[n, 3] - boxes[n, 1] + eps)
                        + box_area
                        - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps


@numba.jit(nopython=True)
def iou_3d_jit(boxes, query_boxes, add1=True):
    """calculate box iou3d,
    ----------
    boxes: (N, 6) ndarray of float
    query_boxes: (K, 6) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    if add1:
        add1 = 1.0
    else:
        add1 = 0.0
    for k in range(K):
        box_area = (
            (query_boxes[k, 3] - query_boxes[k, 0] + add1)
            * (query_boxes[k, 4] - query_boxes[k, 1] + add1)
            * (query_boxes[k, 5] - query_boxes[k, 2] + add1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 3], query_boxes[k, 3])
                - max(boxes[n, 0], query_boxes[k, 0])
                + add1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 4], query_boxes[k, 4])
                    - max(boxes[n, 1], query_boxes[k, 1])
                    + add1
                )
                if ih > 0:
                    il = (
                        min(boxes[n, 5], query_boxes[k, 5])
                        - max(boxes[n, 2], query_boxes[k, 2])
                        + add1
                    )
                    if il > 0:
                        ua = float(
                            (boxes[n, 3] - boxes[n, 0] + add1)
                            * (boxes[n, 4] - boxes[n, 1] + add1)
                            * (boxes[n, 5] - boxes[n, 2] + add1)
                            + box_area
                            - iw * ih * il
                        )
                        overlaps[n, k] = iw * ih * il / ua
    return overlaps


@numba.jit(nopython=True)
def iou_nd_jit(boxes, query_boxes, add1=True):
    """calculate box iou nd, 2x slower than iou_jit.
    ----------
    boxes: (N, ndim * 2) ndarray of float
    query_boxes: (K, ndim * 2) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    ndim = boxes.shape[1] // 2
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    side_lengths = np.zeros((ndim,), dtype=boxes.dtype)
    if add1:
        add1 = 1.0
    else:
        add1 = 0.0
    invalid = False
    for k in range(K):
        qbox_area = query_boxes[k, ndim] - query_boxes[k, 0] + add1
        for i in range(1, ndim):
            qbox_area *= query_boxes[k, ndim + i] - query_boxes[k, i] + add1
        for n in range(N):
            invalid = False
            for i in range(ndim):
                side_length = (
                    min(boxes[n, i + ndim], query_boxes[k, i + ndim])
                    - max(boxes[n, i], query_boxes[k, i])
                    + add1
                )
                if side_length <= 0:
                    invalid = True
                    break
                side_lengths[i] = side_length
            if not invalid:
                box_area = boxes[n, ndim] - boxes[n, 0] + add1
                for i in range(1, ndim):
                    box_area *= boxes[n, ndim + i] - boxes[n, i] + add1
                inter = side_lengths[0]
                for i in range(1, ndim):
                    inter *= side_lengths[i]
                # inter = np.prod(side_lengths)
                ua = float(box_area + qbox_area - inter)
                overlaps[n, k] = inter / ua

    return overlaps


def points_in_rbbox(points, rbbox, z_axis=2, origin=(0.5, 0.5, 0.5)):
    # get point indexs in the rotated boxes
    rbbox_corners = center_to_corner_box3d(rbbox[:, :3], rbbox[:, 3:6], rbbox[:, -1], origin=origin, axis=z_axis)
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices


def corner_to_surfaces_3d(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.

    corners (in velo coord): [N, 8, 3]
            6 -------- 5
           /|         /|
          2 -------- 1 .
          | |        | |
          . 7 -------- 4
          |/         |/
          3 -------- 0

    Args:
        corners (float array, [N, 8, 3]): 3d box corners.
    Returns:
        surfaces (float array, [N, 6, 4, 3]):
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    surfaces = np.array(
        [
            [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
            [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
            [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
            [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
            [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
            [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
        ]
    ).transpose([2, 0, 1, 3])
    return surfaces

@numba.jit(nopython=True)
def corner_to_surfaces_3d_jit(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.

    Args:
        corners (float array, [N, 8, 3]): 3d box corners.
    Returns:
        surfaces (float array, [N, 6, 4, 3]):
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    # print("### corners = ", corners)
    num_boxes = corners.shape[0]
    surfaces = np.zeros((num_boxes, 6, 4, 3), dtype=corners.dtype)
    corner_idxes = np.array(
        [0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 1, 5, 6, 2, 0, 4, 5, 1, 3, 2, 6, 7]
    ).reshape(6, 4)
    for i in range(num_boxes):
        for j in range(6):
            for k in range(4):
                surfaces[i, j, k] = corners[i, corner_idxes[j, k]]
    # print("### surfaces = ", surfaces)
    return surfaces


def assign_label_to_voxel(gt_boxes, coors, voxel_size, coors_range):
    """assign a 0/1 label to each voxel based on whether
    the center of voxel is in gt_box. LIDAR.
    """
    voxel_size = np.array(voxel_size, dtype=gt_boxes.dtype)
    coors_range = np.array(coors_range, dtype=gt_boxes.dtype)
    shift = coors_range[:3]
    voxel_origins = coors[:, ::-1] * voxel_size + shift
    voxel_centers = voxel_origins + voxel_size * 0.5
    gt_box_corners = center_to_corner_box3d(
        gt_boxes[:, :3] - voxel_size * 0.5,
        gt_boxes[:, 3:6] + voxel_size,
        gt_boxes[:, 6],
        origin=[0.5, 0.5, 0.5],
        axis=2,
    )
    gt_surfaces = corner_to_surfaces_3d(gt_box_corners)
    ret = points_in_convex_polygon_3d_jit(voxel_centers, gt_surfaces)
    return np.any(ret, axis=1).astype(np.int64)


def assign_label_to_voxel_v3(gt_boxes, coors, voxel_size, coors_range):
    """assign a 0/1 label to each voxel based on whether
    the center of voxel is in gt_box. LIDAR.
    """
    voxel_size = np.array(voxel_size, dtype=gt_boxes.dtype)
    coors_range = np.array(coors_range, dtype=gt_boxes.dtype)
    shift = coors_range[:3]
    voxel_origins = coors[:, ::-1] * voxel_size + shift
    voxel_maxes = voxel_origins + voxel_size
    voxel_minmax = np.concatenate([voxel_origins, voxel_maxes], axis=-1)
    voxel_corners = minmax_to_corner_3d(voxel_minmax)
    gt_box_corners = center_to_corner_box3d(
        gt_boxes[:, :3],
        gt_boxes[:, 3:6],
        gt_boxes[:, 6],
        origin=[0.5, 0.5, 0.5],
        axis=2,
    )
    gt_surfaces = corner_to_surfaces_3d(gt_box_corners)
    voxel_corners_flat = voxel_corners.reshape([-1, 3])
    ret = points_in_convex_polygon_3d_jit(voxel_corners_flat, gt_surfaces)
    ret = ret.reshape([-1, 8, ret.shape[-1]])
    return ret.any(-1).any(-1).astype(np.int64)


def image_box_region_area(img_cumsum, bbox):
    """check a 2d voxel is contained by a box. used to filter empty
    anchors.
    Summed-area table algorithm:
    ==> W
    ------------------
    |      |         |
    |------A---------B
    |      |         |
    |      |         |
    |----- C---------D
    Iabcd = ID-IB-IC+IA
    Args:
        img_cumsum: [M, H, W](yx) cumsumed image.
        bbox: [N, 4](xyxy) bounding box,
    """
    N = bbox.shape[0]
    M = img_cumsum.shape[0]
    ret = np.zeros([N, M], dtype=img_cumsum.dtype)
    ID = img_cumsum[:, bbox[:, 3], bbox[:, 2]]
    IA = img_cumsum[:, bbox[:, 1], bbox[:, 0]]
    IB = img_cumsum[:, bbox[:, 3], bbox[:, 0]]
    IC = img_cumsum[:, bbox[:, 1], bbox[:, 2]]
    ret = ID - IB - IC + IA
    return ret


def get_minimum_bounding_box_bv(points, voxel_size, bound, downsample=8, margin=1.6):
    x_vsize = voxel_size[0]
    y_vsize = voxel_size[1]
    max_x = points[:, 0].max()
    max_y = points[:, 1].max()
    min_x = points[:, 0].min()
    min_y = points[:, 1].min()
    max_x = np.floor(max_x / (x_vsize * downsample) + 1) * (x_vsize * downsample)
    max_y = np.floor(max_y / (y_vsize * downsample) + 1) * (y_vsize * downsample)
    min_x = np.floor(min_x / (x_vsize * downsample)) * (x_vsize * downsample)
    min_y = np.floor(min_y / (y_vsize * downsample)) * (y_vsize * downsample)
    max_x = np.minimum(max_x + margin, bound[2])
    max_y = np.minimum(max_y + margin, bound[3])
    min_x = np.maximum(min_x - margin, bound[0])
    min_y = np.maximum(min_y - margin, bound[1])
    return np.array([min_x, min_y, max_x, max_y])


@numba.jit(nopython=True)
def get_anchor_bv_in_feature_jit(anchors_bv, voxel_size, coors_range, grid_size):
    anchors_bv_coors = np.zeros(anchors_bv.shape, dtype=np.int32)
    anchor_coor = np.zeros(anchors_bv.shape[1:], dtype=np.int32)
    grid_size_x = grid_size[0] - 1
    grid_size_y = grid_size[1] - 1
    for i in range(anchors_bv.shape[0]):
        anchor_coor[0] = np.floor((anchors_bv[i, 0] - coors_range[0]) / voxel_size[0])
        anchor_coor[1] = np.floor((anchors_bv[i, 1] - coors_range[1]) / voxel_size[1])
        anchor_coor[2] = np.floor((anchors_bv[i, 2] - coors_range[0]) / voxel_size[0])
        anchor_coor[3] = np.floor((anchors_bv[i, 3] - coors_range[1]) / voxel_size[1])
        anchor_coor[0] = max(anchor_coor[0], 0)
        anchor_coor[1] = max(anchor_coor[1], 0)
        anchor_coor[2] = min(anchor_coor[2], grid_size_x)
        anchor_coor[3] = min(anchor_coor[3], grid_size_y)
        anchors_bv_coors[i] = anchor_coor
    return anchors_bv_coors


def get_anchor_bv_in_feature(anchors_bv, voxel_size, coors_range, grid_size):
    vsize_bv = np.tile(voxel_size[:2], 2)
    anchors_bv[..., [1, 3]] -= coors_range[1]
    anchors_bv_coors = np.floor(anchors_bv / vsize_bv).astype(np.int64)
    anchors_bv_coors[..., [0, 2]] = np.clip(
        anchors_bv_coors[..., [0, 2]], a_max=grid_size[0] - 1, a_min=0
    )
    anchors_bv_coors[..., [1, 3]] = np.clip(
        anchors_bv_coors[..., [1, 3]], a_max=grid_size[1] - 1, a_min=0
    )
    anchors_bv_coors = anchors_bv_coors.reshape([-1, 4])
    return anchors_bv_coors


@numba.jit(nopython=True)
def sparse_sum_for_anchors_mask(coors, shape):
    ret = np.zeros(shape, dtype=np.float32)
    for i in range(coors.shape[0]):
        ret[coors[i, 1], coors[i, 2]] += 1
    return ret


@numba.jit(nopython=True)
def fused_get_anchors_area(dense_map, anchors_bv, stride, offset, grid_size):
    anchor_coor = np.zeros(anchors_bv.shape[1:], dtype=np.int32)
    grid_size_x = grid_size[0] - 1
    grid_size_y = grid_size[1] - 1
    N = anchors_bv.shape[0]
    ret = np.zeros((N), dtype=dense_map.dtype)
    for i in range(N):
        anchor_coor[0] = np.floor((anchors_bv[i, 0] - offset[0]) / stride[0])
        anchor_coor[1] = np.floor((anchors_bv[i, 1] - offset[1]) / stride[1])
        anchor_coor[2] = np.floor((anchors_bv[i, 2] - offset[0]) / stride[0])
        anchor_coor[3] = np.floor((anchors_bv[i, 3] - offset[1]) / stride[1])
        anchor_coor[0] = max(anchor_coor[0], 0)
        anchor_coor[1] = max(anchor_coor[1], 0)
        anchor_coor[2] = min(anchor_coor[2], grid_size_x)
        anchor_coor[3] = min(anchor_coor[3], grid_size_y)
        ID = dense_map[anchor_coor[3], anchor_coor[2]]
        IA = dense_map[anchor_coor[1], anchor_coor[0]]
        IB = dense_map[anchor_coor[3], anchor_coor[0]]
        IC = dense_map[anchor_coor[1], anchor_coor[2]]
        ret[i] = ID - IB - IC + IA
    return ret


@numba.jit(nopython=True)
def distance_similarity(points, qpoints, dist_norm, with_rotation=False, rot_alpha=0.5):
    N = points.shape[0]
    K = qpoints.shape[0]
    dists = np.zeros((N, K), dtype=points.dtype)
    rot_alpha_1 = 1 - rot_alpha
    for k in range(K):
        for n in range(N):
            if np.abs(points[n, 0] - qpoints[k, 0]) <= dist_norm:
                if np.abs(points[n, 1] - qpoints[k, 1]) <= dist_norm:
                    dist = np.sum((points[n, :2] - qpoints[k, :2]) ** 2)
                    dist_normed = min(dist / dist_norm, dist_norm)
                    if with_rotation:
                        dist_rot = np.abs(np.sin(points[n, -1] - qpoints[k, -1]))
                        dists[n, k] = (
                            1 - rot_alpha_1 * dist_normed - rot_alpha * dist_rot
                        )
                    else:
                        dists[n, k] = 1 - dist_normed
    return dists


def box3d_to_bbox(box3d, rect, Trv2c, P2):
    box3d_to_cam = box_lidar_to_camera(box3d, rect, Trv2c)
    box_corners = center_to_corner_box3d(
        box3d[:, :3], box3d[:, 3:6], box3d[:, 6], [0.5, 1.0, 0.5], axis=1
    )
    box_corners_in_image = project_to_image(box_corners, P2)
    # box_corners_in_image: [N, 8, 2]
    minxy = np.min(box_corners_in_image, axis=1)
    maxxy = np.max(box_corners_in_image, axis=1)
    bbox = np.concatenate([minxy, maxxy], axis=1)
    return bbox


def change_box3d_center_(box3d, src, dst):
    dst = np.array(dst, dtype=box3d.dtype)
    src = np.array(src, dtype=box3d.dtype)
    box3d[..., :3] += box3d[..., 3:6] * (dst - src)
