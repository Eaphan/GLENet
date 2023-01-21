"""
3D IoU Calculation and Rotated NMS
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
"""
import torch
import numpy as np

from ...utils import common_utils
from . import iou3d_nms_cuda

# def limit_period_torch(val, offset=0.5, period=2 * np.pi):
#     return val - torch.floor(val / period + offset) * period


# def center_to_minmax_2d_torch(centers, dims):
#     return torch.cat([centers - dims / 2, centers + dims / 2], dim=-1)


# def boxes3d_to_bev_torch(boxes3d,rect=False,box_mode='wlh'):
#     """
#     Input(torch):
#         boxes3d: (N, 7) [x, y, z, h, w, l, ry]
#         rect: True/False means boxes in camera/velodyne coord system.
#     Output:
#         boxes_bev: (N, 5) [x1, y1, x2, y2, ry/rz], left-bottom: (x1, y1), right-top: (x2, y2), ry/rz: clockwise rotation angle
#     """
#     boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))
#     if boxes3d.shape[-1] == 5:
#         w_index, l_index = box_mode.index('w') + 2, box_mode.index('l') + 2
#     elif boxes3d.shape[-1] == 7:
#         w_index, l_index = box_mode.index('w') + 3, box_mode.index('l') + 3
#     else:
#         raise NotImplementedError

#     half_w, half_l = boxes3d[:, w_index] / 2., boxes3d[:, l_index] / 2.
#     if rect:
#         cu, cv = boxes3d[:, 0], boxes3d[:, 2]   # cam coord: x, z
#         boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_l, cv - half_w  # left-bottom in cam coord
#         boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_l, cv + half_w  # right-top in cam coord
#     else:
#         cu, cv = boxes3d[:, 0], boxes3d[:, 1]   # velo coord: x, y
#         boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_w, cv - half_l  # left-bottom in velo coord
#         boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_w, cv + half_l  # right-top in cam coord
#     # rz in velo should have the same effect as ry of cam coord in 2D. Points in box will clockwisely rotate with rz/ry angle.
#     boxes_bev[:, 4] = boxes3d[:, -1]
#     return boxes_bev




def boxes_bev_iou_cpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    boxes_a, is_numpy = common_utils.check_numpy_to_torch(boxes_a)
    boxes_b, is_numpy = common_utils.check_numpy_to_torch(boxes_b)
    assert not (boxes_a.is_cuda or boxes_b.is_cuda), 'Only support CPU tensors'
    assert boxes_a.shape[1] == 7 and boxes_b.shape[1] == 7
    ans_iou = boxes_a.new_zeros(torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    iou3d_nms_cuda.boxes_iou_bev_cpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou.numpy() if is_numpy else ans_iou


def boxes_iou_bev(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_nms_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d


# def boxes_aligned_iou3d_gpu(boxes_a, boxes_b, box_mode='wlh', rect=False, need_bev=False):
#     """
#     Input (torch):
#         boxes_a: (N, 7) [x, y, z, w, l, h, ry], torch tensor with type float32.
#         boxes_b: (N, 7) [x, y, z, w, l, h, ry], torch tensor with type float32.
#         rect: True/False means boxes in camera/velodyne coord system.
#         Notice: (x, y, z) are real center.
#     Output:
#         iou_3d: (N)
#     """
#     assert boxes_a.shape[0] == boxes_b.shape[0]
#     w_index, l_index, h_index = box_mode.index('w') + 3, box_mode.index('l') + 3, box_mode.index('h') + 3
#     boxes_a_bev = boxes3d_to_bev_torch(boxes_a, box_mode, rect)
#     boxes_b_bev = boxes3d_to_bev_torch(boxes_b, box_mode, rect)

#     # bev overlap
#     overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], 1))).zero_()  # (N, 1)
#     iou3d_cuda.boxes_aligned_overlap_bev_gpu(boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), overlaps_bev)

#     # bev iou
#     area_a = (boxes_a[:, w_index] * boxes_a[:, l_index]).view(-1, 1)  # (N, 1)
#     area_b = (boxes_b[:, w_index] * boxes_b[:, l_index]).view(-1, 1)  # (N, 1)
#     iou_bev = overlaps_bev / torch.clamp(area_a + area_b - overlaps_bev, min=1e-7)  # [N, 1]

#     # height overlap
#     if rect:
#         raise NotImplementedError
#         # boxes_a_height_min = (boxes_a[:, 1] - boxes_a[:, h_index]).view(-1, 1)  # y - h
#         # boxes_a_height_max = boxes_a[:, 1].view(-1, 1)                    # y
#         # boxes_b_height_min = (boxes_b[:, 1] - boxes_b[:, h_index]).view(1, -1)
#         # boxes_b_height_max = boxes_b[:, 1].view(1, -1)
#     else:
#         # todo: notice if (x, y, z) is the real center
#         half_h_a = boxes_a[:, h_index] / 2.0
#         half_h_b = boxes_b[:, h_index] / 2.0
#         boxes_a_height_min = (boxes_a[:, 2] - half_h_a).view(-1, 1)  # z - h/2, (N, 1)
#         boxes_a_height_max = (boxes_a[:, 2] + half_h_a).view(-1, 1)  # z + h/2, (N, 1)
#         boxes_b_height_min = (boxes_b[:, 2] - half_h_b).view(-1, 1)
#         boxes_b_height_max = (boxes_b[:, 2] + half_h_b).view(-1, 1)

#     max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)   # (N, 1)
#     min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)   # (N, 1)
#     overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)         # (N, 1)

#     # 3d iou
#     overlaps_3d = overlaps_bev * overlaps_h

#     vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)   # (N, 1)
#     vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(-1, 1)   # (N, 1)

#     iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-7)

#     if need_bev:
#         return iou3d, iou_bev

#     return iou3d


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()
    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None


def new_nms_gpu(boxes, scores, iou_threshold, pre_maxsize=None, score_threshold=0, variance=None, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """


    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    variance = variance.cpu().numpy() if variance is not None else None

    boxes[:, 6] = common_utils.limit_period(
            boxes[:, 6], offset=0.5, period=np.pi*2
        )
    new_scores, new_boxes = nms_func(boxes, scores, iou_threshold, score_threshold,
                                     variance=variance)

    try:
        keep = (new_scores > 0).nonzero()[0]
        keep = keep[new_scores[keep].argsort()[::-1]]
    except:
        import pdb;pdb.set_trace()
    return keep, None, new_boxes


def nms_func(boxes, scores, iou_threshold, score_threshold=0, variance=None):

    undone_mask = scores >= score_threshold

    '''
        scores.shape, undone_mask: [90000]
        boxes.shape: [9000, 7]
    '''
    ious_all = boxes_bev_iou_cpu(boxes, boxes)

    while undone_mask.sum() > 0:
        idx = scores[undone_mask].argmax()

        idx = undone_mask.nonzero()[0][idx]
        top_box = boxes[idx:idx+1]
        _boxes = boxes[undone_mask]

        ious = ious_all[undone_mask, idx]

        if variance is not None:
            _variance = variance[undone_mask, :7]
            ioumask = ious > iou_threshold
            klbox = _boxes[ioumask]

            if top_box[:, 6] > 0:
                klbox[np.abs(klbox[:, 6]-top_box[:, 6])>=np.pi*3/2, 6] += np.pi*2
            else:
                klbox[np.abs(klbox[:, 6]-top_box[:, 6])>=np.pi*3/2, 6] -= np.pi*2

            kliou = ious[ioumask]

            klvar = _variance[ioumask]
            std_iou_sigma = 0.05 #ok
            pi = (np.exp(-1 * (1 - kliou) ** 2 / std_iou_sigma)).reshape(-1, 1)
            pi = pi / klvar

            pi[np.abs(klbox[:, 6]-top_box[:, 6])>=np.pi/4, 6] = 0
            pi = pi / pi.sum(0)
            boxes[idx, :7] = (pi * klbox[:, :7]).sum(0)


        undone_mask[idx] = False
        scores[undone_mask] *= (ious_all[undone_mask, idx] < iou_threshold)
        undone_mask[scores < score_threshold] = False

    # print("nms_func end", time.time() - nms_func_start, 'do_time = ', do_time, 'boxes', boxes.shape)
    return scores, boxes


def nms_normal_gpu(boxes, scores, thresh, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_normal_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None

def softnms_gpu(boxes, scores, iou_threshold, score_threshold=0.1, soft_mode='gaussian', variance=None, soft_sigma=0.3, **kwargs):

    assert soft_mode in ["linear", "gaussian"]
    assert boxes.shape[-1] == 7

    # import pdb;pdb.set_trace()
    new_scores, new_boxes = softnms(boxes, scores, iou_threshold, soft_sigma, score_threshold, soft_mode, variance=variance)

    keep = (new_scores > score_threshold).nonzero(as_tuple=False).view(-1)
    keep = keep[new_scores[keep].argsort(descending=True)]
    return keep, None, new_boxes

def scale_by_iou(ious, soft_sigma, soft_mode="gaussian"):
    if soft_mode == "linear":
        scale = ious.new_ones(ious.size())
        scale[ious >= soft_sigma] = 1 - ious[ious >= soft_sigma]
    else:
        scale = torch.exp(-ious ** 2 / soft_sigma)

    return scale

def softnms(boxes, scores, iou_threshold, soft_sigma, score_threshold, soft_mode="gaussian", variance=None):
    assert soft_mode in ["linear", "gaussian"]

    undone_mask = scores >= score_threshold
    while undone_mask.sum() > 1:
        idx = scores[undone_mask].argmax()
        idx = undone_mask.nonzero(as_tuple=False)[idx].item()
        top_box = boxes[idx:idx+1]
        undone_mask[idx] = False
        _boxes = boxes[undone_mask]

        ious = boxes_iou_bev(_boxes, top_box).flatten()

        if variance is not None:
            _variance = variance[undone_mask, :6]
            ioumask = ious > iou_threshold
            klbox = _boxes[ioumask]
            klbox = torch.cat((klbox, top_box), 0)
            kliou = ious[ioumask]

            # klvar = klbox[:, -4:]
            # klvar = _variance[ioumask]
            klvar = torch.cat((_variance[ioumask], variance[idx:idx+1, :6]), 0)
            std_iou_sigma = 0.05#ok
            pi = torch.exp(-1 * torch.pow((1 - kliou), 2) / std_iou_sigma)
            pi = torch.cat((pi, torch.ones(1).cuda()), 0).unsqueeze(1)
            pi = pi / klvar
            pi = pi / pi.sum(0)
            # print("### original boxes[idx, :7]",  boxes[idx, :7])
            boxes[idx, :6] = (pi * klbox[:, :6]).sum(0)   
            # print("### new boxes[idx, :7]",  boxes[idx, :7])

            # print("### top_box = ", top_box)
            # print("### pi = ", pi)
            # print("### klbox = ", klbox)
            # print("### new top box = ", boxes[idx, :7])

        scales = scale_by_iou(ious, soft_sigma, soft_mode)

        scores[undone_mask] *= scales.flatten()
        undone_mask[scores < score_threshold] = False
    # import pdb;pdb.set_trace()
    # print("### return", scores, boxes)
    return scores, boxes

