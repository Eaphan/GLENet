import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pcdet.utils import box_utils

class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()
        else:
            self.code_weights = None

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)


def compute_fg_mask(gt_boxes2d, shape, downsample_factor=1, device=torch.device("cpu")):
    """
    Compute foreground mask for images
    Args:
        gt_boxes2d: (B, N, 4), 2D box labels
        shape: torch.Size or tuple, Foreground mask desired shape
        downsample_factor: int, Downsample factor for image
        device: torch.device, Foreground mask desired device
    Returns:
        fg_mask (shape), Foreground mask
    """
    fg_mask = torch.zeros(shape, dtype=torch.bool, device=device)

    # Set box corners
    gt_boxes2d /= downsample_factor
    gt_boxes2d[:, :, :2] = torch.floor(gt_boxes2d[:, :, :2])
    gt_boxes2d[:, :, 2:] = torch.ceil(gt_boxes2d[:, :, 2:])
    gt_boxes2d = gt_boxes2d.long()

    # Set all values within each box to True
    B, N = gt_boxes2d.shape[:2]
    for b in range(B):
        for n in range(N):
            u1, v1, u2, v2 = gt_boxes2d[b, n]
            fg_mask[b, v1:v2, u1:u2] = True

    return fg_mask


### Compute the IOU of two rotated 2D rectangle

import math
import numpy as np
import sys
import random
import torch
from torch.autograd import Function
import torch.nn as nn
# from compute_ious import compute_ious_whih_shapely
from scipy.spatial import ConvexHull


## This function is used to determine whether a point is inside a rectangle or not
class compute_vertex(Function):
    '''
    Compute the corners which are inside the rectangles
    '''

    @staticmethod
    def forward(ctx, corners_gboxes, corners_qboxes):

        np_corners_gboxes = corners_gboxes.cpu().numpy()
        np_corners_qboxes = corners_qboxes.cpu().detach().numpy()
        N = corners_gboxes.shape[0]
        num_of_intersections = np.zeros((N,), dtype=np.int32)
        intersections = np.zeros((N, 16), dtype=np.float32)
        flags_qboxes = np.zeros((N, 4), dtype=np.float32)
        flags_gboxes = np.zeros((N, 4), dtype=np.float32)
        flags_inters = np.zeros((N, 4, 4), dtype=np.float32)

        for iter in range(N):
            # step 1: determine how many corners from corners_gboxes inside the np_qboxes
            ab0 = np_corners_qboxes[iter, 2] - np_corners_qboxes[iter, 0]
            ab1 = np_corners_qboxes[iter, 3] - np_corners_qboxes[iter, 1]
            ad0 = np_corners_qboxes[iter, 6] - np_corners_qboxes[iter, 0]
            ad1 = np_corners_qboxes[iter, 7] - np_corners_qboxes[iter, 1]
            # print(f"# {ab0} {ab1} {ad0} {ad1}")
            for i in range(4):
                ap0 = np_corners_gboxes[iter, i * 2] - np_corners_qboxes[iter, 0]
                ap1 = np_corners_gboxes[iter, i * 2 + 1] - np_corners_qboxes[iter, 1]
                abab = ab0 * ab0 + ab1 * ab1
                abap = ab0 * ap0 + ab1 * ap1
                adad = ad0 * ad0 + ad1 * ad1
                adap = ad0 * ap0 + ad1 * ap1

                # debug
                # if abab<0.01 or adad<0.01:
                # print(f"# abab adad abnormal {ap0} {ap1} {abab} {abap} {adad} {adap}")
                # print(f"np_corners_qboxes iter = {np_corners_qboxes[iter]}")
                # print(f"np_corners_gboxes iter = {np_corners_gboxes[iter]}")

                # import pdb;pdb.set_trace()
                if (abab >= abap and abap >= 0 and adad >= adap and adap >= 0 and adad>0 and abab>0):
                    intersections[iter, num_of_intersections[iter] * 2] = np_corners_gboxes[iter, i * 2]
                    intersections[iter, num_of_intersections[iter] * 2 + 1] = np_corners_gboxes[iter, i * 2 + 1]
                    num_of_intersections[iter] += 1
                    flags_gboxes[iter, i] = 1.0

            # step 2: determine how many corners from np_qboxes inside corners_gboxes
            ab0 = np_corners_gboxes[iter, 2] - np_corners_gboxes[iter, 0]
            ab1 = np_corners_gboxes[iter, 3] - np_corners_gboxes[iter, 1]
            ad0 = np_corners_gboxes[iter, 6] - np_corners_gboxes[iter, 0]
            ad1 = np_corners_gboxes[iter, 7] - np_corners_gboxes[iter, 1]
            for i in range(4):
                ap0 = np_corners_qboxes[iter, i * 2] - np_corners_gboxes[iter, 0]
                ap1 = np_corners_qboxes[iter, i * 2 + 1] - np_corners_gboxes[iter, 1]
                abab = ab0 * ab0 + ab1 * ab1
                abap = ab0 * ap0 + ab1 * ap1
                adad = ad0 * ad0 + ad1 * ad1
                adap = ad0 * ap0 + ad1 * ap1
                if (abab >= abap and abap >= 0 and adad >= adap and adap >= 0):
                    intersections[iter, num_of_intersections[iter] * 2] = np_corners_qboxes[iter, i * 2]
                    intersections[iter, num_of_intersections[iter] * 2 + 1] = np_corners_qboxes[iter, i * 2 + 1]
                    num_of_intersections[iter] += 1
                    flags_qboxes[iter, i] = 1.0

            # step 3: find the intersection of all the edges
            for i in range(4):
                for j in range(4):
                    A = np.zeros((2,), dtype=np.float32)
                    B = np.zeros((2,), dtype=np.float32)
                    C = np.zeros((2,), dtype=np.float32)
                    D = np.zeros((2,), dtype=np.float32)

                    A[0] = np_corners_gboxes[iter, 2 * i]
                    A[1] = np_corners_gboxes[iter, 2 * i + 1]
                    B[0] = np_corners_gboxes[iter, 2 * ((i + 1) % 4)]
                    B[1] = np_corners_gboxes[iter, 2 * ((i + 1) % 4) + 1]

                    C[0] = np_corners_qboxes[iter, 2 * j]
                    C[1] = np_corners_qboxes[iter, 2 * j + 1]
                    D[0] = np_corners_qboxes[iter, 2 * ((j + 1) % 4)]
                    D[1] = np_corners_qboxes[iter, 2 * ((j + 1) % 4) + 1]

                    BA0 = B[0] - A[0]
                    BA1 = B[1] - A[1]
                    CA0 = C[0] - A[0]
                    CA1 = C[1] - A[1]
                    DA0 = D[0] - A[0]
                    DA1 = D[1] - A[1]

                    acd = DA1 * CA0 > CA1 * DA0
                    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
                    if acd != bcd:
                        abc = CA1 * BA0 > BA1 * CA0
                        abd = DA1 * BA0 > BA1 * DA0
                        if abc != abd:
                            DC0 = D[0] - C[0]
                            DC1 = D[1] - C[1]
                            ABBA = A[0] * B[1] - B[0] * A[1]
                            CDDC = C[0] * D[1] - D[0] * C[1]
                            DH = BA1 * DC0 - BA0 * DC1
                            Dx = ABBA * DC0 - BA0 * CDDC
                            Dy = ABBA * DC1 - BA1 * CDDC
                            # DH = (B[1] - A[1]) * (D[0] - C[0]) - (B[0] - A[0]) * (D[1] - C[1])
                            # Dx = (A[0] * B[1] - B[0] * A[1]) * (D[0] - C[0]) - (B[0] - A[0]) * (C[0] * D[1] - D[0] * C[1])
                            # Dy = (A[0] * B[1] - B[0] * A[1]) * (D[1] - C[1]) - (B[1] - A[1]) * (C[0] * D[1] - D[0] * C[1])
                            if (num_of_intersections[iter] > 7):
                                print("iter = ", iter)
                                print("(%.4f %.4f) (%.4f %.4f) (%.4f %.4f) (%.4f %.4f)" % (
                                    np_corners_gboxes[iter, 0], np_corners_gboxes[iter, 1],
                                    np_corners_gboxes[iter, 2], np_corners_gboxes[iter, 3],
                                    np_corners_gboxes[iter, 4], np_corners_gboxes[iter, 5],
                                    np_corners_gboxes[iter, 6], np_corners_gboxes[iter, 7]))
                                print("(%.4f %.4f) (%.4f %.4f) (%.4f %.4f) (%.4f %.4f)" % (
                                    np_corners_qboxes[iter, 0], np_corners_qboxes[iter, 1],
                                    np_corners_qboxes[iter, 2], np_corners_qboxes[iter, 3],
                                    np_corners_qboxes[iter, 4], np_corners_qboxes[iter, 5],
                                    np_corners_qboxes[iter, 6], np_corners_qboxes[iter, 7]))
                                continue
                            intersections[iter, num_of_intersections[iter] * 2] = Dx / DH
                            intersections[iter, num_of_intersections[iter] * 2 + 1] = Dy / DH
                            num_of_intersections[iter] += 1
                            flags_inters[iter, i, j] = 1.0

        ctx.save_for_backward(corners_qboxes)
        ctx.corners_gboxes = corners_gboxes
        ctx.flags_qboxes = flags_qboxes
        ctx.flags_gboxes = flags_gboxes
        ctx.flags_inters = flags_inters
        # conver numpy to tensor
        tensor_intersections = torch.from_numpy(intersections)
        tensor_num_of_intersections = torch.from_numpy(num_of_intersections)
        return tensor_intersections, tensor_num_of_intersections.detach()

    @staticmethod
    def backward(ctx, *grad_outputs):
        _variables = ctx.saved_tensors
        corners_qboxes = _variables[0]
        corners_gboxes = ctx.corners_gboxes
        flags_qboxes = ctx.flags_qboxes
        flags_gboxes = ctx.flags_gboxes
        flags_inters = ctx.flags_inters
        grad_output = grad_outputs[0]

        np_corners_gboxes = corners_gboxes.cpu().numpy()
        np_corners_qboxes = corners_qboxes.cpu().detach().numpy()

        N = flags_qboxes.shape[0]
        n_of_inter = np.zeros((N,), dtype=np.int32)

        ### Check whether here is correct or not
        Jacbian_qboxes = np.zeros((N, 8, 16), dtype=np.float32)
        Jacbian_gboxes = np.zeros((N, 8, 16), dtype=np.float32)

        for iter in range(N):

            for i in range(4):
                if (flags_gboxes[iter, i] > 0):
                    Jacbian_gboxes[iter, i * 2, n_of_inter[iter] * 2] += 1.0
                    Jacbian_gboxes[iter, i * 2 + 1, n_of_inter[iter] * 2 + 1] += 1.0
                    n_of_inter[iter] += 1

            for i in range(4):
                if (flags_qboxes[iter, i] > 0):
                    Jacbian_qboxes[iter, i * 2, n_of_inter[iter] * 2] += 1.0
                    Jacbian_qboxes[iter, i * 2 + 1, n_of_inter[iter] * 2 + 1] += 1.0
                    n_of_inter[iter] += 1

            for i in range(4):
                for j in range(4):
                    if (flags_inters[iter, i, j] > 0):
                        ###
                        A = np.zeros((2,), dtype=np.float32)
                        B = np.zeros((2,), dtype=np.float32)
                        C = np.zeros((2,), dtype=np.float32)
                        D = np.zeros((2,), dtype=np.float32)
                        A[0] = np_corners_gboxes[iter, 2 * i]
                        A[1] = np_corners_gboxes[iter, 2 * i + 1]

                        B[0] = np_corners_gboxes[iter, 2 * ((i + 1) % 4)]
                        B[1] = np_corners_gboxes[iter, 2 * ((i + 1) % 4) + 1]

                        C[0] = np_corners_qboxes[iter, 2 * j]
                        C[1] = np_corners_qboxes[iter, 2 * j + 1]

                        D[0] = np_corners_qboxes[iter, 2 * ((j + 1) % 4)]
                        D[1] = np_corners_qboxes[iter, 2 * ((j + 1) % 4) + 1]
                        BA0 = B[0] - A[0]
                        BA1 = B[1] - A[1]
                        CA0 = C[0] - A[0]
                        CA1 = C[1] - A[1]
                        DA0 = D[0] - A[0]
                        DA1 = D[1] - A[1]
                        acd = DA1 * CA0 > CA1 * DA0
                        bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])

                        if acd != bcd:
                            abc = CA1 * BA0 > BA1 * CA0
                            abd = DA1 * BA0 > BA1 * DA0
                            if abc != abd:
                                DC0 = D[0] - C[0]
                                DC1 = D[1] - C[1]
                                ABBA = A[0] * B[1] - B[0] * A[1]
                                CDDC = C[0] * D[1] - D[0] * C[1]
                                DH = BA1 * DC0 - BA0 * DC1
                                Dx = ABBA * DC0 - BA0 * CDDC
                                Dy = ABBA * DC1 - BA1 * CDDC

                                # DH = (B[1] - A[1]) * (D[0] - C[0]) - (B[0] - A[0]) * (D[1] - C[1])
                                # Dx = (A[0] * B[1] - B[0] * A[1]) * (D[0] - C[0]) - (B[0] - A[0]) * (C[0] * D[1] - D[0] * C[1])
                                det_DxA0 = B[1] * (D[0] - C[0]) + (C[0] * D[1] - D[0] * C[1])
                                det_DxA1 = - B[0] * (D[0] - C[0])
                                det_DxB0 = - A[1] * (D[0] - C[0]) - (C[0] * D[1] - D[0] * C[1])
                                det_DxB1 = A[0] * (D[0] - C[0])
                                det_DxC0 = - (A[0] * B[1] - B[0] * A[1]) - (B[0] - A[0]) * D[1]
                                det_DxC1 = (B[0] - A[0]) * D[0]
                                det_DxD0 = (A[0] * B[1] - B[0] * A[1]) + (B[0] - A[0]) * C[1]
                                det_DxD1 = -(B[0] - A[0]) * C[0]
                                # Dy = (A[0] * B[1] - B[0] * A[1]) * (D[1] - C[1]) - (B[1] - A[1]) * (C[0] * D[1] - D[0] * C[1])
                                det_DyA0 = B[1] * (D[1] - C[1])
                                det_DyA1 = - B[0] * (D[1] - C[1]) + (C[0] * D[1] - D[0] * C[1])
                                det_DyB0 = -  A[1] * (D[1] - C[1])
                                det_DyB1 = A[0] * (D[1] - C[1]) - (C[0] * D[1] - D[0] * C[1])

                                det_DyC0 = - (B[1] - A[1]) * D[1]
                                det_DyC1 = - (A[0] * B[1] - B[0] * A[1]) + (B[1] - A[1]) * D[0]
                                det_DyD0 = (B[1] - A[1]) * C[1]
                                det_DyD1 = (A[0] * B[1] - B[0] * A[1]) - (B[1] - A[1]) * C[0]
                                # DH = (B[1] - A[1]) * (D[0] - C[0]) - (B[0] - A[0]) * (D[1] - C[1])
                                det_DHA0 = (D[1] - C[1])
                                det_DHA1 = - (D[0] - C[0])
                                det_DHB0 = - (D[1] - C[1])
                                det_DHB1 = (D[0] - C[0])
                                det_DHC0 = - (B[1] - A[1])
                                det_DHC1 = (B[0] - A[0])
                                det_DHD0 = (B[1] - A[1])
                                det_DHD1 = - (B[0] - A[0])

                                DHDH = DH * DH
                                Jacbian_gboxes[iter, i * 2, n_of_inter[iter] * 2] += (det_DxA0 * DH - Dx * det_DHA0) / DHDH
                                Jacbian_gboxes[iter, i * 2, n_of_inter[iter] * 2 + 1] += (det_DyA0 * DH - Dy * det_DHA0) / DHDH

                                Jacbian_gboxes[iter, i * 2 + 1, n_of_inter[iter] * 2] += (det_DxA1 * DH - Dx * det_DHA1) / DHDH
                                Jacbian_gboxes[iter, i * 2 + 1, n_of_inter[iter] * 2 + 1] += (det_DyA1 * DH - Dy * det_DHA1) / DHDH

                                Jacbian_gboxes[iter, 2 * ((i + 1) % 4), n_of_inter[iter] * 2] += (det_DxB0 * DH - Dx * det_DHB0) / DHDH
                                Jacbian_gboxes[iter, 2 * ((i + 1) % 4), n_of_inter[iter] * 2 + 1] += (det_DyB0 * DH - Dy * det_DHB0) / DHDH

                                Jacbian_gboxes[iter, 2 * ((i + 1) % 4) + 1, n_of_inter[iter] * 2] += (det_DxB1 * DH - Dx * det_DHB1) / DHDH
                                Jacbian_gboxes[iter, 2 * ((i + 1) % 4) + 1, n_of_inter[iter] * 2 + 1] += (det_DyB1 * DH - Dy * det_DHB1) / DHDH

                                Jacbian_qboxes[iter, j * 2, n_of_inter[iter] * 2] += (det_DxC0 * DH - Dx * det_DHC0) / DHDH
                                Jacbian_qboxes[iter, j * 2, n_of_inter[iter] * 2 + 1] += (det_DyC0 * DH - Dy * det_DHC0) / DHDH

                                Jacbian_qboxes[iter, j * 2 + 1, n_of_inter[iter] * 2] += (det_DxC1 * DH - Dx * det_DHC1) / DHDH
                                Jacbian_qboxes[iter, j * 2 + 1, n_of_inter[iter] * 2 + 1] += (det_DyC1 * DH - Dy * det_DHC1) / DHDH

                                Jacbian_qboxes[iter, 2 * ((j + 1) % 4), n_of_inter[iter] * 2] += (det_DxD0 * DH - Dx * det_DHD0) / DHDH
                                Jacbian_qboxes[iter, 2 * ((j + 1) % 4), n_of_inter[iter] * 2 + 1] += (det_DyD0 * DH - Dy * det_DHD0) / DHDH

                                Jacbian_qboxes[iter, 2 * ((j + 1) % 4) + 1, n_of_inter[iter] * 2] += (det_DxD1 * DH - Dx * det_DHD1) / DHDH
                                Jacbian_qboxes[iter, 2 * ((j + 1) % 4) + 1, n_of_inter[iter] * 2 + 1] += (det_DyD1 * DH - Dy * det_DHD1) / DHDH

                                n_of_inter[iter] += 1

        tensor_Jacbian_gboxes = torch.from_numpy(Jacbian_gboxes).to(torch.device(corners_qboxes.device))
        tensor_Jacbian_qboxes = torch.from_numpy(Jacbian_qboxes).to(torch.device(corners_qboxes.device))
        grad_output_cuda = grad_output.to(torch.device(corners_qboxes.device))
        # print("grad_output_cuda =", grad_output_cuda.shape)
        tensor_grad_corners_gboxes = tensor_Jacbian_gboxes.matmul(grad_output_cuda.unsqueeze(2)).squeeze(2)
        tensor_grad_corners_qboxes = tensor_Jacbian_qboxes.matmul(grad_output_cuda.unsqueeze(2)).squeeze(2)
        return tensor_grad_corners_gboxes, tensor_grad_corners_qboxes


class sort_vertex(Function):
    @staticmethod
    def forward(ctx, int_pts, num_of_inter):
        np_int_pts = int_pts.detach().numpy()
        #np_num_of_inter = num_of_inter.detach().numpy()
        np_num_of_inter = num_of_inter
        N = int_pts.shape[0]
        np_sorted_indexs = np.zeros((N, 8), dtype=np.int32)
        sorted_int_pts = np.zeros((N, 16), dtype=np.float32)
        for iter in range(N):
            if np_num_of_inter[iter] > 0:
                center = np.zeros((2,), dtype=np.float32)
                for i in range(np_num_of_inter[iter]):
                    center[0] += np_int_pts[iter, 2 * i]
                    center[1] += np_int_pts[iter, 2 * i + 1]
                center[0] /= np_num_of_inter[iter].float()
                center[1] /= np_num_of_inter[iter].float()

                angle = np.zeros((8,), dtype=np.float32)
                v = np.zeros((2,), dtype=np.float32)

                for i in range(np_num_of_inter[iter]):
                    v[0] = np_int_pts[iter, 2 * i] - center[0]
                    v[1] = np_int_pts[iter, 2 * i + 1] - center[1]
                    d = math.sqrt(v[0] * v[0] + v[1] * v[1])
                    v[0] = v[0] / d
                    v[1] = v[1] / d
                    anglei = math.atan2(v[1], v[0])
                    if anglei < 0:
                        angle[i] = anglei + 2 * 3.1415926
                    else:
                        angle[i] = anglei
                # sort angles with descending
                np_sorted_indexs[iter, :] = np.argsort(-angle)
                for i in range(np_num_of_inter[iter]):
                    sorted_int_pts[iter, 2 * i] = np_int_pts[iter, 2 * np_sorted_indexs[iter, i]]
                    sorted_int_pts[iter, 2 * i + 1] = np_int_pts[iter, 2 * np_sorted_indexs[iter, i] + 1]

        # conver numpy to tensor
        ctx.save_for_backward(int_pts, num_of_inter)
        ctx.np_sorted_indexs = np_sorted_indexs
        tensor_sorted_int_pts = torch.from_numpy(sorted_int_pts)
        return tensor_sorted_int_pts

    @staticmethod
    def backward(ctx, grad_output):
        int_pts, num_of_inter = ctx.saved_tensors
        np_sorted_indexs = ctx.np_sorted_indexs

        N = int_pts.shape[0]
        Jacbian_int_pts = np.zeros((N, 16, 16), dtype=np.float32)
        for iter in range(N):
            for i in range(num_of_inter[iter]):
                Jacbian_int_pts[iter, 2 * np_sorted_indexs[iter, i], 2 * i] = 1
                Jacbian_int_pts[iter, 2 * np_sorted_indexs[iter, i] + 1, 2 * i + 1] = 1

        tensor_Jacbian_int_pts = torch.from_numpy(Jacbian_int_pts).to(torch.device(int_pts.device))
        grad_output_cuda = grad_output.to(torch.device(int_pts.device))
        tensor_grad_int_pts = tensor_Jacbian_int_pts.matmul(grad_output_cuda.unsqueeze(2)).squeeze(2)
        # todo: my second addtion
        # my_add_1 = torch.zeros(tensor_grad_int_pts.shape[0], dtype=torch.float32)
        return tensor_grad_int_pts, None


class area_polygon(Function):

    @staticmethod
    def forward(ctx, int_pts, num_of_inter):
        ctx.save_for_backward(int_pts, num_of_inter)
        np_int_pts = int_pts.detach().numpy()
        #np_num_of_inter = num_of_inter.detach().numpy()
        np_num_of_inter = num_of_inter
        N = int_pts.shape[0]
        areas = np.zeros((N,), dtype=np.float32)

        for iter in range(N):
            for i in range(np_num_of_inter[iter] - 2):
                p1 = np_int_pts[iter, 0:2]
                p2 = np_int_pts[iter, 2 * i + 2:2 * i + 4]
                p3 = np_int_pts[iter, 2 * i + 4:2 * i + 6]
                areas[iter] += abs(((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])) / 2.0)

        tensor_areas = torch.from_numpy(areas)

        return tensor_areas

    @staticmethod
    def backward(ctx, *grad_outputs):

        int_pts, num_of_inter = ctx.saved_tensors
        np_int_pts = int_pts.detach().numpy()
        np_num_of_inter = num_of_inter.detach().numpy()
        grad_output0 = grad_outputs[0]
        N = int_pts.shape[0]
        grad_int_pts = np.zeros((N, 16), dtype=np.float32)

        for iter in range(N):
            if (np_num_of_inter[iter] > 2):
                for i in range(np_num_of_inter[iter]):
                    if i == 0:
                        for j in range(np_num_of_inter[iter] - 2):
                            p1 = np_int_pts[iter, 0:2]
                            p2 = np_int_pts[iter, 2 * j + 2:2 * j + 4]
                            p3 = np_int_pts[iter, 2 * j + 4:2 * j + 6]

                            if ((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])) > 0:
                                grad_int_pts[iter, 0] += (p2[1] - p3[1]) * grad_output0[iter] * 0.5
                                grad_int_pts[iter, 1] += -(p2[0] - p3[0]) * grad_output0[iter] * 0.5
                            else:
                                grad_int_pts[iter, 0] += -(p2[1] - p3[1]) * grad_output0[iter] * 0.5
                                grad_int_pts[iter, 1] += (p2[0] - p3[0]) * grad_output0[iter] * 0.5

                    elif i == 1:
                        p1 = np_int_pts[iter, 0:2]
                        p2 = np_int_pts[iter, 2:4]
                        p3 = np_int_pts[iter, 4:6]
                        if ((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])) > 0:
                            grad_int_pts[iter, 2] = -(p1[1] - p3[1]) * grad_output0[iter] * 0.5
                            grad_int_pts[iter, 3] = (p1[0] - p3[0]) * grad_output0[iter] * 0.5
                        else:
                            grad_int_pts[iter, 2] = (p1[1] - p3[1]) * grad_output0[iter] * 0.5
                            grad_int_pts[iter, 3] = -(p1[0] - p3[0]) * grad_output0[iter] * 0.5

                    elif i == np_num_of_inter[iter] - 1:

                        p1 = np_int_pts[iter, 2 * (np_num_of_inter[iter] - 2):2 * (np_num_of_inter[iter] - 1)]
                        p2 = np_int_pts[iter, 2 * (np_num_of_inter[iter] - 1):2 * (np_num_of_inter[iter])]
                        p3 = np_int_pts[iter, 0:2]
                        if ((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])) > 0:
                            grad_int_pts[iter, 2 * (np_num_of_inter[iter] - 1)] = - (p1[1] - p3[1]) * grad_output0[
                                iter] * 0.5
                            grad_int_pts[iter, 2 * np_num_of_inter[iter] - 1] = (p1[0] - p3[0]) * grad_output0[
                                iter] * 0.5
                        else:
                            grad_int_pts[iter, 2 * (np_num_of_inter[iter] - 1)] = (p1[1] - p3[1]) * grad_output0[
                                iter] * 0.5
                            grad_int_pts[iter, 2 * np_num_of_inter[iter] - 1] = - (p1[0] - p3[0]) * grad_output0[
                                iter] * 0.5
                    else:
                        p1 = np_int_pts[iter, 0:2]
                        p2 = np_int_pts[iter, 2 * i - 2: 2 * i]
                        p3 = np_int_pts[iter, 2 * i: 2 * i + 2]
                        if ((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])) > 0:
                            grad_int_pts[iter, i * 2] += (- (p2[1] - p3[1]) + (p1[1] - p3[1])) * grad_output0[
                                iter] * 0.5
                            grad_int_pts[iter, i * 2 + 1] += (- (p1[0] - p3[0]) + (p2[0] - p3[0])) * grad_output0[
                                iter] * 0.5
                        else:
                            grad_int_pts[iter, i * 2] += ((p2[1] - p3[1]) - (p1[1] - p3[1])) * grad_output0[iter] * 0.5
                            grad_int_pts[iter, i * 2 + 1] += ((p1[0] - p3[0]) - (p2[0] - p3[0])) * grad_output0[
                                iter] * 0.5

                        p1 = np_int_pts[iter, 0:2]
                        p2 = np_int_pts[iter, 2 * i: 2 * i + 2]
                        p3 = np_int_pts[iter, 2 * i + 2: 2 * i + 4]
                        if ((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])) > 0:
                            grad_int_pts[iter, i * 2] += - (p1[1] - p3[1]) * grad_output0[iter] * 0.5
                            grad_int_pts[iter, i * 2 + 1] += (p1[0] - p3[0]) * grad_output0[iter] * 0.5
                        else:
                            grad_int_pts[iter, i * 2] += (p1[1] - p3[1]) * grad_output0[iter] * 0.5
                            grad_int_pts[iter, i * 2 + 1] += -(p1[0] - p3[0]) * grad_output0[iter] * 0.5

        tensor_grad_int_pts = torch.from_numpy(grad_int_pts)
        # todo: my first addition.
        # my_add_0 = torch.zeros(tensor_grad_int_pts.shape[0], dtype=torch.float32)
        #print("area_polygon backward")
        return tensor_grad_int_pts, None


## Transform the (cx, cy, w, l, theta) representation to 4 corners representation
class rbbox_to_corners(nn.Module):

    def _init_(self, rbbox):
        super(rbbox_to_corners, self)._init_()
        self.rbbox = rbbox
        return

    def forward(ctx, rbbox):
        '''
                    There is no rotation performed here. As axis are aligned.
                                          ^ [y]
                                     1 --------- 2
                                     /          /    --->
                                    0 -------- 3     [x]
                    Each node has the coordinate of [x, y]. Corresponding the order of input.

                    Output: [N, 8]
                            [x_0, y_0, x_1, y_1, x_2, y_2, x_3, y_3],
                            if ry > 0, then rotate clockwisely.

                '''

        assert rbbox.shape[1] == 5
        device = rbbox.device
        corners = torch.zeros((rbbox.shape[0], 8), dtype=torch.float32, device=device)
        dxcos = rbbox[:, 2].mul(torch.cos(rbbox[:, 4])) / 2.0
        dxsin = rbbox[:, 2].mul(torch.sin(rbbox[:, 4])) / 2.0
        dycos = rbbox[:, 3].mul(torch.cos(rbbox[:, 4])) / 2.0
        dysin = rbbox[:, 3].mul(torch.sin(rbbox[:, 4])) / 2.0
        corners[:, 0] = -dxcos - dysin + rbbox[:, 0]
        corners[:, 1] = dxsin - dycos + rbbox[:, 1]
        corners[:, 2] = -dxcos + dysin + rbbox[:, 0]
        corners[:, 3] = dxsin + dycos + rbbox[:, 1]

        corners[:, 4] = dxcos + dysin + rbbox[:, 0]
        corners[:, 5] = -dxsin + dycos + rbbox[:, 1]
        corners[:, 6] = dxcos - dysin + rbbox[:, 0]
        corners[:, 7] = -dxsin - dycos + rbbox[:, 1]
        return corners

class rinter_area_compute(nn.Module):

    def _init_(self, corners_gboxes, corners_qboxes):
        super(rinter_area_compute, self)._init_()
        self.corners_gboxes = corners_gboxes
        self.corners_qboxes = corners_qboxes
        return

    def forward(ctx, corners_gboxes, corners_qboxes):
        intersections, num_of_intersections = compute_vertex(corners_gboxes, corners_qboxes)
        num_of_intersections = num_of_intersections.detach()
        sorted_int_pts = sort_vertex(intersections, num_of_intersections)
        # x = sorted_int_pts.clone()
        # x[0, 4:6] = sorted_int_pts[0, 6:8]
        # x[0, 6:8] = sorted_int_pts[0, 4:6]
        inter_area = area_polygon(sorted_int_pts, num_of_intersections)
        return inter_area


class find_convex_hull(Function):
    # get the minimum bounding box from a set of points, points are reordered with a anti-clockwise order.
    # and those points inside the minimum bbox are removed.
    @staticmethod
    def forward(ctx, corners):
        np_corners = corners.cpu().detach().numpy()
        hull = ConvexHull(np_corners)
        M = hull.nsimplex
        index = hull.vertices
        hull_points_2d = np.zeros((M, 2), np.float32)
        for i in range(M):
            hull_points_2d[i, 0] = np_corners[index[i], 0]
            hull_points_2d[i, 1] = np_corners[index[i], 1]
        tensor_hull_points_2d = torch.from_numpy(hull_points_2d).to(torch.device(corners.device))
        ctx.index = index
        return tensor_hull_points_2d

    @staticmethod
    def backward(ctx, *grad_outputs):
        index = ctx.index
        grad_output0 = grad_outputs[0]
        device = grad_output0.device
        grad_corners = torch.zeros((8, 2), dtype=torch.float32, device=device)
        for i in range(len(index)):
            grad_corners[index[i], 0] = grad_output0[i, 0]
            grad_corners[index[i], 1] = grad_output0[i, 1]
        return grad_corners


## nn Module
class mbr_convex_hull(nn.Module):
    '''
        Miminum Bounding Rectangle (MBR)
        Algorithm core: The orientation of the MBR is the same as the one of one of the edges of the point cloud convex hull, which means
        the result rectangle must overlap with at least one of the edges.
    '''

    def _init_(self, hull_points_2d):
        super(mbr_convex_hull, self)._init_()
        self.hull_points_2d = hull_points_2d
        return

    def forward(ctx, hull_points_2d):
        device = hull_points_2d.device
        N = hull_points_2d.shape[0]
        edges = hull_points_2d[1:N, :].add(- hull_points_2d[0:N - 1, :])
        edge_angles = torch.atan2(edges[:, 1], edges[:, 0])
        edge_angles = torch.fmod(edge_angles, 3.1415926 / 2.0)
        edge_angles = torch.abs(edge_angles)
        # edge_angles = torch.unique(edge_angles)
        # print("edge_angles =", edge_angles)
        a = torch.stack((torch.cos(edge_angles), torch.cos(edge_angles - 3.1415926 / 2.0)), 1)
        a = torch.unsqueeze(a, 1)
        b = torch.stack((torch.cos(edge_angles + 3.1415926 / 2.0), torch.cos(edge_angles)), 1)
        b = torch.unsqueeze(b, 1)
        R_tensor = torch.cat((a, b), 1)
        hull_points_2d_ = torch.unsqueeze(torch.transpose(hull_points_2d, 0, 1), 0)
        rot_points = R_tensor.matmul(hull_points_2d_)
        min_x = torch.min(rot_points, 2)[0]
        max_x = torch.max(rot_points, 2)[0]
        areas = (max_x[:, 0] - min_x[:, 0]).mul(max_x[:, 1] - min_x[:, 1])
        return torch.min(areas)


class mbr_area_compute(nn.Module):
    # get the minimum bounding box from a set of points

    def _init_(self, corners):
        super(mbr_area_compute, self)._init_()
        self.corners = corners
        return

    def forward(ctx, corners):
        # np_corners = corners.numpy()
        N = corners.shape[0]
        # mbr_rect_areas   = torch.zeros((N,), dtype=torch.float32)
        mbr_rect_area = []
        for i in range(N):
            mbr_rect_area.append(torch.zeros((1,), dtype=torch.float32, device=corners.device))
        # mbr_rect_areas   = torch.zeros((N,), dtype=torch.float32, device = corners_gboxes.device)
        for iter in range(N):
            convex_hull_pts = find_convex_hull(corners[iter, :, :].squeeze())
            mbr_convex_hull_object = mbr_convex_hull()
            mbr_rect_area[iter] = mbr_convex_hull_object(convex_hull_pts)
        mbr_rect_areas = torch.stack(mbr_rect_area)  # torch.cat(mbr_rect_area)
        # ctx.save_for_backward(corners)
        return mbr_rect_areas



class mbr_diag_convex_hull(nn.Module):
    '''
        # added by zhengwu
    '''

    def _init_(self, hull_points_2d):
        super(mbr_diag_convex_hull, self)._init_()
        self.hull_points_2d = hull_points_2d
        return

    def forward(ctx, hull_points_2d):
        device = hull_points_2d.device
        N = hull_points_2d.shape[0]
        edges = hull_points_2d[1:N, :].add(- hull_points_2d[0:N - 1, :])
        edge_angles = torch.atan2(edges[:, 1], edges[:, 0])
        edge_angles = torch.fmod(edge_angles, 3.1415926 / 2.0)
        edge_angles = torch.abs(edge_angles)
        # edge_angles = torch.unique(edge_angles)
        # print("edge_angles =", edge_angles)
        a = torch.stack((torch.cos(edge_angles), torch.cos(edge_angles - 3.1415926 / 2.0)), 1)
        a = torch.unsqueeze(a, 1)
        b = torch.stack((torch.cos(edge_angles + 3.1415926 / 2.0), torch.cos(edge_angles)), 1)
        b = torch.unsqueeze(b, 1)
        R_tensor = torch.cat((a, b), 1)
        hull_points_2d_ = torch.unsqueeze(torch.transpose(hull_points_2d, 0, 1), 0)
        rot_points = R_tensor.matmul(hull_points_2d_)
        min_x = torch.min(rot_points, 2)[0]
        max_x = torch.max(rot_points, 2)[0]
        areas = (max_x[:, 0] - min_x[:, 0]).mul(max_x[:, 1] - min_x[:, 1])
        # modified here
        min_index = torch.argmin(areas)
        corner_max, corner_min = max_x[min_index], min_x[min_index]
        diag = torch.sqrt((corner_max[0] - corner_min[0]) ** 2 + (corner_max[1] - corner_min[1]) ** 2)
        return diag


class mbr_diag_compute(nn.Module):
    # added by zhengwu
    def _init_(self, corners):
        super(mbr_diag_compute, self)._init_()
        self.corners = corners
        return

    def forward(ctx, corners):
        N = corners.shape[0]
        mbr_rect_diag = []
        for iter in range(N):
            convex_hull_pts = find_convex_hull(corners[iter, :, :].squeeze())
            mbr_diag_convex_hull_object = mbr_diag_convex_hull()
            mbr_rect_diag.append(mbr_diag_convex_hull_object(convex_hull_pts))
        mbr_rect_diags = torch.stack(mbr_rect_diag)
        return mbr_rect_diags


class _second_box_decode_operation(nn.Module):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """

    # need to convert box_encodings to z-bottom format

    def _init_(self, box_encodings, anchors, encode_angle_to_vector, smooth_dim):
        super(_second_box_decode_operation, self)._init_()
        self.box_encodings = box_encodings
        self.anchors = anchors
        self.encode_angle_to_vector = False
        self.smooth_dim = False
        return

    def forward(ctx, box_encodings, anchors, encode_angle_to_vector, smooth_dim):

        """box decode for VoxelNet in lidar
        Args:
            boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
            anchors ([N, 7] Tensor): anchors
        """
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        if encode_angle_to_vector:
            xt, yt, zt, wt, lt, ht, rtx, rty = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)
        # xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)
        za = za + ha / 2.
        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za
        if smooth_dim:
            lg = (lt + 1) * la
            wg = (wt + 1) * wa
            hg = (ht + 1) * ha
        else:

            lg = torch.exp(lt) * la
            wg = torch.exp(wt) * wa
            hg = torch.exp(ht) * ha
        if encode_angle_to_vector:
            rax = torch.cos(ra)
            ray = torch.sin(ra)
            rgx = rtx + rax
            rgy = rty + ray
            rg = torch.atan2(rgy, rgx)
        else:
            rg = rt + ra
        zg = zg - hg / 2.
        return torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)


###################################
# simplified version
###################################

class rbbox_corners_aligned(nn.Module):

    def _init_(self, gboxes):
        super(rbbox_corners_aligned, self)._init_()
        self.corners_gboxes = gboxes
        return

    def forward(ctx, gboxes):
        '''
            There is no rotation performed here. As axis are aligned.
                                  ^ [y]
                             1 --------- 2
                             /          /    --->
                            0 -------- 3     [x]
            Each node has the coordinate of [x, y]. Corresponding the order of input.
            Output: [N, 2, 4]
                    [[x_0, x_1, x_2, x_3],
                     [y_0, y_1, y_2, y_3]].
        '''
        N = gboxes.shape[0]

        center_x = gboxes[:, 0]
        center_y = gboxes[:, 1]

        x_d = gboxes[:, 2]
        y_d = gboxes[:, 3]

        corners = torch.zeros([N, 2, 4], device=gboxes.device, dtype=torch.float32)

        corners[:, 0, 0] = x_d.mul(-0.5)
        corners[:, 1, 0] = y_d.mul(-0.5)

        corners[:, 0, 1] = x_d.mul(-0.5)
        corners[:, 1, 1] = y_d.mul(0.5)

        corners[:, 0, 2] = x_d.mul(0.5)
        corners[:, 1, 2] = y_d.mul(0.5)

        corners[:, 0, 3] = x_d.mul(0.5)
        corners[:, 1, 3] = y_d.mul(-0.5)

        b = center_x.unsqueeze(1).repeat(1, 4).unsqueeze(1)
        c = center_y.unsqueeze(1).repeat(1, 4).unsqueeze(1)

        return (corners + torch.cat((b, c), 1))


class align_inter_aligned(nn.Module):

    def _init_(self, gboxes, qboxes):
        super(align_inter_aligned, self)._init_()
        self.gboxes = gboxes
        self.qboxes = qboxes
        return

    def forward(ctx, gboxes, qboxes):
        N = gboxes.shape[0]
        M = qboxes.shape[0]
        eps = 1e-5
        assert N == M

        ## we can project the 3D bounding boxes into 3 different plane
        ## Notice: ry is not used here.

        ## view1 xoz plane
        inter_area_xoz = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        mbr_area_xoz = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        rbbox_corners_aligned_object = rbbox_corners_aligned()
        rotated_corners1 = rbbox_corners_aligned_object(gboxes[:, [0, 2, 3, 5, 6]])
        rotated_corners2 = rbbox_corners_aligned_object(qboxes[:, [0, 2, 3, 5, 6]])

        for i in range(N):
            iw = (min(rotated_corners1[i, 0, 2], rotated_corners2[i, 0, 2]) -
                  max(rotated_corners1[i, 0, 1], rotated_corners2[i, 0, 1]) + eps)
            if (iw > 0):
                ih = ((min(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) -
                       max(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps))
                if (ih > 0):
                    inter_area_xoz[i] = iw * ih
            iwmbr = (max(rotated_corners1[i, 0, 3], rotated_corners2[i, 0, 3]) -
                     min(rotated_corners1[i, 0, 0], rotated_corners2[i, 0, 0]) + eps)
            ihmbr = ((max(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) -
                      min(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps))
            mbr_area_xoz[i] = iwmbr * ihmbr

        ### view2 xoy plane
        inter_area_xoy = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        mbr_area_xoy = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        rotated_corners1 = rbbox_corners_aligned_object(gboxes[:, [0, 1, 3, 4, 6]])
        rotated_corners2 = rbbox_corners_aligned_object(qboxes[:, [0, 1, 3, 4, 6]])
        for i in range(N):
            iw = (min(rotated_corners1[i, 0, 2], rotated_corners2[i, 0, 2]) -
                  max(rotated_corners1[i, 0, 1], rotated_corners2[i, 0, 1]) + eps)
            if (iw > 0):
                ih = ((min(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) -
                       max(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps))
                if (ih > 0):
                    inter_area_xoy[i] = iw * ih
            iwmbr = (max(rotated_corners1[i, 0, 3], rotated_corners2[i, 0, 3]) -
                     min(rotated_corners1[i, 0, 0], rotated_corners2[i, 0, 0]) + eps)
            ihmbr = ((max(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) -
                      min(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps))
            mbr_area_xoy[i] = iwmbr * ihmbr

        ### view3 yoz plane
        inter_area_yoz = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        mbr_area_yoz = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        rotated_corners1 = rbbox_corners_aligned_object(gboxes[:, [1, 2, 4, 5, 6]])
        rotated_corners2 = rbbox_corners_aligned_object(qboxes[:, [1, 2, 4, 5, 6]])
        for i in range(N):
            iw = (min(rotated_corners1[i, 0, 2], rotated_corners2[i, 0, 2]) -
                  max(rotated_corners1[i, 0, 1], rotated_corners2[i, 0, 1]) + eps)
            if (iw > 0):
                ih = ((min(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) -
                       max(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps))
                if (ih > 0):
                    inter_area_yoz[i] = iw * ih
            iwmbr = (max(rotated_corners1[i, 0, 3], rotated_corners2[i, 0, 3]) -
                     min(rotated_corners1[i, 0, 0], rotated_corners2[i, 0, 0]) + eps)
            ihmbr = ((max(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) -
                      min(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps))
            mbr_area_yoz[i] = iwmbr * ihmbr

        return inter_area_xoz, mbr_area_xoz, inter_area_xoy, mbr_area_xoy, inter_area_yoz, mbr_area_yoz


class odiou_3D(nn.Module):
    def _init_(self, gboxes, qboxes, aligned=False):
        super(odiou_3D, self)._init_()
        self.gboxes = gboxes
        self.qboxes = qboxes
        self.aligned = aligned
        return

    def forward(ctx, gboxes, qboxes, weights, batch_size):
        '''
            gboxes / qboxes: [N, 7], [x, y, z, w, l, h, ry] in velo coord.
            Notice: (x, y, z) is the real center of bbox.
        '''
        assert gboxes.shape[0] == qboxes.shape[0]
        indicator = torch.gt(gboxes[:, 3], 0) & torch.gt(gboxes[:, 4], 0) & torch.gt(gboxes[:, 5], 0) \
                    & torch.gt(qboxes[:, 3], 0) & torch.gt(qboxes[:, 4], 0) & torch.gt(qboxes[:, 5], 0)
        index_loc = torch.nonzero(indicator)
        # todo: my addtion to avoid too large number after model initialization.
        gboxes = torch.clamp(gboxes, -200.0, 200.0)
        qboxes = torch.clamp(qboxes, -200.0, 200.0)
        odious = torch.zeros([gboxes.shape[0], ], device=gboxes.device, dtype=torch.float32)
        if gboxes.shape[0] == 0 or qboxes.shape[0] == 0:
            return torch.unsqueeze(odious, 1)

        diff_angle = qboxes[:, -1] - gboxes[:, -1]
        angle_factor = 1.25 * (1.0 - torch.abs(torch.cos(diff_angle)))
        rbbox_to_corners_object = rbbox_to_corners()
        corners_gboxes = rbbox_to_corners_object(gboxes[:, [0, 1, 3, 4, 6]])
        corners_qboxes = rbbox_to_corners_object(qboxes[:, [0, 1, 3, 4, 6]])
        corners_gboxes_1 = torch.stack((corners_gboxes[:, [0, 2, 4, 6]], corners_gboxes[:, [1, 3, 5, 7]]), 2)
        corners_qboxes_1 = torch.stack((corners_qboxes[:, [0, 2, 4, 6]], corners_qboxes[:, [1, 3, 5, 7]]), 2)
        corners_pts = torch.cat((corners_gboxes_1, corners_qboxes_1), 1)

        # compute the inter area
        # print("### compute the inter area")
        rinter_area_compute_object = rinter_area_compute()
        inter_area = rinter_area_compute_object(corners_gboxes, corners_qboxes)

        # compute center distance
        # print("### compute center distance")        
        center_dist_square = torch.pow(gboxes[:, 0:3] - qboxes[:, 0:3], 2).sum(-1)

        # compute the mbr bev diag
        # print("### compute the mbr bev diag")        
        mbr_diag_compute_object = mbr_diag_compute()
        # try:
        mbr_diag_bev = mbr_diag_compute_object(corners_pts)
        # finally:
        #     print("### gboxes = ", gboxes)
        #     print("### qboxes = ", qboxes)
        #     print("### corners_pts = ", corners_pts, "######e=", e)
        #     # raise ValueError("######### mbr_diag_compute_object #########")
        inter_h = (torch.min(gboxes[:, 2] + 0.5 * gboxes[:, 5], qboxes[:, 2] + 0.5 * qboxes[:, 5]) -
                   torch.max(gboxes[:, 2] - 0.5 * gboxes[:, 5], qboxes[:, 2] - 0.5 * qboxes[:, 5]))
        oniou_h = (torch.max(gboxes[:, 2] + 0.5 * gboxes[:, 5], qboxes[:, 2] + 0.5 * qboxes[:, 5]) -
                   torch.min(gboxes[:, 2] - 0.5 * gboxes[:, 5], qboxes[:, 2] - 0.5 * qboxes[:, 5]))
        inter_h[inter_h < 0] = 0
        mbr_diag_3d_square = mbr_diag_bev**2 + inter_h ** 2 + 1e-7

        volume_gboxes = gboxes[:, 3].mul(gboxes[:, 4]).mul(gboxes[:, 5])
        volume_qboxes = qboxes[:, 3].mul(qboxes[:, 4]).mul(qboxes[:, 5])
        inter_area_cuda = inter_area.to(torch.device(gboxes.device))
        volume_inc = inter_h.mul(inter_area_cuda)
        volume_union = (volume_gboxes + volume_qboxes - volume_inc)
        center_dist_square_cuda = center_dist_square.to(torch.device(gboxes.device))
        mbr_diag_3d_square_cuda = mbr_diag_3d_square.to(torch.device(gboxes.device))

        ious = torch.div(volume_inc, volume_union)
        dp = torch.div(center_dist_square_cuda[index_loc[:, 0]], mbr_diag_3d_square_cuda[index_loc[:, 0]])
        try:
            odious[index_loc[:, 0]] = 1 - ious[index_loc[:, 0]] + dp + angle_factor[index_loc[:, 0]]
        except:
            print("#####", ious[index_loc[:, 0]].shape, dp.shape, angle_factor.shape)
            print("###", gboxes[:, 3:6], qboxes[:, 3:6])
            import pdb;pdb.set_trace()
            raise ValueError
        batch_ious = odious * weights
        ious_loss = 2.0 * batch_ious.sum() / batch_size
        # if not torch.isfinite(ious_loss):
        return ious_loss


compute_vertex = compute_vertex.apply
sort_vertex = sort_vertex.apply
area_polygon = area_polygon.apply
find_convex_hull = find_convex_hull.apply

if __name__ == '__main__':
    # from det3d.core.iou3d import iou3d_utils
    odious_3d_loss = odiou_3D()
    x = torch.tensor([ 20.8845, -16.0514,  -0.5310,   1.8061,   4.6556,   1.8546,   0.2290], dtype=torch.float32).view(1, 7)
    y = torch.tensor([ 20.8869, -15.9686,  -0.5253,   1.7909,   4.6727,   1.7605,   0.2375], dtype=torch.float32).view(1, 7)
    print(x.shape, y.shape)
    print(odious_3d_loss(x, y, 1, 1))

    # print(iou3d_utils.boxes_aligned_iou3d_gpu(x.cuda(), y.cuda(), need_bev=True))

def neg_loss_cornernet(pred, gt, mask=None):
    """
    Refer to https://github.com/tianweiy/CenterPoint.
    Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
    Args:
        pred: (batch x c x h x w)
        gt: (batch x c x h x w)
        mask: (batch x h x w)
    Returns:
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    if mask is not None:
        mask = mask[:, None, :, :].float()
        pos_loss = pos_loss * mask
        neg_loss = neg_loss * mask
        num_pos = (pos_inds.float() * mask).sum()
    else:
        num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """
    def __init__(self):
        super(FocalLossCenterNet, self).__init__()
        self.neg_loss = neg_loss_cornernet

    def forward(self, out, target, mask=None):
        return self.neg_loss(out, target, mask=mask)


def _reg_loss(regr, gt_regr, mask):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    L1 regression loss
    Args:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    Returns:
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()
    isnotnan = (~ torch.isnan(gt_regr)).float()
    mask *= isnotnan
    regr = regr * mask
    gt_regr = gt_regr * mask

    loss = torch.abs(regr - gt_regr)
    loss = loss.transpose(2, 0)

    loss = torch.sum(loss, dim=2)
    loss = torch.sum(loss, dim=1)
    # else:
    #  # D x M x B
    #  loss = loss.reshape(loss.shape[0], -1)

    # loss = loss / (num + 1e-4)
    loss = loss / torch.clamp_min(num, min=1.0)
    # import pdb; pdb.set_trace()
    return loss


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class RegLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super(RegLossCenterNet, self).__init__()

    def forward(self, output, mask, ind=None, target=None):
        """
        Args:
            output: (batch x dim x h x w) or (batch x max_objects)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim)
        Returns:
        """
        if ind is None:
            pred = output
        else:
            pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss
