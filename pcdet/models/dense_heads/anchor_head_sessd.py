import numpy as np
import torch.nn as nn
import torch

from .anchor_head_template import AnchorHeadTemplate
from .anchor_head_single import AnchorHeadSingle
from pcdet.utils.loss_utils import odiou_3D


class WeightedSmoothL1Loss(nn.Module):
    """Smooth L1 localization loss function.

    The smooth L1_loss is defined elementwise as 0.5*x^2 if |x|<1 and |x|-0.5
    otherwise, where x is the difference between predictions and target.

    See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
    """

    def __init__(self, sigma=3.0, reduction="mean", code_weights=None, codewise=True, loss_weight=1.0,):
        super(WeightedSmoothL1Loss, self).__init__()

        # if code_weights is not None:
        #     self._code_weights = torch.tensor(code_weights, dtype=torch.float32)
        # else:
        #     self._code_weights = None

        self._sigma = sigma               # 3
        self._code_weights = None
        self._codewise = codewise         # True
        self._reduction = reduction       # mean
        self._loss_weight = loss_weight   # 2.0 here

    def forward(self, prediction_tensor, target_tensor, weights=None):
        """Compute loss function.
            Args:
            prediction_tensor: A float tensor of shape [batch_size, num_anchors, code_size] representing the (encoded) predicted locations of objects.
            target_tensor: A float tensor of shape [batch_size, num_anchors, code_size] representing the regression targets
            weights: a float tensor of shape [batch_size, num_anchors]

            Returns:
            loss: a float tensor of shape [batch_size, num_anchors] tensor representing the value of the loss function.
        """
        diff = prediction_tensor - target_tensor
        if self._code_weights is not None:   # False: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], xyzhwlr are equal
            diff = self._code_weights.view(1, 1, -1).to(diff.device) * diff

        # this sml1: 0.5*(3x)^2 if |x|<1/3^2 otherwise |x|-0.5/3^2
        abs_diff = torch.abs(diff)
        abs_diff_lt_1 = torch.le(abs_diff, 1 / (self._sigma ** 2)).type_as(abs_diff)   # compare elements in abs_diff with 1/9, less -> 1.0, otherwise -> 0.0

        # todo???: why 1/9
        # if abs_diff_lt_1 = 1 (abs_diff < 1/9), loss = 0.5 * 9 * (abs_diff)^2, when abs_diff=1/9, loss=0.5/9
        # else if abs_diff_lt=0, (abs_diff > 1/9), loss = abs_diff - (0.5/9), when abs_diff=1/9, loss=0.5/9
        loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * self._sigma, 2) + (abs_diff - 0.5 / (self._sigma ** 2)) * (1.0 - abs_diff_lt_1)

        if self._codewise:    # True
            anchorwise_smooth_l1norm = loss
            if weights is not None:
                anchorwise_smooth_l1norm *= weights.unsqueeze(-1) # pos_anchors multiply the weight: 1/num_pos_anchor in each sample
        else:
            anchorwise_smooth_l1norm = torch.sum(loss, 2)  #  * weights
            if weights is not None:
                anchorwise_smooth_l1norm *= weights

        return anchorwise_smooth_l1norm


class AnchorHeadSessd(AnchorHeadSingle):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        AnchorHeadTemplate.__init__(
            self, model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )

        # self.conv_iou = nn.Conv2d(
        #     input_channels, self.num_anchors_per_location * 1,
        #     kernel_size=1
        # )

        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

        self.odiou_3d_loss = odiou_3D()
        # self.loss_iou_pred = WeightedSmoothL1Loss(sigma=3.0, code_weights=None, codewise=True, loss_weight=1.0)


    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        # nn.init.constant_(self.conv_iou.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
        if self.conv_dir_cls is not None:
            nn.init.constant_(self.conv_dir_cls.bias, -np.log((1 - pi) / pi))


    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        # iou_preds = self.conv_iou(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        # iou_preds = iou_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        # self.forward_ret_dict['iou_preds'] = iou_preds

        # self.forward_ret_dict['gt_boxes'] = data_dict['gt_boxes']

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        # iou_preds = (iou_preds.squeeze() + 1) * 0.5
        # rec_confs = cls_preds * torch.pow(iou_preds, 4)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

    # def get_loss(self):
    #     cls_loss, tb_dict = self.get_cls_layer_loss()
    #     iou_pred_loss, tb_dict_iou_pred = self.get_iou_layer_loss()
    #     tb_dict.update(tb_dict_iou_pred)
    #     box_loss, tb_dict_box = self.get_box_reg_layer_loss()
    #     tb_dict.update(tb_dict_box)
    #     rpn_loss = cls_loss + box_loss + iou_pred_loss

    #     tb_dict['rpn_loss'] = rpn_loss.item()
    #     return rpn_loss, tb_dict

    def get_box_reg_layer_loss(self):
        # diou + dir
        box_preds = self.forward_ret_dict['box_preds']
        # iou_preds = self.forward_ret_dict['iou_preds']

        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        reg_weights = reg_weights[positives]

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)

        batch_box_preds = box_preds.view(batch_size, num_anchors, -1)
        pred_boxes = self.box_coder.decode_torch(batch_box_preds, anchors)[positives].view(-1, 7)
        gt_boxes = self.box_coder.decode_torch(box_reg_targets, anchors)[positives].view(-1, 7)
        
        # import pdb;pdb.set_trace()
        # print("### mark1")
        try:
            loc_loss = self.odiou_3d_loss(gt_boxes, pred_boxes, reg_weights, batch_size)
        except:
            # print("###box_cls_labels=", box_cls_labels)
            print("###box_reg_targets[positives]=", box_reg_targets[positives].view(-1, 7))
            print("###batch_box_preds[positives]=", batch_box_preds[positives].view(-1, 7))
            print("###gt_boxes=", gt_boxes)
            print("###pred_boxes=", pred_boxes)
            print("#################################################################################")


        # box_preds = box_preds.view(batch_size, -1,
        #                            box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
        #                            box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        # box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        # loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        # loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        print(f"### loc_loss = {loc_loss}")
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        # print("### compute dir loss")
        # dir
        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

    
        # iou_pos_targets = iou3d_utils.boxes_aligned_iou3d_gpu(pred_boxes, gt_boxes).detach()
        # iou_pos_targets = 2 * iou_pos_targets - 1
        # batch_iou_preds = iou_preds.view(batch_size, num_anchors, -1)[positives]

        # iou_pred_loss = self.loss_iou_pred(batch_iou_preds, iou_pos_targets, reg_weights)
        # iou_pred_loss = iou_pred_loss.sum() / batch_size
        # box_loss += iou_pred_loss
        # tb_dict['rpn_loss_iou_pred'] = iou_pred_loss.item()
        return box_loss, tb_dict
