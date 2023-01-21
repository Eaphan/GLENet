import numpy as np
import torch.nn as nn
import torch

from .anchor_head_single import AnchorHeadSingle
from ...utils import box_coder_utils, common_utils, loss_utils
from pcdet.ops.iou3d.iou3d_utils import boxes_aligned_iou3d_gpu


class AnchorHeadIoU(AnchorHeadSingle):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, input_channels=input_channels, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.conv_iou = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.add_module(
            'iou_loss_func',
            loss_utils.WeightedSmoothL1Loss()
        )

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        iou_preds = self.conv_iou(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        iou_preds = iou_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['iou_preds'] = iou_preds

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

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_iou_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, iou_preds=iou_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            batch_cls_preds = torch.sigmoid(batch_cls_preds)

             # ad hoc
            # batch_cls_preds[batch_cls_preds < 0.1] = -1
            batch_cls_preds[batch_cls_preds < self.model_cfg.PRE_CLS_THRESH] = 0

            batch_iou_preds = (batch_iou_preds + 1) * 0.5 
            batch_iou_preds[batch_iou_preds < self.model_cfg.PRE_IOU_THRESH] = 0
            batch_cls_preds = batch_cls_preds * torch.pow(batch_iou_preds, self.model_cfg.POW)


            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds

            data_dict['cls_preds_normalized'] = True

        return data_dict

    def wrong_get_box_iou_layer_loss(self):
        iou_preds = self.forward_ret_dict['iou_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']

        # print("### positive samples = {}, negative samples = {}".format((box_cls_labels>0).sum(), (box_cls_labels==0).sum()))

        batch_size = int(iou_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        iou_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        iou_weights /= torch.clamp(pos_normalizer, min=1.0)

        box_iou_targets = self.forward_ret_dict['box_iou_targets']

        # details
        box_iou_targets = 2 * box_iou_targets - 1

        iou_preds = iou_preds.view(batch_size, -1, 1)
        box_iou_targets = box_iou_targets.view(batch_size, -1, 1)
        
        # pos_pred_mask = iou_weights > 0
        # iou_preds = iou_preds[pos_pred_mask].view()

        iou_pred_loss = self.iou_loss_func(iou_preds, box_iou_targets, weights=iou_weights)
        iou_pred_loss = iou_pred_loss.sum() / batch_size

        tb_dict = {
            'rpn_loss_iou': iou_pred_loss.item()
        }
        return iou_pred_loss, tb_dict

    

    def generate_predicted_boxes(self, batch_size, cls_preds, iou_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_iou_preds = iou_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(iou_preds, list) else iou_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_iou_preds, batch_box_preds

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        iou_loss, tb_dict_iou = self.get_box_iou_layer_loss()
        tb_dict.update(tb_dict_iou)
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)

        rpn_loss = cls_loss + box_loss + iou_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def get_box_iou_layer_loss(self):
        iou_preds = self.forward_ret_dict['iou_preds']
        # cls_preds = self.forward_ret_dict['cls_preds']
        box_preds = self.forward_ret_dict['box_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(iou_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        pos_pred_mask = reg_weights > 0
        iou_weights = reg_weights[pos_pred_mask]

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        # batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
        #     if not isinstance(cls_preds, list) else cls_preds
        batch_iou_preds = iou_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(iou_preds, list) else iou_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        batch_box_gts = self.box_coder.decode_torch(box_reg_targets, batch_anchors)
        iou_pos_targets = boxes_aligned_iou3d_gpu(batch_box_preds[pos_pred_mask], batch_box_gts[pos_pred_mask]).detach()
        iou_pos_targets = 2 * iou_pos_targets - 1

        iou_pred_loss = self.iou_loss_func(batch_iou_preds[pos_pred_mask].unsqueeze(0), iou_pos_targets.unsqueeze(0),
                                           weights=iou_weights.unsqueeze(0))
        iou_pred_loss = iou_pred_loss.sum() / batch_size

        tb_dict = {
            'rpn_loss_iou': iou_pred_loss.item()
        }
        return iou_pred_loss, tb_dict



