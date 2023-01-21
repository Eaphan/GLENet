import numpy as np
import torch.nn as nn
import torch

from .anchor_head_template import AnchorHeadTemplate
from ...utils import box_coder_utils, common_utils, loss_utils
from pcdet.ops.iou3d.iou3d_utils import boxes_aligned_iou3d_gpu
from .anchor_head_iou import AnchorHeadIoU

class AnchorHeadKLLabel(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
             model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size,
             point_cloud_range=point_cloud_range, predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )
        self.conv_box_std = nn.Conv2d(
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

        # self.uncertainty_alpha = self.model_cfg.UNCERTAINTY_ALPHA
        # self.uncertainty_beta = self.model_cfg.UNCERTAINTY_BETA
        # self.uncertainty_gamma = self.model_cfg.UNCERTAINTY_GAMMA

        self.init_weights()
        self.last_forward_ret_dict = None
        self.log_map = {}

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
        nn.init.normal_(self.conv_box_std.weight, mean=0, std=0.0001)


    def assign_targets(self, gt_boxes, gt_uncertaintys):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes, gt_uncertaintys
        )
        return targets_dict


    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        box_std_preds = self.conv_box_std(spatial_features_2d)
        # box_std_preds = torch.sigmoid(box_std_preds)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_std_preds = box_std_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['box_std_preds'] = box_std_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes'],
                gt_uncertaintys=data_dict['gt_uncertaintys'],
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            # generate_predicted_boxes is true when testing or roi_head
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            batch_box_std_preds = box_std_preds.view(batch_box_preds.shape)
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['batch_box_std_preds'] = batch_box_std_preds
            data_dict['cls_preds_normalized'] = False

        # import pdb;pdb.set_trace()

        # ad hoc debug
        # self.log_map['data_dict'] = data_dict

        return data_dict

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)
        return torch.index_select(a, dim, order_index)

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_std_preds = self.forward_ret_dict['box_std_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets'] # [4, 70400, 7]
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        gt_uncertaintys = self.forward_ret_dict['reg_weights']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
                # [1, 200, 176, 1, 2, 7])
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors

        # anchors [N, 200*176*anchor_num(2)=70400, 7]
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1) # [4, 281600, 7]
        # box_preds = box_preds.view(batch_size, -1,
        #                            box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
        #                            box_preds.shape[-1])
        
        # batch_box_preds = box_preds.view(batch_size, num_anchors, -1)
        # pred_boxes = self.box_coder.decode_torch(batch_box_preds, anchors)[positives].view(-1, 7)
        # gt_boxes = self.box_coder.decode_torch(box_reg_targets, anchors)[positives].view(-1, 7)


        # fake gt_uncertaintys
        label_var_log = torch.log(gt_uncertaintys + 1e-10)

        # import pdb;pdb.set_trace()
        # label_std = self.uncertainty_beta * torch.exp(gt_uncertaintys * self.uncertainty_alpha) + self.uncertainty_gamma # todo parameter
        # label_std = torch.clamp(label_std, 1.0, 100.0)
        # label_std = label_std.unsqueeze(2)
        # label_std = self.tile(label_std, 2, 7)

        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        
        box_std_preds[box_std_preds < -50 ] = -50
        box_std_preds = box_std_preds.view(box_preds.shape)
        label_var_log = label_var_log.reshape(box_preds.shape)
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        l1_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]

        # loc_loss_src = torch.exp(-box_std_preds) * l1_loss_src \
        #                + torch.exp(label_var_log)/torch.exp(box_std_preds) * reg_weights.unsqueeze(-1) \
        #                + 0.5 * torch.log(torch.exp(box_std_preds)/torch.exp(label_var_log)) * reg_weights.unsqueeze(-1)
        loc_loss_src = torch.exp(-box_std_preds) * l1_loss_src
        
        # loc_loss_square = torch.exp(label_var_log)/torch.exp(box_std_preds) * reg_weights.unsqueeze(-1)
        loc_loss_square = torch.exp(label_var_log - box_std_preds) * reg_weights.unsqueeze(-1)

        # loc_loss_log = 0.5 * torch.log(torch.exp(box_std_preds)/torch.exp(label_var_log)) * reg_weights.unsqueeze(-1)
        # loc_loss_log = - 0.5 * torch.log(torch.exp(label_var_log) / torch.exp(box_std_preds) + 1e-10) * reg_weights.unsqueeze(-1)
        loc_loss_log = - 0.5 * (label_var_log - box_std_preds) * reg_weights.unsqueeze(-1)

        # print(f'loc_loss_log min = {loc_loss_log.min()}, loc_loss_log.max= {loc_loss_log.max()}')
        

        loc_loss_src_mean = loc_loss_src.sum() / batch_size
        loc_loss_square_mean = loc_loss_square.sum() / batch_size
        loc_loss_log_mean = loc_loss_log.sum() / batch_size
        # import pdb;pdb.set_trace()

        # print(f"loc_loss_src_mean={loc_loss_src_mean} loc_loss_square_mean={loc_loss_square_mean} loc_loss_log_mean = {loc_loss_log_mean}")
        # if torch.any(torch.isnan(loc_loss_src)) or torch.any(torch.isnan(loc_loss_square)) or torch.any(torch.isnan(loc_loss_log)):
        #     # np.save('box_std_preds.npy', box_std_preds.cpu().numpy())
        #     # np.save('label_var_log.npy', label_var_log.cpu().numpy())
        #     # np.save('reg_weights.npy', reg_weights.cpu().numpy())
        #     import pdb;pdb.set_trace()



        loc_loss = loc_loss_src_mean + loc_loss_square_mean + loc_loss_log_mean
        # print("### l1_loss part = ", (torch.exp(-box_std_preds) * l1_loss_src).sum()/batch_size/2.0)
        # print("### std part = ", (0.5 * box_std_preds * reg_weights.unsqueeze(-1)).sum()/batch_size/2.0)

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item(),
            'rpn_loss_loc_src': loc_loss_src_mean.item() * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight'],
            'rpn_loss_loc_square': loc_loss_square_mean.item() * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight'],
            'rpn_loss_loc_log': loc_loss_log_mean.item() * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        }
        # ad hoc debug
        # tb_dict['l1_loss_src'] = l1_loss_src
        # tb_dict['box_std_preds'] = box_std_preds
        # tb_dict['label_var_log'] = label_var_log
        # tb_dict['reg_weights'] = reg_weights

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

        self.last_forward_ret_dict = self.forward_ret_dict
        return box_loss, tb_dict


class AnchorHeadKLLabelIoU(AnchorHeadKLLabel):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(model_cfg=model_cfg, input_channels=input_channels, num_class=num_class,
                                class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
                                 predict_boxes_when_training = predict_boxes_when_training
        )

        self.conv_iou = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.add_module(
            'iou_loss_func',
            loss_utils.WeightedSmoothL1Loss()
        )

        nn.init.normal_(self.conv_iou.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        box_std_preds = self.conv_box_std(spatial_features_2d)
        iou_preds = self.conv_iou(spatial_features_2d)

        # box_std_preds = torch.sigmoid(box_std_preds)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        iou_preds = iou_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_std_preds = box_std_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['box_std_preds'] = box_std_preds
        self.forward_ret_dict['iou_preds'] = iou_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes'],
                gt_uncertaintys=data_dict['gt_uncertaintys'],
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            # generate_predicted_boxes is true when testing or roi_head
            batch_cls_preds, batch_iou_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, iou_preds=iou_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            batch_box_std_preds = box_std_preds.view(batch_box_preds.shape)
            batch_cls_preds = torch.sigmoid(batch_cls_preds)
            # ad hoc
            # batch_cls_preds[batch_cls_preds < 0.1] = -1
            batch_cls_preds[batch_cls_preds < self.model_cfg.PRE_CLS_THRESH] = 0

            batch_iou_preds = (batch_iou_preds + 1) * 0.5 
            batch_iou_preds[batch_iou_preds < self.model_cfg.PRE_IOU_THRESH] = 0
            batch_cls_preds = batch_cls_preds * torch.pow(batch_iou_preds, self.model_cfg.POW)

            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['batch_box_std_preds'] = batch_box_std_preds
            data_dict['cls_preds_normalized'] = True

        return data_dict
    
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


class AnchorHeadKLLabelIoUGuide(AnchorHeadKLLabelIoU):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True,**kwargs):
        super().__init__(model_cfg=model_cfg, input_channels=input_channels, num_class=num_class,
                                class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
                                 predict_boxes_when_training = predict_boxes_when_training
        )

        self.std_conv1 = nn.Conv2d(self.num_anchors_per_location * self.box_coder.code_size, 64, 1)
        self.std_relu = nn.ReLU()
        self.std_conv2 = nn.Conv2d(64, 1, 1)
        self.std_sigmoid = nn.Sigmoid()

        nn.init.normal_(self.std_conv1.weight, mean=0, std=0.001)
        nn.init.constant_(self.std_conv1.bias, 0)
        nn.init.normal_(self.std_conv2.weight, mean=0, std=0.001)
        nn.init.constant_(self.std_conv2.bias, 0)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        box_std_preds = self.conv_box_std(spatial_features_2d)
        iou_preds = self.conv_iou(spatial_features_2d)

        std_fc1_out = self.std_conv1(box_std_preds)
        std_relu_out = self.std_relu(std_fc1_out)
        std_fc2_out = self.std_conv2(std_relu_out)
        std_sigmoid_out = self.std_sigmoid(std_fc2_out)
        iou_preds = iou_preds * std_sigmoid_out

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        iou_preds = iou_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_std_preds = box_std_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['box_std_preds'] = box_std_preds
        self.forward_ret_dict['iou_preds'] = iou_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes'],
                gt_uncertaintys=data_dict['gt_uncertaintys'],
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            # generate_predicted_boxes is true when testing or roi_head
            batch_cls_preds, batch_iou_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, iou_preds=iou_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            batch_box_std_preds = box_std_preds.view(batch_box_preds.shape)
            batch_cls_preds = torch.sigmoid(batch_cls_preds)
            # ad hoc
            # batch_cls_preds[batch_cls_preds < 0.1] = -1
            batch_cls_preds[batch_cls_preds < self.model_cfg.PRE_CLS_THRESH] = 0

            batch_iou_preds = (batch_iou_preds + 1) * 0.5 
            batch_iou_preds[batch_iou_preds < self.model_cfg.PRE_IOU_THRESH] = 0
            batch_cls_preds = batch_cls_preds * torch.pow(batch_iou_preds, self.model_cfg.POW)

            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['batch_box_std_preds'] = batch_box_std_preds
            data_dict['cls_preds_normalized'] = True

        return data_dict
  