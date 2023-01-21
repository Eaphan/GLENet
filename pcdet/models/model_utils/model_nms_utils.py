import torch

from ...ops.iou3d_nms import iou3d_nms_utils


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None, box_preds_std=None):
    src_box_scores = box_scores
    # import pdb;pdb.set_trace()
    # box_scores = box_scores * box_preds_std.mean(1)
    

    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    # ad hoc
    if nms_config.NMS_TYPE == 'new_nms_gpu' and box_preds_std is not None:
        variance = torch.exp(box_preds_std)
        if score_thresh is not None:
            variance = variance[scores_mask]
    else:
        variance = None

    selected = []
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]

        # print("### boxes_for_nms.shape", boxes_for_nms.shape)
        # print("### boxes_std_for_nms.shape", boxes_std_for_nms.shape)
        variance_for_nms = variance[indices] if variance is not None else None

        # import pdb;pdb.set_trace()
        if nms_config.NMS_TYPE in ['new_nms_gpu']: # todo: softnms_gpu
            # 这里将NMS_POST_MAXSIZE也输入进去，返回 keep_idx(相对 indices), 新 box(如果有 variance)
            

            # print("###", variance_for_nms.shape)
            keep_idx, selected_scores, new_boxes = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH,
                     **nms_config, variance=variance_for_nms
            )
            # import pdb;pdb.set_trace()
            # print("### new_boxes mark1 = ", new_boxes)
            selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]
            new_boxes = new_boxes[keep_idx[:nms_config.NMS_POST_MAXSIZE]]
            # print("### new_boxes mark2 = ", new_boxes)
        else:
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
            )
            selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]
            new_boxes = box_preds[selected]
    else:
        new_boxes = box_preds[[]]

    # import pdb;pdb.set_trace()
    # print("### new_boxes = ", new_boxes)
    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected], new_boxes


def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.shape[1]):
        if score_thresh is not None:
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]
            cur_box_preds = box_preds

        selected = []
        if box_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
            boxes_for_nms = cur_box_preds[indices]
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
            )
            selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

        pred_scores.append(box_scores[selected])
        pred_labels.append(box_scores.new_ones(len(selected)).long() * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes
