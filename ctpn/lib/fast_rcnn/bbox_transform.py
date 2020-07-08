# -*- coding: utf-8 -*-
import numpy as np


def bbox_transform(ex_rois, gt_rois):
    """
    computes the distance from ground-truth boxes to the given boxes, normed by their size
    :param ex_rois: n * 4 numpy array, given boxes
    :param gt_rois: n * 4 numpy array, ground-truth boxes
    :return: deltas: n * 4 numpy array, ground-truth boxes
    """
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    assert np.min(ex_widths) > 0.1 and np.min(ex_heights) > 0.1, \
        'Invalid boxes found: {} {}'. \
            format(ex_rois[np.argmin(ex_widths), :], ex_rois[np.argmin(ex_heights), :])

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    # warnings.catch_warnings()
    # warnings.filterwarnings('error')
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths  # 这里的数值可以用来表示预期模型计算出的标记框信息
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)  # 使用log的原因是为了把两个差别拉大
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    return targets


def bbox_transform_inv(boxes, deltas):
    """
    利用deltas 调整 bbox
    """
    print("bbox_transform_inv: The shape of boxes is {}. The shape of deltas is {}.".format(boxes.shape, deltas.shape))
    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths  # 原始框x中心点
    ctr_y = boxes[:, 1] + 0.5 * heights  # 原始框y中心点

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]
    # print("bbox_transform_inv: : The minimum and maximum values of dy are [{},{}]".format(np.min(dy), np.max(dy)))
    # print("bbox_transform_inv: : The minimum and maximum values of dh are [{},{}]".format(np.min(dh), np.max(dh)))

    pred_ctr_x = ctr_x[:, np.newaxis]
    # heights的范围
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    预测框可以超过图像边界，对这个框进行截取
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes
