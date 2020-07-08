# -*- coding:utf-8 -*-
import cv2
import numpy as np

from ctpn.lib.fast_rcnn.config import cfg
from ctpn.lib.fast_rcnn.nms_wrapper import nms
from .other import normalize
from .text_connect_cfg import Config as TextLineCfg
from .text_proposal_connector import TextProposalConnector
from .text_proposal_connector_oriented import TextProposalConnector as TextProposalConnectorOriented


class TextDetector:
    def __init__(self):
        self.mode = cfg.TEST.DETECT_MODE
        if self.mode == "H":  # 表示沿着水平方向检测
            self.text_proposal_connector = TextProposalConnector()
        elif self.mode == "O":
            self.text_proposal_connector = TextProposalConnectorOriented()

    def detect(self, text_proposals, scores, size):
        # 删除得分较低的proposal
        print(
            "TextDetector.detect: \n"
            "\tThe shape of text_proposals is {}.\n "
            "\tThe shape of scores is {}.\n "
            "\tThe size of image is {}.\n".format(text_proposals.shape, scores.shape, size))
        keep_inds = np.where(scores > TextLineCfg.TEXT_PROPOSALS_MIN_SCORE)[0]
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]  # filter

        # 按得分排序
        sorted_indices = np.argsort(scores.ravel())[::-1]  # 按照得分从高到低排列
        text_proposals, scores = text_proposals[sorted_indices], scores[sorted_indices]  # sort

        # 对proposal做nms
        keep_inds = nms(np.hstack((text_proposals, scores)), TextLineCfg.TEXT_PROPOSALS_NMS_THRESH)
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]
        if False:
            # ++++++++++++++ draw rect to black image ++++++++++++++++++++++
            black_img = np.zeros(size, dtype=np.uint8)
            for x1, y1, x2, y2 in text_proposals:
                xmin = int(min(x1, x2))
                xmax = int(max(x1, x2))
                ymin = int(min(y1, y2))
                ymax = int(max(y1, y2))
                cv2.rectangle(black_img, (xmin, ymin), (xmax, ymax), 255)
            cv2.imshow("default", black_img)
            cv2.waitKey()
            cv2.destroyWindow("default")

        # 获取检测结果
        scores = normalize(scores)  # 对得分进行归一化
        text_recs = self.text_proposal_connector.get_text_lines(text_proposals, scores, size)
        # print(text_recs.shape)
        # 过滤boxes
        keep_inds = self.filter_boxes(text_recs)
        text_lines = text_recs[keep_inds]

        # 对lines做nms
        if text_lines.shape[0] != 0:
            keep_inds = nms(text_lines, TextLineCfg.TEXT_LINE_NMS_THRESH)
            text_lines = text_lines[keep_inds]

        return text_lines

    def filter_boxes(self, boxes):
        heights = np.zeros((len(boxes), 1), np.float)
        widths = np.zeros((len(boxes), 1), np.float)
        scores = np.zeros((len(boxes), 1), np.float)
        index = 0
        for box in boxes:
            heights[index] = (abs(box[5] - box[1]) + abs(box[7] - box[3])) / 2.0 + 1
            widths[index] = (abs(box[2] - box[0]) + abs(box[6] - box[4])) / 2.0 + 1
            scores[index] = box[8]
            index += 1

        return np.where((widths / heights > TextLineCfg.MIN_RATIO) & (scores > TextLineCfg.LINE_MIN_SCORE) &
                        (widths > (TextLineCfg.TEXT_PROPOSALS_WIDTH * TextLineCfg.MIN_NUM_PROPOSALS)))[0]
