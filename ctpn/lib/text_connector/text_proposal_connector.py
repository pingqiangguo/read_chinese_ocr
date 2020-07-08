# -×- coding: utf-8 -*-
import numpy as np

from .other import clip_boxes
from .text_proposal_graph_builder import TextProposalGraphBuilder


class TextProposalConnector:
    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        return graph.sub_graphs_connected()

    @staticmethod
    def fit_y(X, Y, x1, x2):
        assert len(X) != 0
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        p = np.poly1d(np.polyfit(X, Y, 1))  # 采用最小二乘法进行直线拟合，拟合次数为1次
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        # tp=text proposal
        # ++++++++++++++++++ 将选框按照行进行归类 ++++++++++++++++++++++++++++++
        # 原始OCR检出是一系列小框，这里把这些小框按照行统计起来，得到一个包含一行内容的大框
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)
        text_lines = np.zeros((len(tp_groups), 5), np.float32)
        print("TextProposalConnector.get_text_lines: tp_groups is {}".format(tp_groups))
        # +++++++++++++++ 对同一行的选框进行优化，利用直线拟合的方式对框进行估计并进行融合 ++++++++++++++++
        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]

            x0 = np.min(text_line_boxes[:, 0])  # 一行的最左边点
            x1 = np.max(text_line_boxes[:, 2])  # 一行的最右边点

            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5  # w / 2

            # +++++++++++++++++++++++ 直线拟合，并利用拟合结果对左右端点的y进行调整 +++++++++++++++++++++++++
            # lt_y 表示 left top y 这一行的左上角y 剩下的变量也可以类似的理解
            lt_y, rt_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
            lb_y, rb_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)

            # the score of a text line is the average score of the scores
            # of all text proposals contained in the text line
            score = scores[list(tp_indices)].sum() / float(len(tp_indices))
            # ++++++++++++++++++++++++ 记录直线拟合后得到的框 +++++++++++++++++++++++++++++++
            text_lines[index, 0] = x0
            text_lines[index, 1] = min(lt_y, rt_y)
            text_lines[index, 2] = x1
            text_lines[index, 3] = max(lb_y, rb_y)
            text_lines[index, 4] = score
        # 对小框进行一下截取，防止越界
        text_lines = clip_boxes(text_lines, im_size)
        print("TextProposalConnector.get_text_lines: the shape of text_lines is {}".format(text_lines.shape))
        text_recs = np.zeros((len(text_lines), 9), np.float32)
        index = 0
        # ++++++++++++++++++ 记录四个角点坐标 +++++++++++++++++++
        for line in text_lines:
            xmin, ymin, xmax, ymax = line[0], line[1], line[2], line[3]
            # 左上角点
            text_recs[index, 0] = xmin
            text_recs[index, 1] = ymin
            # 右上角点
            text_recs[index, 2] = xmax
            text_recs[index, 3] = ymin
            # 左下角点
            text_recs[index, 4] = xmin
            text_recs[index, 5] = ymax
            # 右下角点
            text_recs[index, 6] = xmax
            text_recs[index, 7] = ymax
            # score
            text_recs[index, 8] = line[4]  # score
            index = index + 1

        return text_recs
