import cv2
import numpy as np

from .other import Graph
from .text_connect_cfg import Config as TextLineCfg


class TextProposalGraphBuilder:
    """
        Build Text proposals into a graph.
    """

    def get_successions(self, index):
        """
        找到右边最邻近的一个矩形框
        """
        box = self.text_proposals[index]
        results = []
        for right in range(int(box[0]) + 1, min(int(box[0]) + TextLineCfg.MAX_HORIZONTAL_GAP + 1, self.im_size[1])):
            adj_box_indices = self.boxes_table[right]  # 每次取一列
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def get_precursors(self, index):
        """
        获得当前框右边最邻近的一个矩形框，要求这个框与目标框大小相似，竖直方向重合
        """
        box = self.text_proposals[index]
        results = []
        # 获得当前框右边的一个框
        for left in range(int(box[0]) - 1, max(int(box[0] - TextLineCfg.MAX_HORIZONTAL_GAP), 0) - 1, -1):
            adj_box_indices = self.boxes_table[left]
            if adj_box_indices and False:
                adj_box = self.text_proposals[adj_box_indices]
                im_size = self.im_size
                black_img = np.zeros(im_size, dtype=np.uint8)
                for x1, y1, x2, y2 in adj_box:
                    xmin = int(min(x1, x2))
                    xmax = int(max(x1, x2))
                    ymin = int(min(y1, y2))
                    ymax = int(max(y1, y2))
                    cv2.rectangle(black_img, (xmin, ymin), (xmax, ymax), 255)
                cv2.imshow("default", black_img)
                cv2.waitKey()
                cv2.destroyWindow("default")
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def is_succession_node(self, index, succession_index):
        """
        利用右边框找左边框，因为框存在重叠的情况，所以可能找到多个，要求当前框得分最高
        """
        precursors = self.get_precursors(succession_index)
        # print("TextProposalGraphBuilder.is_succession_node:\n"
        #       "\t Index is {}.\n"
        #       "\tsuccession_index is {}.\n"
        #       "\tprecursors is {}".format(index, succession_index, precursors))
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True
        return False

    def meet_v_iou(self, index1, index2):
        """
        判断两个矩形框是否是同一列的相似矩形框
        要求1: 矩形框高度相似
        要求2: 矩形框不互相重叠
        """

        def overlaps_v(index1, index2):
            """
            计算两个矩形框竖直方向的重合度
            """
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            y0 = max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1 = min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            # max(0, y1 - y0 + 1) 等价于两个矩形框中间间隙的高度
            return max(0, y1 - y0 + 1) / min(h1, h2)

        def size_similarity(index1, index2):
            """
            计算两个矩形框的高度相似性
            """
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            return min(h1, h2) / max(h1, h2)

        if False:
            x1, y1, x2, y2 = self.text_proposals[index1]
            x3, y3, x4, y4 = self.text_proposals[index2]
            im_size = self.im_size
            black_img = np.zeros(im_size, dtype=np.uint8)
            cv2.rectangle(black_img, (x1, y1), (x2, y2), 255)
            cv2.rectangle(black_img, (x3, y3), (x4, y4), 255)
            wname = "meet_v_iou:overlaps_v is {} size_similarity is {}".format(overlaps_v(index1, index2),
                                                                               size_similarity(index1, index2))
            cv2.imshow(wname, black_img)
            cv2.waitKey()
            cv2.destroyWindow(wname)
        # overlaps_v(index1, index2) >= TextLineCfg.MIN_V_OVERLAPS
        return overlaps_v(index1, index2) >= TextLineCfg.MIN_V_OVERLAPS and \
               size_similarity(index1, index2) >= TextLineCfg.MIN_SIZE_SIM

    def build_graph(self, text_proposals, scores, im_size):
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1

        boxes_table = [[] for _ in range(self.im_size[1])]  # self.im_size[1] is the width of the image
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table = boxes_table

        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions = self.get_successions(index)  # 根据当前索引获得其所有的重叠框索引
            if len(successions) == 0:
                continue
            # print(successions)
            # 当一个框有多个重叠框时候，选择得分最大的重叠框
            succession_index = successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions(precursors) if multiple successions(precursors)
                # have equal scores.
                graph[index, succession_index] = True  # 这个记为True表示当前框与下一个框项链
        return Graph(graph)
