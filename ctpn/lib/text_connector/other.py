import numpy as np


def normalize(data):
    if data.shape[0] == 0:
        return data
    max_ = data.max()
    min_ = data.min()
    return (data - min_) * 1.0 / (max_ - min_) if max_ - min_ != 0 else data - min_


def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes[:, 0::2] = threshold(boxes[:, 0::2], 0, im_shape[1] - 1)
    boxes[:, 1::2] = threshold(boxes[:, 1::2], 0, im_shape[0] - 1)
    return boxes


class Graph:
    def __init__(self, graph):
        self.graph = graph

    def sub_graphs_connected(self):
        """
        将一行的框连接起来，组成一个大框
        """
        sub_graphs = []
        for index in range(self.graph.shape[0]):
            # 要求水平方向连接而不是竖直方向连接
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v = index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        print("Graph.sub_graphs_connected:\n"
              "\tSub_grapth is {}".format(sub_graphs))
        return sub_graphs
