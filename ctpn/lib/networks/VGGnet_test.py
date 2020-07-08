import tensorflow as tf

from .network import Network
from ..fast_rcnn.config import cfg


class VGGnet_test(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        # the shape of self.data is [batch size, height, channels]
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        # the value of self.im_info is [width, height, image scaling]
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        # self.keep_prob is used for dropout layer.
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'im_info': self.im_info})
        self.trainable = trainable
        self.setup()

    def setup(self):
        anchor_scales = cfg.ANCHOR_SCALES  # The smallest boxes in ocr
        print("VGGnet_test.setup: the value of anchor_scales is {}".format(anchor_scales))
        _feat_stride = [16, ]

        (self.feed('data')
         .conv(3, 3, 64, 1, 1, name='conv1_1')
         .conv(3, 3, 64, 1, 1, name='conv1_2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')  # 下采样 downsample 1 / 2
         .conv(3, 3, 128, 1, 1, name='conv2_1')
         .conv(3, 3, 128, 1, 1, name='conv2_2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')  # downsample 1 / 4
         .conv(3, 3, 256, 1, 1, name='conv3_1')
         .conv(3, 3, 256, 1, 1, name='conv3_2')
         .conv(3, 3, 256, 1, 1, name='conv3_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')  # downsample 1 / 8
         .conv(3, 3, 512, 1, 1, name='conv4_1')
         .conv(3, 3, 512, 1, 1, name='conv4_2')
         .conv(3, 3, 512, 1, 1, name='conv4_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')  # downsample 1 / 16
         .conv(3, 3, 512, 1, 1, name='conv5_1')
         .conv(3, 3, 512, 1, 1, name='conv5_2')
         .conv(3, 3, 512, 1, 1, name='conv5_3'))

        # The image is scaled by a factor of 16 by four downsamples

        (self.feed('conv5_3').conv(3, 3, 512, 1, 1, name='rpn_conv/3x3'))
        (self.feed('rpn_conv/3x3').Bilstm(512, 128, 512, name='lstm_o'))
        # 有一点不明白的就是如果这里有多个框，那么它是如何起作用的
        (self.feed('lstm_o').lstm_fc(512, len(anchor_scales) * 10 * 4, name='rpn_bbox_pred'))  # 预测框
        (self.feed('lstm_o').lstm_fc(512, len(anchor_scales) * 10 * 2, name='rpn_cls_score'))  #

        #  shape is (1, H, W, Ax2) -> (1, H, WxA, 2)
        (self.feed('rpn_cls_score')
         .spatial_reshape_layer(2, name='rpn_cls_score_reshape')
         .spatial_softmax(name='rpn_cls_prob'))

        # shape is (1, H, WxA, 2) -> (1, H, W, Ax2)
        (self.feed('rpn_cls_prob')
         .spatial_reshape_layer(len(anchor_scales) * 10 * 2, name='rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
         .proposal_layer(_feat_stride, anchor_scales, 'TEST', name='rois'))
