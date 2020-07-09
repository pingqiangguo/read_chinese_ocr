# -*- coding: utf-8 -*-
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation, Permute
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l2


def conv_block(input, growth_rate, dropout_rate=None, weight_decay=1e-4):
    """
    卷积块, 输入输出图像宽和高不变
    """
    # 批量标准化 -> relu -> conv -> dropout
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(growth_rate, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    if (dropout_rate):
        x = Dropout(dropout_rate)(x)
    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, droput_rate=0.2, weight_decay=1e-4):
    """
    密连块，输入输出图像宽和高不变
    """
    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, droput_rate, weight_decay)
        x = concatenate([x, cb], axis=-1)
        nb_filter += growth_rate  # 额外增加的层数
    return x, nb_filter


def transition_block(input, nb_filter, dropout_rate=None, pooltype=1, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    # 通道压缩
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    if (dropout_rate):
        x = Dropout(dropout_rate)(x)

    if (pooltype == 2):
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    elif (pooltype == 1):
        x = ZeroPadding2D(padding=(0, 1))(x)
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    elif (pooltype == 3):
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    return x, nb_filter


def dense_cnn(input, nclass):
    """
    利用密连网络对一个ocr汉字进行预测
    """
    # TODO: 需要理解一下为什么是这个输入
    print("dense_cnn: The shape of input value is {}".format(input.shape))
    _dropout_rate = 0.2
    _weight_decay = 1e-4

    _nb_filter = 64
    # conv 64 5*5 s=2
    x = Conv2D(_nb_filter, (5, 5), strides=(2, 2), kernel_initializer='he_normal', padding='same',
               use_bias=False, kernel_regularizer=l2(_weight_decay))(input)

    # 64 + 8 * 8 = 128
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # 128 可以看出transition_block 的作用是实现不同通道的信息交互
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # 192 -> 128 # 通道压缩
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    # [B, H, w, C] -> [B, W, H, C]
    print("dense_cnn: Before Permute, the shape of x is {}".format(x.shape))
    x = Permute((2, 1, 3), name='permute')(x)
    print("dense_cnn: After Permute,the shape of x is {}".format(x.shape))
    # [B, W, H, C] -> [B, W, HC]
    x = TimeDistributed(Flatten(), name='flatten')(x)
    print("dense_cnn: After TimeDistribute and Flatten the shape of x is {}".format(x.shape))
    y_pred = Dense(nclass, name='out', activation='softmax')(x)
    #
    # basemodel = Model(inputs=input, outputs=y_pred)
    # basemodel.summary()
    # dense_cnn 把原始图像压缩了8倍， 等价于每次用 8 x 8 的像素对文字进行预测tu
    # 另外一定需要注意的是，他假定一个文字的宽高都大于8. 如果小于8 那么这个文字是找不到的
    # 不过从图像固定高度这件事上，应该是没有风险的
    # 但是也有一点风险需要说明， 那么是对于数字1，在检测的时候非常容易弄丢，因为它非常的窄
    # 比如，假设一个1高度为64个像素，宽度为6个像素，那么随着压缩，他会变成高32个像素，宽3的像素
    # 而dense_net一个像素压缩了原来8个像素信息，所以内容非常容易丢失
    print("dense_cnn: The shape of y_pred is {}".format(y_pred.shape))
    return y_pred


def dense_blstm(input):
    pass


# why?
input = Input(shape=(32, 280, 1), name='the_input')
dense_cnn(input, 5000)
