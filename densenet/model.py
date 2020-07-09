# -*- coding:utf-8 -*-
import os
from imp import reload

import numpy as np
from PIL import Image
from keras.layers import Input
from keras.models import Model

from . import densenet
from . import keys

# import keras.backend as K

reload(densenet)

characters = keys.alphabet[:]
characters = characters[1:] + u'卍'
nclass = len(characters)

input = Input(shape=(32, None, 1), name='the_input')
y_pred = densenet.dense_cnn(input, nclass)
basemodel = Model(inputs=input, outputs=y_pred)

modelPath = os.path.join(os.getcwd(), 'densenet/models/weights_densenet.h5')
if os.path.exists(modelPath):
    basemodel.load_weights(modelPath)


def decode(pred):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    # print("decode: pred_text is {}".format(pred_text))
    # (not (i > 0 and pred_text[i] == pred_text[i - 1]))
    for i in range(len(pred_text)):
        if (pred_text[i] != nclass - 1) and (
                (not (i > 0 and pred_text[i] == pred_text[i - 1])) or
                (i > 1 and pred_text[i] == pred_text[i - 2])):
            # 如果当前字与前一个不相同，或者与前面第二个相同，则保留
            # print("decode: ++++++++++++++++++++++++")
            # print("decode: pred_text[{}] is {}".format(i - 2, characters[pred_text[i - 2]]))
            # print("decode: pred_text[{}] is {}".format(i - 1, characters[pred_text[i - 1]]))
            # print("decode: pred_text[{}] is {}".format(i, characters[pred_text[i]]))
            # print("decode: ++++++++++++++++++++++++")

            char_list.append(characters[pred_text[i]])
    print("decode: char_list is {}".format(u''.join(char_list)))
    return u''.join(char_list)


def predict(img):
    # 从结果看， 他输入的确实是图片的一行
    # print("predict: the shape of image is {}".format(img.size))
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    width = int(width / scale)

    img = img.resize([width, 32], Image.ANTIALIAS)

    '''
    img_array = np.array(img.convert('1'))
    boundary_array = np.concatenate((img_array[0, :], img_array[:, width - 1], img_array[31, :], img_array[:, 0]), axis=0)
    if np.median(boundary_array) == 0:  # 将黑底白字转换为白底黑字
        img = ImageOps.invert(img)
    '''

    img = np.array(img).astype(np.float32) / 255.0 - 0.5

    X = img.reshape([1, 32, width, 1])
    # 这个X确实不是固定的
    # print("predict: the shape of X is {}".format(X.shape))

    y_pred = basemodel.predict(X)

    y_pred = y_pred[:, :, :]

    # out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :]
    # out = u''.join([characters[x] for x in out[0]])
    out = decode(y_pred)
    print("predict:The shape of y_pred is {}".format(y_pred.shape))

    return out
