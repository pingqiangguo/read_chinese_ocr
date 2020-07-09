# chinese_ocr项目读后感

整个项目整体读完也是花了我一个礼拜时间，因为我自己水平也是比较菜，的深度学习相关知识的积累也还不够，所以初期感觉还是比较吃力的。

不过也还好， 我坚持下来了，感觉读完这个项目我收获了很多，感觉这或许是我后面学习深度学习很好的一个起点。

## 项目环境配置

虽然项目需要的库在`setup.py`文件中有说明，但是我配置后发现项目还是没有办法运行。后面看了一看项目的更新时间，猜想这个项目可能需要python3.5才能运行。当然后面证明我的猜想是正确的。

可以使用anaconda自己建立一个环境

```shell
conda create -n  py35ocr python=3.5
```

`setup.py` 中的shell脚步如下：

```shell
pip install numpy scipy matplotlib pillow
pip install easydict opencv-python keras h5py PyYAML
pip install cython==0.24

# for gpu
# pip install tensorflow-gpu==1.3.0
# chmod +x ./ctpn/lib/utils/make.sh
# cd ./ctpn/lib/utils/ && ./make.sh

# for cpu
# pip install tensorflow==1.3.0
# chmod +x ./ctpn/lib/utils/make_cpu.sh
# cd ./ctpn/lib/utils/ && ./make_cpu.sh

```

其中大部分可以使用`conda install` 直接安装，比较特别的是 opencv 和 easydict

opencv 虽然可以使用conda安装,但是conda 默认安装的是阉割版,无法进行图像显示, 我觉得还是使用`pip install opencv-python`安装把， 这样也方便调试。

easydict 应该是conda 没有收入，所以也使用pip安装。

安装上面配置完环境之后，就是运行项目的`demo.py`文件了，是一个示例文件，可以跑通整个项目。

## 项目流程说明

整个项目可以分为两个部分：

- 第一部分使用`cptn`网络识别出图像中每一行，把这一行给框出来
- 第二部分是使用`densenet`对每一行文字进行识别

### cptn 流程说明

个人感觉这一部分有点想目标检测，`cptn`使用了VGG作为主干网络，整个网络会把图像进行压缩16倍，也就是现在的一个像素等于原来的16的像素，然后再对现在的每一个像素进行预测。

简单理解就是原来图像为[16,16,3], 经过压缩后为[1,1,256], 从主观上可以理解为，现在的一个像素等于包含了原来的16的像素的信息，当然通道数量会多一点。

现在项目要做的就是使用双向LSTM对每个像素进行预测。整个过程可以这样理解：

如果我把一行看成一个时间序列，那么我就可以把原始图像拆成很多行，每一行就是一个时间序列，让算法利用时间序列的前后内容对当前位置进行预测，判定它是不是文字信息区域。

从代码上看就是这个双向神经网络层函数，具体步骤如下:

1. [batch, height, width, channel] -> [batch * height, width, channel] 以前batch的基本对象是一张图片，现在的基本对象是图像的一行
2. 采用双向LSTM进行预测
3. 进行维度调整，继续变为[batch, hegiht, width, channel]

```python
@layer
    def Bilstm(self, input, d_i, d_h, d_o, name, trainable=True):
        """ Bidirectional LSTM network

        """
        img = input
        print("Network.Bilstm: the shape of input layer is {}".format(img.shape))
        with tf.variable_scope(name) as scope:
            shape = tf.shape(img)
            # [N H W C] is [batch size, image height, image width, channels]
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            # Because the VGG network compresses the data 16 times.
            # So one pixel now equals 16 pixels.
            # 现在神经的网络的输入数据， 使用一行作为一个时间序列， channel作为一个单次输入数据的长度
            img = tf.reshape(img, [N * H, W, C])
            img.set_shape([None, None, d_i])

            lstm_fw_cell = tf.contrib.rnn.LSTMCell(d_h, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(d_h, state_is_tuple=True)

            lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, img, dtype=tf.float32)
            # 对于双向神经网络， 假设输出维度是128维， 那么双向就是一个元组总共256维，在后面进行维度拼接
            print("Network.Bilstm: lstm_out by bidirectional_dynamic_rnn is {}".format(lstm_out))
            lstm_out = tf.concat(lstm_out, axis=-1)  # Concat by channels

            # 最后一维 2 x d_h相当与对利用一个像素进行一个输出框的预测
            lstm_out = tf.reshape(lstm_out, [N * H * W, 2 * d_h])

            init_weights = tf.truncated_normal_initializer(stddev=0.1)
            init_biases = tf.constant_initializer(0.0)
            weights = self.make_var('weights', [2 * d_h, d_o], init_weights, trainable,
                                    regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [d_o], init_biases, trainable)
            outputs = tf.matmul(lstm_out, weights) + biases

            outputs = tf.reshape(outputs, [N, H, W, d_o])
            # It is equivalent to an output sequence for each pixel of the input image
            return outputs
```

总结一下上面流程做完之后的状况:

1. 上面处理完的图像， 一个像素等于原始图像的16个像素
2. 上面处理完的图像，每个像素是一个高维度向量，它是利用双向LSTM结合图像左右像素信息得到的。

接下来要做事情就是，对每一个像素产生10个anchors的评估，评估信息为对默认anchors的调整信息和这个anchors的包含文字的可能性评分

这里输出的框非常多，可以利用每个框包含文字的可能性信息对框进行一下筛选， 默认只保留得分大于0.7的框， 下一步就是设定阈值， 只保留得分最高的前1000个框。

项目默认设定的10个框， 这10个框， 宽度默认为16，但是高度不同。当然这些预定义的框也是不准的，需要进行调整。 上面神经网络输出的第一组信息就是调整信息: [x_adj, y_adj,  w_adj,h _adj]。

对于y的调整, 算法默认是以y的竖直方向中心的基准点, y_adj 的分数作为基准点需要向上或是向下移动的比例，利用这个来调整基准点。

对于h的调整， 算法默认exp(w_adj)作为调整比例，目的也是希望当真实框大小与预定框相差比较大时候， 这个分数能成指数增加， 这样可以使得得到的框更加精确。

在下面一步就是对框的越界处理的，把框截取到在图像范围以内。

接下来需要做的事情就是对框进行融合，现在是对每一个像素产生一个框，一个像素代表原来图像16个像素。现在需要吧这些像素串起来， 找到图像从的一行。

算法是利用当前框与左右框的高度关系以及两个框高度相似性把它串起来的，这样就等于知道了当前框的下一个框哪一个框了。完成了这一步，接下来就是吧这些框进行融合。 算法的做法是：
现在的一行文字区域相当与有很多小框组成， 现在对这些小框的顶部顶点与底部顶点进行线性回归， 这样那些检测出来比较小的框也能很好的矫正回来。 然后就是将这些小块进行融合成大框。这个框里面框的文字就是一行。

### densenet流程说明

这一部分流程还是比较简单的， 经过上面的CPTN处理之后， 我们已经得到了每行文字的信息，接下来要做的事情就是讲每行文字图像截取出来， 并调整高度为32像素，当然水平方向也进行等比例放大。

 然后就是把图频送到densenet检测， 整个网络会把图像压缩八倍,  也就是原来的 8 x 8 图像会被压缩成为 4 x 4 图像， 相当与现在的每个像素包含了原来的64个像素信息，然后算法就是把图像按列送去检验。 从原始图像角度说就是 按照 32 x 8为一部分图像送去检验。检出一个结果。 
 然后就是对检测出的结果进行筛选