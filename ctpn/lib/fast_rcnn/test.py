import cv2
import numpy as np

from .config import cfg
from ..utils.blob import im_list_to_blob


def _get_image_blob(im):
    im_orig = im.astype(np.float32, copy=True)
    # cfg.PIXEL_MEANS is the grayscale average of the RGB channel of the origin image
    print("_get_image_blob: cfg.PIXEL_MEANS is {}".format(cfg.PIXEL_MEANS))
    im_orig -= cfg.PIXEL_MEANS
    print("_get_image_blob: The mean value of the image is subtracted.")
    im_shape = im_orig.shape
    print("_get_image_blob: the shape of image is {}".format(im_shape))
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    print("_get_image_blob: im_size_min is {}. im_size_max is {}.".format(im_size_min, im_size_max))

    processed_ims = []
    im_scale_factors = []
    print("_get_image_blob: cfg.TEST.SCALES is {}".format(cfg.TEST.SCALES))
    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        print("_get_image_blob: Zoom in on the image. The magnification of the image is {}".format(im_scale))
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)  # Image magnification factor
        processed_ims.append(im)  # List of images to be processed

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _get_blobs(im, rois):
    blobs = {'data': None, 'rois': None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    return blobs, im_scale_factors


def test_ctpn(sess, net, im, boxes=None):
    """

    :param sess: tensorflow session
    """
    blobs, im_scales = _get_blobs(im, boxes)
    # Because test_ctpn only tests for one image. the length of im_scales is always 1
    if cfg.TEST.HAS_RPN:
        # The shape of im_blob is [batch size, height, width, channels]
        # Tn test_ctpn, the batch size is always 1 and channels is always 3
        im_blob = blobs['data']
        blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
    # The mean of RPN is Region Proposal Networks.
    # The specific content can by refered to this link: https://blog.csdn.net/JNingWei/article/details/78847696
    if cfg.TEST.HAS_RPN:
        # Test data are provided to the model
        feed_dict = {
            net.data: blobs['data'],  # net.data the the placeholder of the network
            net.im_info: blobs['im_info'],
            net.keep_prob: 1.0}

    rois = sess.run([net.get_output('rois')[0]], feed_dict=feed_dict)
    rois = rois[0]

    scores = rois[:, 0]
    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        boxes = rois[:, 1:5] / im_scales[0]
    return scores, boxes
