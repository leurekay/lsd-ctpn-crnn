#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-29 下午3:56
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : demo_shadownet.py
# @IDE: PyCharm Community Edition
"""
Use shadow net to recognize the scene text
"""
from typing import Tuple

import tensorflow as tf
import os.path as ops
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from easydict import EasyDict

try:
    from cv2 import cv2
except ImportError:
    pass

from crnn_model import crnn_model
from local_utils import log_utils, data_utils
from local_utils.config_utils import load_config

logger = log_utils.init_logger()


def init_args() -> Tuple[argparse.Namespace, EasyDict]:
    """

    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='Path to the image to be tested',
                        default='data/test_images/test_01.jpg')
    parser.add_argument('--weights_path', type=str, help='Path to the pre-trained weights to use',
                        default='model/shadownet/shadownet_2017-09-29-19-16-33.ckpt-39999')
    parser.add_argument('-f', '--config_file', type=str,
                        help='Use this global configuration file')
    parser.add_argument('-c', '--chardict_dir', type=str,
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-n', '--num_classes', type=int, default=37,
                        help='Force number of character classes to this number. '
                             'Set to 0 for auto (read from charset_dir)')

    args = parser.parse_args()

    config = load_config(args.config_file)
    if args.chardict_dir:
        config.cfg.PATH.CHAR_DICT_DIR = args.chardict_dir

    return args, config.cfg


def recognize(image_path: str, weights_path: str, cfg: EasyDict, is_vis: bool=True, num_classes: int=0):
    """

    :param image_path:
    :param weights_path: Path to stored weights
    :param cfg:
    :param is_vis:
    :param num_classes:
    """

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, tuple(cfg.ARCH.INPUT_SIZE))
    image = np.expand_dims(image, axis=0).astype(np.float32)

    w, h = cfg.ARCH.INPUT_SIZE
    inputdata = tf.placeholder(dtype=tf.float32, shape=[1, h, w, cfg.ARCH.INPUT_CHANNELS], name='input')

    codec = data_utils.TextFeatureIO(char_dict_path=ops.join(cfg.PATH.CHAR_DICT_DIR, 'char_dict.json'),
                                     ord_map_dict_path=ops.join(cfg.PATH.CHAR_DICT_DIR, 'ord_map.json'))

    num_classes = len(codec.reader.char_dict) + 1 if num_classes == 0 else num_classes

    net = crnn_model.ShadowNet(phase='Test',
                               hidden_nums=cfg.ARCH.HIDDEN_UNITS,
                               layers_nums=cfg.ARCH.HIDDEN_LAYERS,
                               num_classes=num_classes)

    with tf.variable_scope('shadow'):
        net_out = net.build_shadownet(inputdata=inputdata)

    decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=net_out, sequence_length=cfg.ARCH.SEQ_LENGTH*np.ones(1),
                                               merge_repeated=False)

    # config tf session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = cfg.TRAIN.TF_ALLOW_GROWTH

    # config tf saver
    saver = tf.train.Saver()

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        preds = sess.run(decodes, feed_dict={inputdata: image})

        preds = codec.writer.sparse_tensor_to_str(preds[0])

        logger.info('Predict image {:s} label {:s}'.format(ops.split(image_path)[1], preds[0]))

        if is_vis:
            plt.figure('CRNN Model Demo')
            plt.imshow(cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, (2, 1, 0)])
            plt.show()

        sess.close()


if __name__ == '__main__':
    args, cfg = init_args()
    if not ops.exists(args.image_path):
        raise ValueError('{:s} doesn\'t exist'.format(args.image_path))

    recognize(image_path=args.image_path, cfg=cfg, weights_path=args.weights_path, num_classes=args.num_classes)
