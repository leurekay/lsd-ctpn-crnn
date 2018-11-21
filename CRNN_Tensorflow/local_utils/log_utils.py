#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-18 下午4:11
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : log_utils.py
# @IDE: PyCharm Community Edition
"""
Set the log config
"""
import logging
from logging import handlers
import os
import os.path as ops
import numpy as np
from typing import List


def init_logger(level=logging.DEBUG, when="D", backup=7,
                _format="%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s",
                datefmt="%m-%d %H:%M:%S"):
    """
    init_log - initialize log module
    :param level: msg above the level will be displayed DEBUG < INFO < WARNING < ERROR < CRITICAL
                  the default value is logging.INFO
    :param when:  how to split the log file by time interval
                  'S' : Seconds
                  'M' : Minutes
                  'H' : Hours
                  'D' : Days
                  'W' : Week day
                  default value: 'D'
    :param backup: how many backup file to keep default value: 7
    :param _format: format of the log default format:
                   %(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s
                   INFO: 12-09 18:02:42: log.py:40 * 139814749787872 HELLO WORLD
    :param datefmt:
    :return:
    """
    formatter = logging.Formatter(_format, datefmt)
    logger = logging.getLogger()
    logger.setLevel(level)

    log_path = ops.join(os.getcwd(), 'logs/shadownet.log')
    _dir = os.path.dirname(log_path)
    if not os.path.isdir(_dir):
        os.makedirs(_dir)

    handler = handlers.TimedRotatingFileHandler(log_path, when=when, backupCount=backup)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = handlers.TimedRotatingFileHandler(log_path + ".log.wf", when=when, backupCount=backup)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def compute_accuracy(ground_truth: List[str], predictions: List[str],
                     display: bool=True) -> np.float32:
    """ Computes accuracy
    TODO: this could probably be optimized

    :param ground_truth:
    :param predictions:
    :param display: Whether to print values to stdout
    :return:
    """
    accuracy = []

    for index, label in enumerate(ground_truth):
        prediction = predictions[index]
        total_count = len(label)
        correct_count = 0
        try:
            for i, tmp in enumerate(label):
                if tmp == prediction[i]:
                    correct_count += 1
        except IndexError:
            continue
        finally:
            try:
                accuracy.append(correct_count / total_count)
            except ZeroDivisionError:
                if len(prediction) == 0:
                    accuracy.append(1)
                else:
                    accuracy.append(0)

    accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
    if display:
        print('Mean accuracy is {:5f}'.format(accuracy))

    return accuracy
