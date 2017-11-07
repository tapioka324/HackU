# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import numpy as np
import tensorflow as tf


NUM_CLASSES = 4 # 分類するクラス数
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 3


def inference(images, keep_prob):
    """
    モデルを作成する関数

    引数:
    images: 画像
    keep_prob: dropout率

    返り値:
    y_conv: 各クラスの確率
    """
    def weight_variable(shape):
        # 重みを標準偏差0.1の正規分布で初期化
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        # バイアスを標準偏差0.1の正規分布で初期化
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        # 畳み込み層の作成
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        # プーリング層の作成
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 入力を28x28x3に変形
    x_image = tf.reshape(images, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    with tf.name_scope('softmax') as scope:
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv


if __name__ == '__main__':
    test_image = []
    img = cv2.imread('/deeparea/tkojima/991c3820515ff2f8c37a381485a41a6908d9d9e9.jpg')
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.flatten().astype(np.float32)/255.0
    test_image.append(img)
    test_image = np.asarray(test_image)

    images = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
    labels = tf.placeholder(tf.float32, [None, NUM_CLASSES])
    keep_prob = tf.placeholder(tf.float32)

    logits = inference(images, keep_prob)
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    #sess.run(tf.initialize_all_variables())
    sess.run(tf.global_variables_initializer())
    #saver = tf.train.import_meta_graph('/deeparea/hack/model.meta')
    saver.restore(sess, '/deeparea/hack/data/model')

    pred = np.argmax(logits.eval(feed_dict={images: test_image, keep_prob: 1.0 })[0])
    print(pred)
