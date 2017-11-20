#!/usr/bin/python
''' based on tfrecords-guide at http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/ '''
from __future__ import print_function

import numpy as np
import cv2
import argparse
import rotator
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def no_alpha(img):
    return img[:, :, :3]

def bgr2rgb(img):
    return img[..., [2, 1, 0]]

def to_rgb(bgra):
    return bgra[:, :, [2, 1, 0]]

def generate_example(img, sz=np.array([224, 224]), margin=5):
    base = rotator.rotate(img, 0, angle_in=0, angle_post=0, fit_in=True)
    base_fitted = rotator.fit_in_size(base, sz, random_pad=True)
    base_raw = to_rgb(base_fitted).tostring()

    out_angle = np.random.rand() * (90 - margin)
    in_angle = np.random.rand() * 360
    post_angle = np.random.rand() * 360

    rot = rotator.rotate(img,
                         angle=out_angle, angle_in=in_angle, angle_post=post_angle,
                         fit_in=True)
    rot_fitted = rotator.fit_in_size(rot, sz, random_pad=True)
    rot_raw = to_rgb(rot_fitted).tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(sz[0]),
        'width': _int64_feature(sz[1]),
        'base_raw': _bytes_feature(base_raw),
        'rot_raw': _bytes_feature(rot_raw),
        'rot_angle': _float_feature(out_angle)}))

    return example
    

if __name__ == '__main__':
    tfrecords_path = 'tmp.tfrecords'
    img = cv2.imread("images/tux.png", cv2.IMREAD_UNCHANGED)
    N = 60

    with tf.python_io.TFRecordWriter(tfrecords_path) as writer:
        for i in range(N):
            if i % (N/10) == 0:
                print('{}%'.format(10*(i/(N/10))))
            example = generate_example(img)
            writer.write(example.SerializeToString())
