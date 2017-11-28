#!/usr/bin/python
''' based on tfrecords-guide at http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/ '''
from __future__ import print_function

import numpy as np
import cv2
import argparse
import rotator
import tensorflow as tf
import os
import sys
import random

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

def to_rgb(bgra, default=0):
    rgb = bgra[:, :, [2, 1, 0]]
    rgb[bgra[:, :, 3] < 127] = default
    return rgb

def get_valid_images(path):
    ''' get all png images with transparency from path '''
    print('Loading images from {}'.format(path))
    images = []
    files = next(os.walk(path))[2]
    for file in files:
        if os.path.splitext(file)[1] == '.png':
            img = cv2.imread(os.path.join(path, file), cv2.IMREAD_UNCHANGED)  # read RGBA
            if img.shape[2] == 4:
                images.append(img)
            else:
                print('Not using image without alpha: {}'.format(file))
    return images

def generate_example(img, sz=np.array([224, 224]), margin=5):
    base_in_angle = np.random.rand() * 360
    base = rotator.rotate(img, 0, angle_in=base_in_angle, angle_post=0, fit_in=True)
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

def generate(images, output, N):
    with tf.python_io.TFRecordWriter(output) as writer:
        for i in range(N):
            print('generating {}/{}'.format(i+1, N))
            img = random.choice(images)
            example = generate_example(img)
            writer.write(example.SerializeToString())
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='output name (without .tfrecords)')
    parser.add_argument('--image', help='select one image in images/')
    parser.add_argument('--val', help='use validation image set', action='store_true')
    parser.add_argument('-N', help='number of generated examples', required=True, type=int)
    args = vars(parser.parse_args())

    if args['val']:
        base_img_dir = 'val_images'
    else:
        base_img_dir = 'images'
    if args['image'] is not None:
        img_name = args['image']
        images = [cv2.imread(os.path.join(base_img_dir, img_name), cv2.IMREAD_UNCHANGED)]
    else:
        images = get_valid_images(base_img_dir)

    N = args['N']
    tfrecords_path = '{}.tfrecords'.format(args['output'])
    generate(images, tfrecords_path, N)
