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

def dropout(img):
    ''' insert random circular hole '''
    h, w = img.shape[:2]
    center_x = int(np.random.uniform(0, w))
    center_y = int(np.random.uniform(0, h))
    radius = int(np.random.uniform(h/10, h/3))

    alpha = img[..., 3].copy()
    cv2.circle(alpha, (center_x, center_y), radius, color=0, thickness=-1)
    img[..., 3] = alpha
    return img

def generate_example(img, sz=np.array([224, 224]), margin=5, rotate_base=True):
    if rotate_base:
        base_in_angle = np.random.rand() * 360
    else:
        base_in_angle = 0
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

    dropout_chance = np.random.rand()
    if dropout_chance < 1:
        rot_fitted = dropout(rot_fitted)
    rot_raw = to_rgb(rot_fitted).tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(sz[0]),
        'width': _int64_feature(sz[1]),
        'base_raw': _bytes_feature(base_raw),
        'rot_raw': _bytes_feature(rot_raw),
        'rot_angle': _float_feature(out_angle)}))

    return example

def generate(images, output, N, max_entries=None, rotate_base=True, compress=False):
    if compress:
        options = tf.python_io.TFRecordOptions(
            compression_type=tf.python_io.TFRecordCompressionType.ZLIB)
    else:
        options = None

    writer = tf.python_io.TFRecordWriter(output, options=options)
    for i in range(N):
        if (i > 0) and (max_entries is not None) and (i%max_entries == 0):
            writer.close()
            shard = i/max_entries
            writer = tf.python_io.TFRecordWriter('{}-{}'.format(output, shard), options=options)
        print('generating {}/{}'.format(i+1, N))
        img = random.choice(images)
        example = generate_example(img, rotate_base=rotate_base)
        writer.write(example.SerializeToString())
    writer.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='output name (without .tfrecords)')
    parser.add_argument('--image', help='select one image in images/')
    parser.add_argument('--val', help='use validation image set', action='store_true')
    parser.add_argument('-N', help='number of generated examples', required=True, type=int)
    parser.add_argument('--max', help='max entries per tfrecords file', type=int)
    parser.add_argument('--compress', help='compress the outputs', action='store_true')
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
    generate(images, tfrecords_path, N, args['max'], compress=args['compress'])
