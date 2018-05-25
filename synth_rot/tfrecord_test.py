#!/usr/bin/python
''' based on http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/ '''

from __future__ import print_function
import tensorflow as tf
import cv2
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', help='tfrecords file to load', required=True)
    parser.add_argument('--compress', help='read compressed files', action='store_true')
    args = vars(parser.parse_args())
    tfrecords_filename = args['db']
    if args['compress']:
        options = tf.python_io.TFRecordOptions(
            compression_type=tf.python_io.TFRecordCompressionType.ZLIB)
    else:
        options = None

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename, options=options)

    angles = []

    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['height']
                     .int64_list
                     .value[0])

        width = int(example.features.feature['width']
                    .int64_list
                    .value[0])

        base_string = (example.features.feature['base_raw']
                       .bytes_list
                       .value[0])

        rot_string = (example.features.feature['rot_raw']
                      .bytes_list
                      .value[0])

        angle = float(example.features.feature['rot_angle']
                      .float_list
                      .value[0])

        angles.append(angle)

        base_1d = np.fromstring(base_string, dtype=np.uint8)
        base = base_1d.reshape((height, width, -1))

        rot_1d = np.fromstring(rot_string, dtype=np.uint8)
        rot = rot_1d.reshape((height, width, -1))

        composition = np.hstack((base, rot))
        cv2.imshow('example', cv2.cvtColor(composition, cv2.COLOR_RGB2BGR))
        print('angle: {}'.format(angle))
        c = cv2.waitKey(0)
        if c == ord('q'):
            break

    hist, _ = np.histogram(angles, bins=10, range=(0, 90))
    print('histogram of angles: {}'.format(hist))
