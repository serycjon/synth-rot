#!/usr/bin/python
''' based on http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/ '''

from __future__ import print_function
import tensorflow as tf
import cv2
import numpy as np

if __name__ == '__main__':
    tfrecords_filename = 'synth_rotation.tfrecords'
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

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
        cv2.imshow('example', composition)
        print('angle: {}'.format(angle))
        c = cv2.waitKey(0)
        if c == ord('q'):
            break

    hist, _ = np.histogram(angles, bins=10, range=(0, 90))
    print('histogram of angles: {}'.format(hist))
