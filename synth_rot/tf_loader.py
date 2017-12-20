#!/usr/bin/python
''' based on http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/ '''

from __future__ import print_function
import tensorflow as tf

IMAGE_HEIGHT = 224
IMAGE_WIDTH  = 224

def read_and_decode(filename_queue, batch_size=2, capacity=30, num_threads=2, compressed=False):
    if compressed:
        options = tf.python_io.TFRecordOptions(
            compression_type=tf.python_io.TFRecordCompressionType.ZLIB)
    else:
        options = None
    reader = tf.TFRecordReader(options=options)
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'base_raw': tf.FixedLenFeature([], tf.string),
            'rot_raw': tf.FixedLenFeature([], tf.string),
            'axis_angle': tf.FixedLenFeature([], tf.string)})

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    base = tf.decode_raw(features['base_raw'], tf.uint8)
    rot = tf.decode_raw(features['rot_raw'], tf.uint8)
    
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    axis_angle = tf.decode_raw(features['axis_angle'], tf.float32)
    axis_angle = tf.slice(axis_angle, [0], [3]) * tf.slice(axis_angle, [3], [1])
    axis_angle = tf.reshape(axis_angle, [1, 3])
    
    base_shape = tf.stack([height, width, 3])
    rot_shape = tf.stack([height, width, 3])
    
    base = tf.reshape(base, base_shape)
    rot = tf.reshape(rot, rot_shape)

    base = tf.image.resize_image_with_crop_or_pad(image=base,
                                                  target_height=IMAGE_HEIGHT,
                                                  target_width=IMAGE_WIDTH)
    rot = tf.image.resize_image_with_crop_or_pad(image=rot,
                                                 target_height=IMAGE_HEIGHT,
                                                 target_width=IMAGE_WIDTH)
    
    bases, rots, axis_angles = tf.train.shuffle_batch([base, rot, axis_angle],
                                                      batch_size=batch_size,
                                                      capacity=capacity,
                                                      num_threads=num_threads,
                                                      min_after_dequeue=10)
    
    return bases, rots, axis_angles

if __name__ == '__main__':
    tfrecords_filename = 'pokus.tfrecords'

    filename_queue = tf.train.string_input_producer(
        [tfrecords_filename], num_epochs=10)

    base, rot, angle = read_and_decode(filename_queue, batch_size=5, compressed=True)

    import numpy as np
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(5):
            b, r, a = sess.run([base, rot, angle])
            print('b[0, ...].shape: {}'.format(b[0, ...].shape))
            # print('a: {}'.format(a))
            # a = a[0, 0, :3]
            print('np.linalg.norm(a): {}'.format(np.linalg.norm(a)))
            print('a: {}'.format(a))

        coord.request_stop()
        coord.join(threads)
