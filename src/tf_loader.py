#!/usr/bin/python
''' based on http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/ '''

from __future__ import print_function
import tensorflow as tf

IMAGE_HEIGHT = 224
IMAGE_WIDTH  = 224

def read_and_decode(filename_queue, batch_size=2):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'base_raw': tf.FixedLenFeature([], tf.string),
            'rot_raw': tf.FixedLenFeature([], tf.string),
            'rot_angle': tf.FixedLenFeature([], tf.float32)})

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    base = tf.decode_raw(features['base_raw'], tf.uint8)
    rot = tf.decode_raw(features['rot_raw'], tf.uint8)
    
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    angle = features['rot_angle']
    
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
    
    bases, rots, angles = tf.train.shuffle_batch([base, rot, angle],
                                                 batch_size=batch_size,
                                                 capacity=30,
                                                 num_threads=2,
                                                 min_after_dequeue=10)
    
    return bases, rots, angles

if __name__ == '__main__':
    tfrecords_filename = 'synth_rotation.tfrecords'

    filename_queue = tf.train.string_input_producer(
        [tfrecords_filename], num_epochs=10)

    base, rot, angle = read_and_decode(filename_queue, batch_size=5)

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(5):
            b, r, a = sess.run([base, rot, angle])
            print('b[0, ...].shape: {}'.format(b[0, ...].shape))
            print('a: {}'.format(a))

        coord.request_stop()
        coord.join(threads)
