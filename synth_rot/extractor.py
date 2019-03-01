'''
extract RGBA object image from RGB, segmentation pair
'''
from __future__ import print_function

import os
import argparse

import numpy as np
import cv2

from .utils import compatible_contours
from . import alpha_utils as au

def crop_to_alpha(img):
    ''' crop BGRA image to the minimal straight rectangle '''
    alpha = img[..., 3]
    ret, thresh = cv2.threshold(alpha,127,255,0)
    contours = compatible_contours(thresh)
    try:
        if len(contours) > 1:
            max_area = 0
            contour = None
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    contour = cnt
            cnt = contour
        else:
            cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
    except IndexError:
        x, y, w, h = 0, 0, 1, 1

    crop = img[y:y+h, x:x+w, :]
    return crop

def extract_object(img, seg):
    obj_with_alpha = np.dstack((img, seg))
    return crop_to_alpha(obj_with_alpha)

def extract_from_sequence(base_path, sequence, frame):
    ''' extract object from cointracking sequence stored in base_path.
    base_path contains images/sequence/%08d.jpg and segmentations/sequence/%08d.png '''
    img_path = os.path.join(base_path, 'images', sequence, '{:08}.jpg'.format(frame))
    seg_path = os.path.join(base_path, 'segmentations', sequence, '{:08}.png'.format(frame))

    img = cv2.imread(img_path)
    seg = cv2.imread(seg_path, 0)

    return extract_object(img, seg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', help='path to dir containing images/ and segmentations/',
                        default='/home/jonas/dev/thesis/data/cointracking/')
    parser.add_argument('--sequence', help='sequence name')
    parser.add_argument('--frame', help='frame number (starts from 1)', type=int)

    args = vars(parser.parse_args())

    if (args['base_path'] is None) or (args['sequence'] is None) or (args['frame'] is None):
        rgb_path = '/home/jonas/dev/thesis/data/cointracking/images/beermat/00000001.jpg'
        seg_path = '/home/jonas/dev/thesis/data/cointracking/segmentations/beermat/00000001.png'

        rgb = cv2.imread(rgb_path)
        seg = cv2.imread(seg_path, 0)

        obj = extract_object(rgb, seg)
    else:
        obj = extract_from_sequence(args['base_path'], args['sequence'], args['frame'])

    obj = crop_to_alpha(obj)

    cv2.imshow("object", au.transparent_blend(obj))
    cv2.waitKey(0)
