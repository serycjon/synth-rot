#!/usr/bin/python
from __future__ import print_function

import numpy as np
import cv2

from . import alpha_utils as au
from .utils import compatible_contours, compatible_boundingrect

def get_corners(img):
    h, w = img.shape[:2]
    corners = np.transpose(np.float32([[0, 0],
                                       [w, 0],
                                       [w, h],
                                       [0, h]]))
    return corners

def rot_x(degrees):
    rads = np.radians(degrees)
    c, s = np.cos(rads), np.sin(rads)
    R = np.matrix([[1, 0, 0],
                   [0, c, -s],
                   [0, s, c]])
    return R

def rot_z(degrees):
    rads = np.radians(degrees)
    c, s = np.cos(rads), np.sin(rads)
    R = np.matrix([[c, -s, 0],
                   [s, c, 0],
                   [0, 0, 1]])
    return R

def add_z(coords, z):
    N = coords.shape[1]
    return np.vstack((coords,
                      np.repeat(z, N)))

def p2e(coords):
    return coords[:-1, ...] / coords[-1, ...]

def e2p(coords):
    return add_z(coords, 1)

def get_camera(img, f=None):
    h, w = img.shape[:2]
    if f is None:
        f = 1

    K = np.matrix([[f, 0, 0],
                   [0, f, 0],
                   [0, 0, 1]])
    return K

def alpha_contour(img):
    alpha = img[:, :, 3]
    ret, thresh = cv2.threshold(alpha,127,255,0)
    contours = compatible_contours(thresh)
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
    return cnt

def alpha_bbox(img):
    cnt = alpha_contour(img)
    return cv2.boundingRect(cnt)

def try_get(xs, index, default):
    try:
        return xs[index]
    except IndexError:
        return default

def fit_in_size(img, sz, random_pad=True, center=False, margin=0, return_scale=False):
    '''
    1. resize so that the img tightly fits into sz, while keeping aspect ratio
    2. pad to the exact sz:
        if random_pad:
            randomly from both sides
        else:
            first img, then pad
    - if margin is specified, there is that amount of padding guaranteed from all sides
    - if center is True, the objects center of mass will be centered
    '''
    channels = try_get(img.shape, 2, 1)
    dtype = img.dtype
    ## Fit while keeping aspect ratio
    img_sz = np.float32(img.shape[:2])
    final_sz = sz.copy()
    sz = sz - margin*2
    ratios = sz / img_sz

    h_scaled = np.round(img_sz * ratios[0])
    w_scaled = np.round(img_sz * ratios[1])
    if np.all(h_scaled <= sz):
        new_sz = h_scaled
        used_ratio = ratios[0]
    elif np.all(w_scaled <= sz):
        new_sz = w_scaled
        used_ratio = ratios[1]
    else:
        raise RuntimeError("cannot fit the image inside sz")

    new_sz = (int(new_sz[0]), int(new_sz[1]))
    resized = cv2.resize(img, (new_sz[1], new_sz[0]), interpolation=cv2.INTER_AREA)
    if channels == 1:
        resized = resized[..., np.newaxis]

    ## pad
    padding = sz - new_sz
    if padding[0] > 0:
        p = padding[0]
        if center:
            pre = int(p/2)
            post = p - pre
        else:
            if random_pad:
                pre = np.random.randint(0, p+1)
                post = p - pre
            else:
                pre = 0
                post = p

        pad_pre = np.zeros((pre, sz[1], channels), dtype=dtype)
        pad_post = np.zeros((post, sz[1], channels), dtype=dtype)
        padded = np.vstack((pad_pre, resized, pad_post))
    elif padding[1] > 0:
        p = padding[1]
        if center:
            pre = int(p/2)
            post = p - pre
        else:
            if random_pad:
                pre = np.random.randint(0, p+1)
                post = p - pre
            else:
                pre = 0
                post = p

        pad_pre = np.zeros((sz[0], pre, channels), dtype=dtype)
        pad_post = np.zeros((sz[0], post, channels), dtype=dtype)
        padded = np.hstack((pad_pre, resized, pad_post))
    else:
        padded = resized
    # finalize by adding the margins
    if margin == 0:
        final = padded
    else:
        final = np.zeros((final_sz[0], final_sz[1], channels), dtype=dtype)
        final[margin:margin+sz[0], margin:margin+sz[1], :] = padded

    if return_scale:
        return final, used_ratio
    else:
        return final

def rotate(img, angle, angle_in=0, angle_post=0, Z=None, center=None, fit_in=True, return_H=False):
    """Synthesize 3D rotation of an object

    The img input image is BGRA, where the alpha channel means an
    object segmentation mask. The object is 3D rotated by three
    consecutive rotations. With coordinate system of x, y along
    columns and rows respectively and z going into the image, the
    three rotations are:
    1) rotation around z axis (angle_in)
    2) rotation around x axis (angle)
    3) rotation around z axis (angle_post)

    - Z -- the focal length of the camera (default to max(img.shape))
    - center -- the center of rotation (default to the - image center)
    - fit_in -- when True, the homography is augmented - in such a way
      that the whole object fits in the result.
    - return_H -- when True, return a tuple (rotated_img, H)
    """

    h, w = img.shape[:2]
    if Z is None:
        Z = max(img.shape)
    if center is None:
        center = np.matrix([[h/2.0], [w/2.0]])
    my_type = np.float32
    img_corners = get_corners(img).astype(my_type)
    space_corners = add_z(img_corners - center, 0)
    R_in = rot_z(angle_in)
    R = rot_x(angle)
    R_post = rot_z(angle_post)
    new_corners = np.matmul(R,
                            np.matmul(R_in,
                                      space_corners))
    new_corners = np.matmul(R_post, new_corners)

    K = get_camera(img, Z)
    P = np.hstack((K, np.transpose(np.matrix([0, 0, Z]))))
    rot_projected = p2e(np.matmul(P, e2p(new_corners))) + center
    rot_projected = rot_projected.astype(my_type)

    H, status = cv2.findHomography(np.transpose(img_corners),
                                   np.transpose(rot_projected))
    if fit_in:
        orig_contour = e2p(np.transpose(np.squeeze(alpha_contour(img))))
        projected_contour = p2e(np.matmul(H, orig_contour))
        mins  = np.amin(projected_contour, axis=1).tolist()
        maxes = np.amax(projected_contour, axis=1).tolist()
        projected_corners = [[mins[0], mins[1]],
                             [mins[0], maxes[1]],
                             [maxes[0], mins[1]],
                             [maxes[0], maxes[1]]]

        wanted_H = int(maxes[1] - mins[1])
        wanted_W = int(maxes[0] - mins[0])

        wanted_corners = [[0, 0],
                          [0, wanted_H],
                          [wanted_W, 0],
                          [wanted_W, wanted_H]]

        fit_H, status = cv2.findHomography(np.array(projected_corners, dtype=np.float64),
                                           np.array(wanted_corners, dtype=np.float64))
        combined_H = np.matmul(fit_H, H)
        H = combined_H
        dst = cv2.warpPerspective(img, H, (wanted_W, wanted_H))
    else:
        dst = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
    if return_H:
        return dst, H
    else:
        return dst

img = cv2.imread("images/tux.png", cv2.IMREAD_UNCHANGED)

if __name__ == '__main__':
    for i in np.linspace(0, 360, 300):
        ax_angle = -60
        rot = rotate(img,
                    angle=i, angle_in=-ax_angle, angle_post=ax_angle,
                    fit_in=True)
        # fitted = rot
        fitted = fit_in_size(rot, np.array([224, 224]), random_pad=False)
        cv2.imshow("fitted", au.transparent_blend(fitted))
        c = cv2.waitKey(20)
        if c == ord('q'):
            break
    cv2.waitKey(0)
