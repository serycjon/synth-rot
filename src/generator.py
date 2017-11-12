#!/usr/bin/python
from __future__ import print_function

import numpy as np
import cv2

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

def alpha_bbox(img):
    alpha = img[:, :, 3]
    ret, thresh = cv2.threshold(alpha,127,255,0)
    contours = cv2.findContours(thresh, 1, 2)
    
    cnt = contours[0]
    return cv2.boundingRect(cnt)

def rotate(img, angle, angle_in=0, angle_post=0, Z=None, center=None, fit_in=True):
    h, w = img.shape[:2]
    if Z is None:
        Z = 800.0
    if center is None:
        center = np.matrix([[h/2.0], [w/2.0]])
    img_corners = get_corners(img)
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

    H, status = cv2.findHomography(np.transpose(img_corners),
                                   np.transpose(rot_projected))
    if fit_in:
        # find a homography that fits the whole image inside
        x_box, y_box, w_box, h_box = cv2.boundingRect(
            np.float32(np.transpose(rot_projected)))
        H, status = cv2.findHomography(np.transpose(img_corners),
                                       np.transpose(rot_projected) - np.float32([x_box, y_box]))
        dst = cv2.warpPerspective(img, H, (w_box, h_box))

        # further refine by tightly fitting the alpha channel
        a_x_box, a_y_box, a_w_box, a_h_box = alpha_bbox(dst)
        H, status = cv2.findHomography(np.transpose(img_corners),
                                       np.transpose(rot_projected) - np.float32([x_box + a_x_box, y_box + a_y_box]))
        dst = cv2.warpPerspective(img, H, (a_w_box, a_h_box))

    else:
        dst = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
    return dst

img = cv2.imread("images/tux.png", cv2.IMREAD_UNCHANGED)

cv2.imshow("tux", img)
for i in np.linspace(0, 360, 300):
    ax_angle = -60
    rot = rotate(img,
                 angle=i, angle_in=-ax_angle, angle_post=ax_angle,
                 fit_in=True)
    cv2.imshow("rot", rot)
    c = cv2.waitKey(20)
    if c == ord('q'):
        break
cv2.waitKey(0)
