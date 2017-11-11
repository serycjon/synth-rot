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

def rotate(img, angle, angle_in=0, Z=None, center=None):
    h, w = img.shape[:2]
    if Z is None:
        Z = 500.0
    if center is None:
        center = np.matrix([[h/2.0], [w/2.0]])
    img_corners = get_corners(img)
    space_corners = add_z(img_corners - center, 0)
    R_in = rot_z(angle_in)
    R = rot_x(angle)
    new_corners = np.matmul(R,
                            np.matmul(R_in,
                                      space_corners))

    K = get_camera(img, Z)
    P = np.hstack((K, np.transpose(np.matrix([0, 0, Z]))))
    rot_projected = p2e(np.matmul(P, e2p(new_corners)))

    H, status = cv2.findHomography(np.transpose(img_corners),
                                   np.transpose(rot_projected + center))
    dst = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
    return dst

img = cv2.imread("images/tux.png")

cv2.imshow("tux", img)
for i in np.linspace(0, 91, 100):
    rot = rotate(img, angle=i, angle_in=0)
    cv2.imshow("rot", rot)
    c = cv2.waitKey(20)
    if c == ord('q'):
        break
cv2.waitKey(0)
