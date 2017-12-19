#!/usr/bin/python
from __future__ import print_function

import numpy as np
import cv2
import alpha_utils as au
from utils import compatible_contours, compatible_boundingrect

def vrrotvec2mat(ax_ang):
    """
    Create a rotation matrix corresponding to the rotation around a general
    axis by a specified angle.

    from http://pydoc.net/GBpy/0.1.1/GBpy.tools/
    """
    if ax_ang.ndim == 1:
        if np.size(ax_ang) == 5:
            ax_ang = np.reshape(ax_ang, (5, 1))
            msz = 1
        elif np.size(ax_ang) == 4:
            ax_ang = np.reshape(np.hstack((ax_ang, np.array([1]))), (5, 1))
            msz = 1
        else:
            raise Exception('Wrong Input Type')
    elif ax_ang.ndim == 2:
        if np.shape(ax_ang)[0] == 5:
            msz = np.shape(ax_ang)[1]
        elif np.shape(ax_ang)[1] == 5:
            ax_ang = ax_ang.transpose()
            msz = np.shape(ax_ang)[1]
        else:
            raise Exception('Wrong Input Type')
    else:
        raise Exception('Wrong Input Type')

    direction = ax_ang[0:3, :]
    angle = ax_ang[3, :]

    d = np.array(direction, dtype=np.float64)
    d /= np.linalg.norm(d, axis=0)
    x = d[0, :]
    y = d[1, :]
    z = d[2, :]
    c = np.cos(angle)
    s = np.sin(angle)
    tc = 1 - c

    mt11 = tc*x*x + c
    mt12 = tc*x*y - s*z
    mt13 = tc*x*z + s*y

    mt21 = tc*x*y + s*z
    mt22 = tc*y*y + c
    mt23 = tc*y*z - s*x

    mt31 = tc*x*z - s*y
    mt32 = tc*y*z + s*x
    mt33 = tc*z*z + c

    mtx = np.column_stack((mt11, mt12, mt13, mt21, mt22, mt23, mt31, mt32, mt33))

    inds1 = np.where(ax_ang[4, :] == -1)
    mtx[inds1, :] = -mtx[inds1, :]

    if msz == 1:
        mtx = mtx.reshape(3, 3)
    else:
        mtx = mtx.reshape(msz, 3, 3)

    return mtx

def vrrotmat2vec(mat1, rot_type='proper'):
    """
    Create an axis-angle np.array from Rotation Matrix:
    ====================
    from http://pydoc.net/GBpy/0.1.1/GBpy.tools/

    @param mat:  The nx3x3 rotation matrices to convert
    @type mat:   nx3x3 numpy array

    @param rot_type: 'improper' if there is a possibility of
                      having improper matrices in the input,
                      'proper' otherwise. 'proper' by default
    @type  rot_type: string ('proper' or 'improper')

    @return:    The 3D rotation axis and angle (ax_ang)
                5 entries:
                   First 3: axis
                   4: angle
                   5: 1 for proper and -1 for improper
    @rtype:     numpy 5xn array

    """
    mat = np.copy(mat1)
    if mat.ndim == 2:
        if np.shape(mat) == (3, 3):
            mat = np.copy(np.reshape(mat, (1, 3, 3)))
        else:
            raise Exception('Wrong Input Type')
    elif mat.ndim == 3:
        if np.shape(mat)[1:] != (3, 3):
            raise Exception('Wrong Input Type')
    else:
        raise Exception('Wrong Input Type')

    msz = np.shape(mat)[0]
    ax_ang = np.zeros((5, msz))

    epsilon = 1e-12
    if rot_type == 'proper':
        ax_ang[4, :] = np.ones(np.shape(ax_ang[4, :]))
    elif rot_type == 'improper':
        for i in range(msz):
            det1 = np.linalg.det(mat[i, :, :])
            if abs(det1 - 1) < epsilon:
                ax_ang[4, i] = 1
            elif abs(det1 + 1) < epsilon:
                ax_ang[4, i] = -1
                mat[i, :, :] = -mat[i, :, :]
            else:
                raise Exception('Matrix is not a rotation: |det| != 1')
    else:
        raise Exception('Wrong Input parameter for rot_type')



    mtrc = mat[:, 0, 0] + mat[:, 1, 1] + mat[:, 2, 2]


    ind1 = np.where(abs(mtrc - 3) <= epsilon)[0]
    ind1_sz = np.size(ind1)
    if np.size(ind1) > 0:
        ax_ang[:4, ind1] = np.tile(np.array([0, 1, 0, 0]), (ind1_sz, 1)).transpose()


    ind2 = np.where(abs(mtrc + 1) <= epsilon)[0]
    ind2_sz = np.size(ind2)
    if ind2_sz > 0:
        # phi = pi
        # This singularity requires elaborate sign ambiguity resolution

        # Compute axis of rotation, make sure all elements >= 0
        # real signs are obtained by flipping algorithm below
        diag_elems = np.concatenate((mat[ind2, 0, 0].reshape(ind2_sz, 1),
                                     mat[ind2, 1, 1].reshape(ind2_sz, 1),
                                     mat[ind2, 2, 2].reshape(ind2_sz, 1)), axis=1)
        axis = np.sqrt(np.maximum((diag_elems + 1)/2, np.zeros((ind2_sz, 3))))
        # axis elements that are <= epsilon are set to zero
        axis = axis*((axis > epsilon).astype(int))

        # Flipping
        #
        # The algorithm uses the elements above diagonal to determine the signs
        # of rotation axis coordinate in the singular case Phi = pi.
        # All valid combinations of 0, positive and negative values lead to
        # 3 different cases:
        # If (Sum(signs)) >= 0 ... leave all coordinates positive
        # If (Sum(signs)) == -1 and all values are non-zero
        #   ... flip the coordinate that is missing in the term that has + sign,
        #       e.g. if 2AyAz is positive, flip x
        # If (Sum(signs)) == -1 and 2 values are zero
        #   ... flip the coord next to the one with non-zero value
        #   ... ambiguous, we have chosen shift right

        # construct vector [M23 M13 M12] ~ [2AyAz 2AxAz 2AxAy]
        # (in the order to facilitate flipping):    ^
        #                                  [no_x  no_y  no_z ]

        m_upper = np.concatenate((mat[ind2, 1, 2].reshape(ind2_sz, 1),
                                  mat[ind2, 0, 2].reshape(ind2_sz, 1),
                                  mat[ind2, 0, 1].reshape(ind2_sz, 1)), axis=1)

        # elements with || smaller than epsilon are considered to be zero
        signs = np.sign(m_upper)*((abs(m_upper) > epsilon).astype(int))

        sum_signs = np.sum(signs, axis=1)
        t1 = np.zeros(ind2_sz,)
        tind1 = np.where(sum_signs >= 0)[0]
        t1[tind1] = np.ones(np.shape(tind1))

        tind2 = np.where(np.all(np.vstack(((np.any(signs == 0, axis=1) == False), t1 == 0)), axis=0))[0]
        t1[tind2] = 2*np.ones(np.shape(tind2))

        tind3 = np.where(t1 == 0)[0]
        flip = np.zeros((ind2_sz, 3))
        flip[tind1, :] = np.ones((np.shape(tind1)[0], 3))
        flip[tind2, :] = np.copy(-signs[tind2, :])

        t2 = np.copy(signs[tind3, :])

        shifted = np.column_stack((t2[:, 2], t2[:, 0], t2[:, 1]))
        flip[tind3, :] = np.copy(shifted + (shifted == 0).astype(int))

        axis = axis*flip
        ax_ang[:4, ind2] = np.vstack((axis.transpose(), np.pi*(np.ones((1, ind2_sz)))))

    ind3 = np.where(np.all(np.vstack((abs(mtrc + 1) > epsilon, abs(mtrc - 3) > epsilon)), axis=0))[0]
    ind3_sz = np.size(ind3)
    if ind3_sz > 0:
        phi = np.arccos((mtrc[ind3]-1)/2)
        den = 2*np.sin(phi)
        a1 = (mat[ind3, 2, 1]-mat[ind3, 1, 2])/den
        a2 = (mat[ind3, 0, 2]-mat[ind3, 2, 0])/den
        a3 = (mat[ind3, 1, 0]-mat[ind3, 0, 1])/den
        axis = np.column_stack((a1, a2, a3))
        ax_ang[:4, ind3] = np.vstack((axis.transpose(), phi.transpose()))

    return ax_ang

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
    return cv2.boundingRect(cnt)

def try_get(xs, index, default):
    try:
        return xs[index]
    except IndexError:
        return default

def fit_in_size(img, sz, random_pad=True, center=False, margin=0):
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
    elif np.all(w_scaled <= sz):
        new_sz = w_scaled
    else:
        raise RuntimeError("cannot fit the image inside sz")

    new_sz = (int(new_sz[0]), int(new_sz[1]))
    resized = cv2.resize(img, (new_sz[1], new_sz[0]), interpolation=cv2.INTER_AREA)

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

    return final

def get_axis_angle(angle, angle_in, angle_post):
    R_in = rot_z(angle_in)
    R = rot_x(angle)
    R_post = rot_z(angle_post)
    R_all = np.matmul(R_post, np.matmul(R, R_in))

    axis_angle = vrrotmat2vec(R_all)

    # convert back to R matrix to check correctness
    R_back = vrrotvec2mat(axis_angle)
    close = np.all(np.isclose(R_all, R_back))
    if not close:
        raise RuntimeError("axis_angle reconstructed R not close to the original\n{}\nvs\n{}".format(R_all, R_back))

    return axis_angle

def rotate(img, angle, angle_in=0, angle_post=0, Z=None, center=None, fit_in=True):
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
        # find a homography that fits the whole image inside
        x_box, y_box, w_box, h_box = compatible_boundingrect(
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

if __name__ == '__main__':
    for i in np.linspace(0, 360, 300):
        ax_angle = -60
        rot = rotate(img,
                    angle=i, angle_in=-ax_angle, angle_post=ax_angle,
                    fit_in=True)
        fitted = fit_in_size(rot, np.array([224, 224]), random_pad=False)
        cv2.imshow("fitted", au.transparent_blend(fitted))
        c = cv2.waitKey(20)
        if c == ord('q'):
            break
    cv2.waitKey(0)
