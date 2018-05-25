from __future__ import print_function

import numpy as np
import cv2

import alpha_utils as au
import rotator
from utils import compatible_contours

def degrees_interp(x, alpha, beta):
    """ convex combination of two angles 
    x is list of combination coefficients from [0, 1]; alpha and beta angles in degrees. Picks the shorter arc. """
    # get both angles to positive values
    alpha = (alpha + 360) % 360
    beta  = (beta + 360) % 360

    diff = abs(beta-alpha)
    if diff >= 180:
        if alpha > beta:
            alpha = alpha - 360
        else:
            beta = beta - 360

    return np.interp(x, [0, 1], [alpha, beta])

def transform_H(H, coords):
    homo = np.array([[coords[0]],
                     [coords[1]],
                     [1]])

    proj = np.matmul(H, homo)
    proj = np.array((proj[0] / proj[2],
                     proj[1] / proj[2]))
    proj = np.squeeze(proj)

    return proj

def motion_blur(img, pre_angles, angles, post_angles, xs, ys, n_steps=100, vis_animation=False):
    """Motion blur an object

    The object is blurred by a 3D rotation and an in-image translation
    using rotator.rotate.
    
    Args:
    img: [H x W x 4] BGRA uint8 image
    pre_angles: (begin, end) angles in degrees, corresponding to rotate() angle_in parameter
    angles: (begin, end) angles in degrees, corresponding to rotate() angle parameter
    post_angles: (begin, end) angles in degrees, corresponding to rotate() angle_post parameter
    xs: (begin, end) pixel coordinates for translation blur
    ys: (begin, end) pixel coordinates for translation blur
    n_steps: number of interpolation steps
    vis_animation: True if the underlying object animation should be shown (default: False)
    
    Outputs:
    canvas: [H' x W' x 4] BGRA uint8 image with the blurred object
    GT_canvas: [H' x W'] uint8 image with the GT object mask in the middle of the motion
    GT_H: the object homography corresponding to the GT_canvas
    homographies: list of homographies from object image frame to the canvas frame (n_steps long)
    """
    ## find object center
    mask = img[..., 3]
    moments = cv2.moments(mask)
    center = (int(moments['m10'] / moments['m00']),
              int(moments['m01'] / moments['m00']))

    ## interpolate the pose
    start_angles = (pre_angles[0], angles[0], post_angles[0])
    end_angles   = (pre_angles[1], angles[1], post_angles[1])

    start_xy = (xs[0], ys[0])
    end_xy   = (xs[1], ys[1])

    interp_xs = np.linspace(0, 1, n_steps)
    angles = [degrees_interp(interp_xs, start_angles[i], end_angles[i]) for i in range(len(start_angles))]
    xys = (np.interp(interp_xs, [0, 1], [start_xy[0], end_xy[0]]),
           np.interp(interp_xs, [0, 1], [start_xy[1], end_xy[1]]))

    xys = zip(*xys)
    angles = zip(*angles)

    for i in range(len(angles)):
        angles[i] = (angles[i], xys[i])

    ## rotate the object in all of the steps
    steps = []
    for i, ((angle, pre_angle, post_angle), (x, y)) in enumerate(angles):
        rot, H = rotator.rotate(img,
                                angle=angle, angle_in=pre_angle, angle_post=post_angle,
                                fit_in=True, return_H=True)

        ## compute the object's center in the rotated image
        new_center = transform_H(H, center)

        # the translation is done by virtually translating the
        # object's center to the opposite side
        center_shift = -np.array((x, y))
        new_center += center_shift

        new_center = new_center.astype(np.int)

        steps.append({'img': rot, 'center': new_center,
                      'shift': center_shift, 'H': H})

    ## now we have the object rotated, but the coordinate system is
    ## different in each image.  This will be unified now.
    # First, collect the extents of the final blurred image
    central_origin = (0, 0)
    extremes = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
    for step in steps:
        img, shift = step['img'], step['shift']

        left = 0 + shift[0]
        right = img.shape[1] + shift[0]
        top = 0 + shift[1]
        bottom = img.shape[0] + shift[1]

        if left < extremes['left']:
            extremes['left'] = left
        if right > extremes['right']:
            extremes['right'] = right
        if top < extremes['top']:
            extremes['top'] = top
        if bottom > extremes['bottom']:
            extremes['bottom'] = bottom

    # The global coordinates and the canvases are created
    top_left = (extremes['left'], extremes['top'])
    new_h, new_w = (extremes['bottom'] - extremes['top'],
                    extremes['right'] - extremes['left'])
    new_h, new_w = int(np.ceil(new_h)), int(np.ceil(new_w))

    canvas = np.zeros((new_h, new_w, 4), dtype=np.float32)
    GT_canvas = np.zeros((new_h, new_w), dtype=np.float32)

    GT_frame = len(steps) / 2

    blur_homographies = []
    ## Finally, the blurring is done by taking average of the images
    for i, step in enumerate(steps):
        img = step['img']
        h, w = img.shape[:2]

        total_shift = step['shift'] - np.array(top_left)

        H = step['H'].copy()

        shift_H = np.eye(3)
        shift_H[:2, 2] = total_shift
        H = np.matmul(shift_H, H)

        blur_homographies.append(H)

        current_corner = np.ceil(total_shift).astype(np.int)

        canvas[current_corner[1]:current_corner[1]+h,
               current_corner[0]:current_corner[0]+w,
               :] += img

        if i == GT_frame:
            GT_canvas[current_corner[1]:current_corner[1]+h,
                      current_corner[0]:current_corner[0]+w] = np.squeeze(img[..., 3])
            GT_H = H

        if vis_animation:
            H_center = transform_H(H, center).astype(np.int)
            local_canvas = np.zeros((new_h, new_w, 4), dtype=np.uint8)
            local_canvas[current_corner[1]:current_corner[1]+h,
                         current_corner[0]:current_corner[0]+w,
                         :] = img
            local_canvas[H_center[1],
                         H_center[0], :] = np.array([0, 0, 255, 255]).T

            cv2.imshow("cv: animation", local_canvas)
            c = cv2.waitKey(0)
            if c == ord('q'):
                import sys
                sys.exit(1)

    canvas /= len(steps)
    ## visualize motion centers
    # for H in blur_homographies:
    #     H_center = transform_H(H, center)
    #     H_center = np.ceil(H_center).astype(np.int)
    #     canvas[H_center[1],
    #            H_center[0],
    #            :] = (0, 0, 255, 255)

    return canvas.astype(np.uint8), GT_canvas.astype(np.uint8), GT_H, blur_homographies
    

def main():
    obj_img = cv2.imread('/home/jonas/dev/thesis/synth_rot/images/tux.png', cv2.IMREAD_UNCHANGED)

    start_angles = (0, 0, 0)
    end_angles = (75, 0, 0)

    start_xy = (0, 0)
    end_xy   = (135, 250)

    blurred, blurred_mask, Hs = motion_blur(obj_img,
                                            pre_angles=(start_angles[0], end_angles[0]),
                                            angles=(start_angles[1], end_angles[1]),
                                            post_angles=(start_angles[2], end_angles[2]),
                                            xs=(start_xy[0], end_xy[0]),
                                            ys=(start_xy[1], end_xy[1]),
                                            n_steps=60,
                                            vis_animation=False)

    ret, thresh = cv2.threshold(blurred_mask,127,255,0)
    contours = compatible_contours(thresh)
    cnt = contours[0]
    cv2.drawContours(blurred, [cnt], 0, (0,0,255,255), 1)

    cv2.imshow("mask", blurred_mask)
    cv2.imshow("blurred blend", au.transparent_blend(blurred))
    cv2.imshow("blurred", blurred)
    c = cv2.waitKey(0)
    
if __name__ == '__main__':
    main()
