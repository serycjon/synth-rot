from __future__ import print_function

import numpy as np
import cv2

import alpha_utils as au
import rotator

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

def main():
    obj_img = cv2.imread('/home/jonas/dev/thesis/synth_rot/images/tux.png', cv2.IMREAD_UNCHANGED)

    ## find object center
    mask = obj_img[..., 3]
    moments = cv2.moments(mask)
    center = (int(moments['m10'] / moments['m00']),
              int(moments['m01'] / moments['m00']))
    start_angles = (0, 0, 0)
    end_angles = (75, 0, 0)

    start_xy = (0, 0)
    end_xy   = (135, 250)

    xs = np.linspace(0, 1, 300)
    angles = [degrees_interp(xs, start_angles[i], end_angles[i]) for i in range(len(start_angles))]
    xys = (np.interp(xs, [0, 1], [start_xy[0], end_xy[0]]),
           np.interp(xs, [0, 1], [start_xy[1], end_xy[1]]))

    xys = zip(*xys)
    angles = zip(*angles)

    for i in range(len(angles)):
        angles[i] = (angles[i], xys[i])

    steps = []
    for i, ((angle, pre_angle, post_angle), (x, y)) in enumerate(angles):
        rot, H = rotator.rotate(obj_img,
                                angle=angle, angle_in=pre_angle, angle_post=post_angle,
                                fit_in=True, return_H=True)
        if i == 0:
            fitted, scale = rotator.fit_in_size(rot, np.array([224, 224]), random_pad=False, return_scale=True)
        else:
            fitted = cv2.resize(rot, (0, 0), fx=scale, fy=scale)
            
        # fitted = rot
        old_center = np.array([[center[0]],
                               [center[1]],
                               [1]])

        new_center = np.matmul(H, old_center)
        new_center = np.array((new_center[0] / new_center[2],
                               new_center[1] / new_center[2])) * scale
        new_center = np.squeeze(new_center)
        xy_shift = np.array((x, y)).transpose()
        new_center -= np.array((x, y)) * scale

        new_center = new_center.astype(np.int)

        steps.append({'img': fitted, 'center': new_center})
        # cv2.circle(fitted, tuple(new_center.astype(np.int)), 3, (0, 0, 255))

        # cv2.imshow("fitted", au.transparent_blend(fitted))
        # c = cv2.waitKey(5)
        # if c == ord('q'):
        #   break

    ## collect the dimensions
    central_origin = (0, 0)
    extremes = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
    for step in steps:
        img, img_center = step['img'], step['center']

        left = 0 - img_center[0]
        right = img.shape[1] - img_center[0]
        top = 0 - img_center[1]
        bottom = img.shape[0] - img_center[1]

        if left < extremes['left']:
            extremes['left'] = left
        if right > extremes['right']:
            extremes['right'] = right
        if top < extremes['top']:
            extremes['top'] = top
        if bottom > extremes['bottom']:
            extremes['bottom'] = bottom

    shift = (-extremes['left'], -extremes['top'])
    new_h, new_w = (extremes['bottom'] - extremes['top'],
                    extremes['right'] - extremes['left'])
    new_h, new_w = int(np.ceil(new_h)), int(np.ceil(new_w))

    canvas = np.zeros((new_h, new_w, 4), dtype=np.float32)
    GT_canvas = np.zeros((new_h, new_w), dtype=np.float32)

    GT_frame = len(steps) / 2
    for i, step in enumerate(steps):
        img, img_center = step['img'], step['center']
        h, w = img.shape[:2]
        current_shift = np.array(shift) - img_center
        current_shift = np.ceil(current_shift).astype(np.int)
        canvas[current_shift[1]:current_shift[1]+h,
               current_shift[0]:current_shift[0]+w,
               :] += img

        if i == GT_frame:
            GT_canvas[current_shift[1]:current_shift[1]+h,
                      current_shift[0]:current_shift[0]+w] = np.squeeze(img[..., 3])

        if True:
            local_canvas = np.zeros((new_h, new_w, 4), dtype=np.uint8)
            local_canvas[current_shift[1]:current_shift[1]+h,
                         current_shift[0]:current_shift[0]+w,
                         :] = img

            cv2.imshow("animation", local_canvas)
            c = cv2.waitKey(5)
            if c == ord('q'):
              break
            

    canvas /= len(steps)

    cv2.imshow("GT", GT_canvas.astype(np.uint8))
    cv2.imshow("canvas", canvas.astype(np.uint8))
    c = cv2.waitKey(0)
    
if __name__ == '__main__':
    main()
