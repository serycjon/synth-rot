import numpy as np
import cv2

def checkerboard(h, w, sq=15):
    color1 = (0x80, 0x80, 0x80)
    color2 = (0x60, 0x60, 0x60)

    # https://stackoverflow.com/a/2171883/1705970
    coords = np.ogrid[0:h, 0:w]
    idx = (coords[0] // sq + coords[1] // sq) % 2
    vals = np.array([color1, color2], dtype=np.uint8)
    img = vals[idx]
    return img

def transparent_blend(img):
    """ blends img with checkerboard pattern to visualize transparency in opencv """
    board = checkerboard(img.shape[0], img.shape[1]).astype(np.float32)

    img_bgr = img[..., :3].astype(np.float32)
    img_alpha = img[..., 3]

    # repeat the alpha channel * 3
    alpha_factor = img_alpha[:, :, np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)

    img_bgr = cv2.multiply(alpha_factor, img_bgr)
    board   = cv2.multiply(1-alpha_factor, board)

    dst = cv2.add(img_bgr, board).astype(np.uint8)
    return dst
