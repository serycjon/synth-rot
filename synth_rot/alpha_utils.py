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

def transparent_blend(img, background=None):
    """ blends img with checkerboard pattern to visualize transparency in opencv

    Args:
      img: HxWx4 np.array
      background: (optional) HxWx4 np.array to use instead of the checkerboard
    """
    if background is not None:
        board = background.astype(np.float32)
    else:
        board = checkerboard(img.shape[0], img.shape[1]).astype(np.float32)

    img_bgr = img[..., :3].astype(np.float32)
    img_alpha = img[..., 3].astype(np.float32)

    alpha_factor = np.expand_dims(img_alpha, axis=2) / 255.0

    dst = alpha_factor * img_bgr + (1 - alpha_factor) * board
    return dst.astype(np.uint8)
