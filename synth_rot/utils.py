import cv2
import numpy as np

def is_cv2():
    # if we are using OpenCV 2, then our cv2.__version__ will start
    # with '2.'
    return check_opencv_version("2.")

def is_cv3():
    # if we are using OpenCV 3.X, then our cv2.__version__ will start
    # with '3.'
    return check_opencv_version("3.")

def check_opencv_version(major, lib=None):
    # if the supplied library is None, import OpenCV
    if lib is None:
        import cv2 as lib
        
    # return whether or not the current OpenCV version matches the
    # major version number
    return lib.__version__.startswith(major)

def compatible_contours(thresh):
    if is_cv2():
	(contours, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
    elif is_cv3():
        (_, contours, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
    return contours
    
def compatible_boundingrect(points):
    if is_cv2():
        points = np.array([p for p in points])
        return cv2.boundingRect(points)
    elif is_cv3():
        return cv2.boundingRect(points)
