import cv2
import numpy as np


def render_displacements(frame, origins, displacements, magnification=1):
    """ Render small vector lines on the image representing the optical flow.

    Parameters
    ----------
    frame: numpy array,
        Array representing the RGB frame.
    origins: numpy array,
        Array representing the origin of each grid point.
    displacements: numpy array,
        Array representing the displacements of each grid point.

    Returns
    -------
        The image as an RBG numpy array with the rendered lines.
    """
    out_frame = np.array(frame)
    for v0, v1 in zip(np.reshape(origins, (-1, 2)), np.reshape(displacements, (-1, 2))):
        v1 = (v1 - v0) * magnification + v0
        out_frame = cv2.arrowedLine(out_frame, tuple(v0), tuple(v1), (0, 255, 0), thickness=1)
    return out_frame


_k_params = dict(
    winSize=(15, 15),
    maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

_feature_params = dict(maxCorners=1000000, qualityLevel=0.1, minDistance=7, blockSize=7)


def get_displacements(image1, image2, k_params=_k_params, feature_params=_feature_params):
    """ Calculate the displacement between "good" features
    given two consecuitive grayscale images.

    Parameters
    ----------
    iamge1: numpy array,
        First image

    iamge2: numpy array,
        First image

    k_params: dictionary, Optional
        cv2.calcOpticalFlowPyrLK parameters. For more information visit
        https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

    _feature_params: dictionary, Optional
        cv2.goodFeaturesToTrack parameter. for more information please visit
        https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

    Returns
    -------
        - a set of 2D coordinates representing the features location in the first.
        - a set of 2D coordinates representing the matching features
            location in the second image.
    """
    p1 = cv2.goodFeaturesToTrack(image1, mask=None, **feature_params)
    if p1 is None:
        return np.array([[0, 0]]), np.array([[0, 0]])

    p2, st, err = cv2.calcOpticalFlowPyrLK(image1, image2, p1, None, **k_params)
    if p2 is None:
        return np.array([[0, 0]]), np.array([[0, 0]])

    if not np.any(st == 1):
        return np.array([[0, 0]]), np.array([[0, 0]])

    origin = p1[st == 1].reshape(-1, 2)
    dispacement = p2[st == 1].reshape(-1, 2)
    return origin, dispacement
