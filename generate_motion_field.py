import argparse
import cv2
import numpy as np
from camera_z_transition.camera_motion import _z_translation_fn
from src.helper import render_displacements


def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='''Renders zoom motion field from equation''')
    parser.add_argument('-z', type=float, help="Amount of transition on the z axis")

    args = parser.parse_args()
    return args


def main(z):
    w = 1000
    h = 1000
    n_cols = 30
    n_rows = 30
    x = np.linspace(-0.5, 0.5, n_cols)
    y = np.linspace(-0.5, 0.5, n_rows)
    origins = np.reshape(np.stack(np.meshgrid(x, y), axis=2), (-1, 2))
    displacements = _z_translation_fn(origins, z)
    displacements = np.reshape(displacements, (-1, 2))

    displacements[:, 0] = (displacements[:, 0] + 0.5) * w
    displacements[:, 1] = (displacements[:, 1] + 0.5) * h
    origins[:, 0] = (origins[:, 0] + 0.5) * w
    origins[:, 1] = (origins[:, 1] + 0.5) * h

    motion_field = render_displacements(np.ones((w, h)) * 250, origins.astype(np.int), displacements.astype(np.int))
    cv2.imwrite("motion_filed_z_{}.jpg".format(z), motion_field)

    pass


if __name__ == "__main__":
    args = parseargs()
    main(**args.__dict__)
