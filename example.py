import argparse
import cv2
import numpy as np
from src.helper import render_displacements, get_displacements
from camera_z_transition import estimate_z_transition
from camera_z_transition.camera_motion import _z_translation_fn


def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='''compute the camera motion optical flow between two images.''')
    parser.add_argument('--images', '-i', nargs=2, type=str, help="path to the two image files")
    args = parser.parse_args()
    return args


def main(images):
    img1 = cv2.imread(images[0], 0)
    img2 = cv2.imread(images[1], 0)
    origins, displacements = get_displacements(img1, img2)
    optical_flow = render_displacements(np.ones_like(img2) * 255, origins, displacements, magnification=1)

    # Normalize and scale optical flow
    h, w = img1.shape
    origins[:, 0] = (origins[:, 0] / w) - 0.5
    origins[:, 1] = (origins[:, 1] / h) - 0.5
    displacements[:, 0] = (displacements[:, 0] / w) - 0.5
    displacements[:, 1] = (displacements[:, 1] / h) - 0.5

    # Estimate transition on the z axis
    z = estimate_z_transition(origins, displacements)

    # up scaling for visualization
    estimated_displacements = np.reshape(_z_translation_fn(origins, z), (-1, 2))
    origins[:, 0] = (origins[:, 0] + 0.5) * w
    origins[:, 1] = (origins[:, 1] + 0.5) * h
    estimated_displacements[:, 0] = (estimated_displacements[:, 0] + 0.5) * w
    estimated_displacements[:, 1] = (estimated_displacements[:, 1] + 0.5) * h
    estimated_optical_flow = render_displacements(np.ones_like(img2) * 255,
                                                  origins.astype(np.int),
                                                  estimated_displacements.astype(np.int),
                                                  magnification=1)


    cv2.imwrite("optical_flow.jpg", optical_flow)
    cv2.imwrite("estimated_optical_flow.jpg", estimated_optical_flow)
    print("Estimated transition:{}".format(z))


if __name__ == "__main__":
    args = parseargs()
    main(**args.__dict__)
