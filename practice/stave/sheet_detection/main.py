import os.path
import xml.etree.cElementTree as ET
from typing import List, Dict, Optional

import cv2 as cv

from preprocess_image.denoise import denoise
from preprocess_image.dewarp import dewarp
from preprocess_image.util import imshow


def main():
    INPUT_DIR = 'data/input/'
    INPUT_BASE = 'angle_normal.jpg'
    INPUT_PATH = INPUT_DIR + INPUT_BASE
    DEWARPED_FILE = INPUT_BASE[:-4] + '_dewarped.png'
    DEWARPED_PATH = INPUT_DIR + DEWARPED_FILE
    SHOW_STEPS = True
    FORCE_PREPRC = False

    # Check if file exists
    if not os.path.isfile(INPUT_PATH):
        raise FileNotFoundError

    # Preprocessing steps
    if os.path.isfile(DEWARPED_PATH) and not FORCE_PREPRC:
        dewarped_img = cv.imread(DEWARPED_PATH, cv.IMREAD_COLOR)
    else:
        original_img = cv.imread(INPUT_PATH, cv.IMREAD_COLOR)
        denoised_img = denoise(original_img, is_rgb=True)
        dewarped_img = dewarp(denoised_img, is_rgb=True)
        imshow(DEWARPED_FILE, dewarped_img)
        cv.imwrite(DEWARPED_PATH, dewarped_img)


if __name__ == '__main__':
    main()