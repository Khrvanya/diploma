from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

from PIL import Image
import numpy as np 

DATA_PATH = os.path.join(os.getcwd(), 'data')
D_MAX = 10


def get_parser():
    """parses terminal arguments"""

    parser = ArgumentParser(description='get parameters', 
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-pl', '--left_image_path', metavar='image.ppm', type=str, 
                        required=True, help='path to the left image file')
    parser.add_argument('-pr', '--right_image_path', metavar='image.ppm', type=str, 
                        required=True, help='path to the right image file')
    parser.add_argument('-sh', '--max_shift', metavar=10, type=int, required=False,
                        default=10, help='max shift parameter for stereo vision')

    return parser


def get_pixel_difference(left_image_path: str, right_image_path: str) -> np.ndarray:
    """
    makes 3dim array of differences of images' pixels (uses l1 norm to flatten last dim)
    takes images' path 
    returns 3dim np.ndarray 
    """

    left_image = Image.open(left_image_path)
    right_image = Image.open(right_image_path)

    np_left_image = np.array(left_image, dtype=np.int64)
    np_right_image = np.array(right_image, dtype=np.int64)[:, :-10]                  #####

    assert np_left_image.shape[0] == np_right_image.shape[0], "!!!input shapes are wrong!!!"

    broad_left_image = np_left_image[:, :, np.newaxis, :]
    broad_right_image = np_right_image[:, np.newaxis, :, :]
    
    pixel_diff = np.abs(broad_left_image - broad_right_image)
    flat_pixel_diff = np.sum(pixel_diff, axis=3)

    return flat_image_diff

    
def find_min_weights(pixel_difference: np.ndarray, alpha: float) -> np.ndarray:
    """
    main function for recognizing algorithm that fullfills the array with minimums
    takes pixels' differences from get_images_difference and smoothing coef alpha
    returns 2dim np.ndarray - 
    """


    return 