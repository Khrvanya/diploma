from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

from PIL import Image
import numpy as np 
from seaborn import heatmap
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.getcwd(), 'data')


def get_parser():
    """parses terminal arguments"""

    parser = ArgumentParser(description='get parameters', 
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-pl', '--left_image_path', metavar='image.ppm', type=str, 
                        required=True, help='path to the left image file')
    parser.add_argument('-pr', '--right_image_path', metavar='image.ppm', type=str, 
                        required=True, help='path to the right image file')
    parser.add_argument('-a', '--alpha', metavar=1, type=float, required=False,
                        default=1, help='alpha smoothing parameter for stereo vision')
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
    np_right_image = np.array(right_image, dtype=np.int64)                #####

    assert np_left_image.shape[0] == np_right_image.shape[0], "!!!input shapes are wrong!!!"

    broad_left_image = np_left_image[:, :, np.newaxis, :]
    broad_right_image = np_right_image[:, np.newaxis, :, :]
    
    pixel_diff = np.abs(broad_left_image - broad_right_image)
    flat_pixel_diff = np.sum(pixel_diff, axis=3)

    return flat_pixel_diff


def find_min_weights(pixel_difference: np.ndarray, alpha: float, d_max=10) -> np.ndarray:
    """
    main function for recognizing algorithm that fullfills the array with minimums
    takes pixels' diff from get_images_difference, smoothing coef alpha, d_max shift
    returns 3dim np.ndarray - all possible shifts of images, and best shift indexs
    """

    left_len, right1_len, right2_len = pixel_difference.shape

    loss_values = np.full((left_len, right1_len, d_max), np.inf)
    shift_values = np.ones((left_len, right1_len - 1, d_max), dtype=np.int64)
    loss_values[:, 0, :np.min([d_max, right2_len])] = pixel_difference[:, 0, :d_max]  

    for idx in range(1, right1_len):
        for d_idx in range(d_max):

            pix_diff = pixel_difference[:, idx, idx + d_idx:idx + d_idx + d_max]
            d_diff = np.abs(d_idx - np.arange(pix_diff.shape[1]))
            prev_loss = loss_values[:, idx-1, :pix_diff.shape[1]]

            new_loss = pix_diff + alpha * d_diff + prev_loss

            if pix_diff.shape[1]:
                loss_values[:, idx, d_idx] = np.min(new_loss, axis=1)
                shift_values[:, idx - 1, d_idx] = np.argmin(new_loss, axis=1) 
            
    return shift_values, np.argmin(loss_values[:, -1, :], axis=1)


def get_disparity_map(shifts: np.ndarray, start_indices: np.ndarray) -> np.ndarray:
    """
    goes forward through all possible shifts and finds the best one
    takes shift np.ndarray from find_min_weights, and best start indices for it
    returns 2dim ndarray - disparity map
    """

    disparity_map = np.ones((shifts.shape[0], shifts.shape[1] + 1), dtype=np.int64)
    disparity_map[:, -1] = start_indices

    for idx in range(shifts.shape[1]):
        indices = disparity_map[:, -1 - idx]
        disparity_map[:, -2 - idx] = shifts[:, -1 - idx, :].T[indices].diagonal()

    return disparity_map


if __name__ == '__main__':
    args = get_parser().parse_args()

    pixel_difference = get_pixel_difference(args.left_image_path, args.right_image_path)

    shifts, start_indices = find_min_weights(pixel_difference, args.alpha, args.max_shift)

    disp_map = get_disparity_map(shifts, start_indices)

    Image.fromarray(disp_map * 255 / (args.max_shift - 1)).show()
