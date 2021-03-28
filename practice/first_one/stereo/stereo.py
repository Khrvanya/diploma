from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

from PIL import Image
import numpy as np 

DATA_PATH = os.path.join(os.getcwd(), 'data')


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
    np_right_image = np.array(right_image, dtype=np.int64)                 #####

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
    shift_values = np.full((left_len, right1_len - 1, d_max), np.inf)
    loss_values[:, 0, :] = pixel_difference[:, 0, :d_max]

    for idx in range(1, right1_len):
        for d_next_idx in range(d_max):

            pix_diff = pixel_difference[:, idx, idx:idx+d_max]
            poss_losses = pix_diff + alpha * np.abs(d_next_idx - np.arange(pix_diff.shape[1])) + loss_values[:, idx-1, :pix_diff.shape[1]]

            if pix_diff.shape[1]:
                loss_values[:, idx, d_next_idx] = np.min(poss_losses, axis=1)
                shift_values[:, idx - 1, d_next_idx] = np.argmin(poss_losses, axis=1) 
            
    return shift_values, np.argmin(loss_values[:, -1, :], axis=1)


def get_disparity_map(shifts: np.ndarray, start_indices: np.ndarray) -> np.ndarray:
    """
    goes forward through all possible shifts and finds the best one
    takes shift np.ndarray from find_min_weights, and best start indices for it
    returns 2dim ndarray - disparity map
    """

    disparity_map = np.ones(shifts.shape[:-1])
    indices = start_indices

    for idx in range(shifts.shape[1]):
        disparity_map[:, -1 - idx] = shifts[:, -1 - idx, :].T[start_indices].diagonal()
        indices = disparity_map[:, -1 - idx]

    return disparity_map
    
    
if __name__ == '__main__':

    d_max = 10
    alpha = .001

    pixel_difference = get_pixel_difference(os.path.join(DATA_PATH, "im0.ppm"), os.path.join(DATA_PATH, "im1.ppm"))
    shifts, start_indices = find_min_weights(pixel_difference, alpha, d_max)

    map = get_disparity_map(shifts, start_indices)

    import seaborn as sns
    sns.heatmap(map, vmax=d_max-1)