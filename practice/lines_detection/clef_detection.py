import time
import numpy as np

from utils import read_image, imshow
from visualization import draw_dots_on_image
from minimums import get_clef_idx_and_locations
from convolution import convolve_mse_rotate_images

np.random.seed(47)


def angle_idx_to_degrees(idx, rotations):
    """Transforms index to degrees"""

    return 360 / rotations * idx


def detect_clef_coordinates(stave, clef, clefs_number, rotations=36):
    """
    Using fft convolution detect clef coordinates on the stave
    :param stave: the stave image
    :param clef: the clef image
    :param clefs_number: number of clefs on the stave
    :param rotations: number of clef rotations, defines the single rotation angle
    :return: stave rotation angle
    """

    convolution = convolve_mse_rotate_images(stave, clef, rotations, 'valid', 'fft')  # convs_raw
    clefs_coordinates, angle_idx = get_clef_idx_and_locations(convolution, clefs_number, min_number=100)

    print(f'angle of rotation: {angle_idx_to_degrees(angle_idx, rotations)}')

    return clefs_coordinates


if __name__ == '__main__':
    a = time.time()
    main_image = read_image('data/stave.jpg')
    sub_image = read_image('data/clef.jpg')
    coords = detect_clef_coordinates(main_image, sub_image, clefs_number=5)
    b = time.time()

    print(f'Running time is {b - a}')

    imshow(draw_dots_on_image(main_image, coords + (int(np.sqrt(np.sum(np.array(sub_image.shape) ** 2))) + 1)//2), 'test.jpg')
