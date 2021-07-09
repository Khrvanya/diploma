import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def imshow(image, save=''):
    """Show image"""

    plt.imshow(image, 'gray')
    plt.show()

    if save:
        cv.imwrite(f'data/{save}', image)


def read_image(path: str):
    """Reads the image on the path"""

    return cv.imread(path, cv.IMREAD_COLOR)


def norm_image(image):
    """Normalize an image"""

    return (image - image.min()) / np.abs(image.max() - image.min())


def log_image(image):
    """Log image array"""

    image[image < 0] = 0

    return np.log(image + 1)


def float_to_int_image(image):
    """Transform image to float"""

    image[image > 1] = 1   # shouldn't be used
    image[image < 0] = 0

    return (image * 255).astype(np.uint8)


def int_to_float_image(image):
    """Transform image to int"""

    image[image > 255] = 255  # shouldn't be used
    image[image < 0] = 0

    return image.astype(np.float32) / 255


def get_sliced_image(image, new_shape: tuple):
    """
    Returns new centered sliced image
    Function is made for gray images!!
    Only last three shapes are considered to be an image
    :param image: np.array
    :param new_shape: shape of the returned image (is smaller)
    :return: centered image with new_shape
    """

    new_shape, curr_shape = np.asarray(new_shape), np.asarray(image.shape)
    start_idx = ((curr_shape - new_shape) // 2)
    end_idx = start_idx + new_shape  # len(image.shape) == len(new_shape)
    img_slice = [slice(start_idx[k], end_idx[k]) for k in range(len(end_idx))]

    return image[tuple(img_slice)]


def get_padded_image(image, new_shape: tuple, const=0):
    """
    Makes constant padding to the image
    :param image: np.array
    :param new_shape: result shape of the image, must be bigger
    :param const: padding constant
    :return: padded image with const
    """

    top = (new_shape[0] - image.shape[0]) // 2
    bottom = new_shape[0] - image.shape[0] - top
    left = (new_shape[1] - image.shape[1]) // 2
    right = new_shape[1] - image.shape[1] - left

    border = cv.BORDER_CONSTANT
    padded_img = cv.copyMakeBorder(image, top, bottom, left, right, border, value=const)
    if len(padded_img.shape) == 2:  # if the image was gray
        padded_img = padded_img[..., np.newaxis]

    return padded_img
