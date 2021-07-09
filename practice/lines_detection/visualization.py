import cv2 as cv
import numpy as np

np.random.seed(47)


def draw_dots_on_image(image, coordinates, color=(0, 0, 255)):
    """
    Draws coordinates on the given image
    :param image: np.array
    :param coordinates: np.array
    :param color: dots color
    :return: modified image
    """

    if len(image.shape) == 2:
        image = image[..., np.newaxis]
    if image.shape[-1] == 1:
        image = np.concatenate([image, image, image], axis=-1)

    for dot in coordinates:
        int_dot = tuple(dot.astype(np.int32)[::-1])   # reversed since dot is coordinates, not shape
        image = cv.circle(image, int_dot, 0, color, 5)

    return image
