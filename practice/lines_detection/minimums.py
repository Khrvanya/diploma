import cv2 as cv
import numpy as np

np.random.seed(47)


def get_n_min_shapes(images, n):
    """Return first shapes of n first min"""
    images = images.copy()

    mins_idxs = np.argpartition(images.flatten(), n)[:n]
    shapes = np.unravel_index(mins_idxs, images.shape)

    return np.array(shapes)  # returns specific shapes: [[0, 0, 0], [0, 2, 1]], not [(0,0), (0,2), (0,1)]


def get_major_idx(min_shapes, axis: slice):
    """
    Returns the most frequent first index in min_shapes at axis
    :param min_shapes: shapes of minimums that we should detect
    :param axis: axis where we detect the most frequent first index
    :return: the most frequent first index in min_shapes at axis
    """

    arr = min_shapes[axis]
    values, counts = np.unique(arr, return_counts=True)
    major_idx = values[np.argmax(counts)]

    print(f'amount of global mins in the image: {np.max(counts)} of {arr.shape[-1]}')
    assert np.max(counts) > arr.shape[-1] / 2, "!!!less than a half of global minimums are detected!!!"

    return major_idx


def cluster_mins_shapes(min_shapes, clusters=5, axis=None):
    """
    Makes clustering of min_shapes in axis at euclidean distance
    :param min_shapes: shapes where we detect clusters
    :param clusters: number of clusters
    :param axis: axes of min_shapes which we use for clustering
    :return: centers shapes of the clusters
    """

    if not axis:
        axis = range(len(min_shapes.shape))

    min_shapes = list(zip(*min_shapes[axis]))
    min_shapes = np.array(min_shapes, dtype=np.float32)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(min_shapes, clusters, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    return center


def get_clef_idx_and_locations(images, clefs_number=5, min_number=100):
    """
    Finds rotation idx and main shapes of clefs location
    :param images: np.array
    :param clefs_number: number of clefs on the image
    :param min_number: number of first glob minimums we take
    :return: shapes, rotation idx
    """

    count_axis, images_axis, color_axis = slice(None, -3), slice(-3, -1), slice(-1)
    min_shapes = get_n_min_shapes(images, min_number)

    rotation_idx = get_major_idx(min_shapes, axis=count_axis)

    mask = (min_shapes[count_axis] == rotation_idx)[0]  # only rotation_idx in count_axis
    idx_min_shapes = min_shapes[1:, mask]  # get rid of count_axis

    shapes = cluster_mins_shapes(idx_min_shapes, clefs_number, axis=images_axis)

    return shapes, rotation_idx
