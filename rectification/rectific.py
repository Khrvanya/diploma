import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.linalg import null_space


def get_sift(image: np.array) -> tuple:
    """
    Uses SIFT to get keypoints and descriptors from image
    :param image: image
    :return: tuple as (keypoints, descriptors) for the image
    """

    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return keypoints, descriptors


def match_knn(descrs_0: np.array, descrs_1: np.array) -> tuple:
    """
    Matches descriptors with knn method
    :param descrs_0: image0 descriptors
    :param descrs_1: image1 descriptors
    :return: tuple of descriptors' matches tuples
    """

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    match = flann.knnMatch(descrs_0, descrs_1, k=2)

    return match


def filter_matches(matches):
    """
    Filters descriptors' matches using distance
    :param matches: tuple of descriptors' matches tuples
    :return: good descriptors' matches
    """

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    return good


def get_keys_points(matches, keys_l, keys_r) -> np.array:
    """
    Get pixel points of matched keys
    :param matches: tuple of descriptors' matches
    :param keys_l: keys of left image
    :param keys_r: keys of right image
    :return: tuple of left and right keys' points arrays
    """

    pts_l = [keys_l[m.queryIdx].pt for m in matches]
    pts_r = [keys_r[m.trainIdx].pt for m in matches]  # may be before pts_l

    return np.float32(pts_l), np.float32(pts_r)


def get_fundamental(img_l: np.array, img_r: np.array, return_points: bool) -> np.array:
    """
    Using SIFT find matches, filter them, find fundamental matrix
    :param img_l: left image of the scene
    :param img_r: right image of the scene
    :param return_points: return best matched points
    :return: fundamental matrix
    """

    keys_l, descrs_l = get_sift(img_l)
    keys_r, descrs_r = get_sift(img_r)

    matches = match_knn(descrs_l, descrs_r)
    good_matches = filter_matches(matches)

    pts_l, pts_r = get_keys_points(good_matches, keys_l, keys_r)

    fundamental, mask = cv.findFundamentalMat(pts_l, pts_r, cv.FM_LMEDS)

    if return_points:
        mask = mask.ravel().astype('bool')
        pts_l, pts_r = pts_l[mask], pts_r[mask]
        # best_matches = np.array(good_matches)[mask]
        return fundamental, pts_l, pts_r

    return fundamental


# ---------------------------------


def imshow(image, save=''):
    """Show image"""

    plt.imshow(image)
    plt.show()

    if save:
        cv.imwrite(f'{save}', image)


def draw_lines_and_points(img, lines, pts):
    """
    Draw epilenes ans points on the image
    :param img: image where epilines are drawn
    :param lines: epilines
    :param pts: points on the image
    :return: image with epilines and points
    """

    c = img.shape[1]
    img_new = img.copy()

    np.random.seed(0)
    for l, pt in zip(lines, pts):
        color = np.random.randint(0, 255, 3).tolist()
        dot0 = np.int32([0, -l[2] / l[1]])
        dot1 = np.int32([c, -(l[2] + l[0] * c) / l[1]])

        img_new = cv.line(img_new, dot0, dot1, color, 1)
        img_new = cv.circle(img_new, np.int32(pt), 5, color, -1)
    return img_new


def draw_epilines(img_l, img_r, fundamental, pts_l, pts_r, show=False) -> tuple:
    """
    Draw epilines on images using fundamental matrix
    :param img_l: left image
    :param img_r: right image
    :param fundamental: fundamental matrix
    :param pts_l: points on the left image
    :param pts_r: points on the right image
    :param show: show images with lines or not
    :return: two images with epilines
    """

    lines_l = cv.computeCorrespondEpilines(pts_r, 2, fundamental).reshape(-1, 3)
    img_l_lines = draw_lines_and_points(img_l, lines_l, pts_l)

    lines_r = cv.computeCorrespondEpilines(pts_l, 1, fundamental).reshape(-1, 3)
    img_r_lines = draw_lines_and_points(img_r, lines_r, pts_r)

    if show:
        imshow(img_l_lines)
        imshow(img_r_lines)

    return img_l_lines, img_r_lines


# ----------------------------


def opencv_rectification(img_l, img_r, fundamental, pts_l, pts_r, show=False) -> tuple:
    """
    Rectifies left and right images using Hartley method in opencv
    :param img_l: left image
    :param img_r: right image
    :param fundamental: fundamental matrix
    :param pts_l: points on the left image
    :param pts_r: points on the right image
    :param show: show rectified images or not
    :return: two matrices for rectification
    """

    h1, w1, _ = img_l.shape
    h2, w2, _ = img_r.shape
    _, H1, H2 = cv.stereoRectifyUncalibrated(pts_l, pts_r, fundamental, imgSize=(w1, h1))

    img_l_rect = cv.warpPerspective(img_l, H1, (w1, h1))
    img_r_rect = cv.warpPerspective(img_r, H2, (w2, h2))

    if show:
        imshow(img_l_rect)  # , 'data/mess/left_opencv.png')
        imshow(img_r_rect)  # , 'data/mess/right_opencv.png')

    return H1, H2


# ----------------------------


def epipole_point(fundamental: np.array, image_position: str, normalize: str = 'Z') -> np.array:
    """
    Returns epipole from fundamental using svd
    :param fundamental: fundamental matrix
    :param image_position: ['left', 'right']
    :param normalize: ['No', 'Euclid', 'Z']
    :return: one of the epipoles
    """

    if image_position == 'right':
        matrix = fundamental.T
    else:
        matrix = fundamental  # image_position == 'left'

    epi = null_space(matrix)

    if normalize == 'Z':
        epi /= epi[2][0]
    elif normalize == 'Euclid':
        epi /= np.sqrt(np.sum(epi ** 2))
    # else 'No' normalization

    return epi.ravel()


def get_cross_matrix(vector: np.array) -> np.array:
    """
    Makes cross matrix from a vector
    :param vector: array/vector
    :return: cross matrix
    """

    matrix = np.array([[0, -vector[2], vector[1]],
                       [vector[2], 0, -vector[0]],
                       [-vector[1], vector[0], 0]])

    return matrix


def transform_right(epi: np.array, translation_coords: tuple = None) -> np.array:
    """
    Right image rectification matrix
    :param epi: epipole to the image
    :param translation_coords: translation shape if none no translation
    :return: right transformation matrix
    """

    if translation_coords:
        cx, cy = np.array(translation_coords) // 2
        T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
        epi = T @ epi
    # mirror = np.array([[-1, 0, cx*2], [0, -1, cy*2], [0, 0, 1]])

    d = np.sqrt(epi[0] ** 2 + epi[1] ** 2)
    R = np.array([[epi[0] / d, epi[1] / d, 0], [-epi[1] / d, epi[0] / d, 0], [0, 0, 1]])
    G = np.array([[1, 0, 0], [0, 1, 0], [-epi[2] / d, 0, 1]])
    R_right = G @ R

    if translation_coords:
        R_right = np.linalg.inv(T) @ R_right @ T

    return R_right


def minimize_abc(left, pts_l, right, pts_r) -> np.array:
    """
    Minimize least squares for x cood: left*pts_l @ result = right*pts_r
    :param left: Left points matrix
    :param pts_l: key points of the left image
    :param right: Right points matrix
    :param pts_r: key points of the right image
    :return: minimized transformation matrix
    """

    pts_l_3d = np.hstack([pts_l, np.ones((pts_l.shape[0], 1))])
    pts_r_3d = np.hstack([pts_r, np.ones((pts_r.shape[0], 1))])

    a = (left @ pts_l_3d.T)
    aT_norm = (a / a[2]).T
    b = (right @ pts_r_3d.T)
    bx_norm = b[0] / b[2]
    x = np.linalg.lstsq(aT_norm, bx_norm, rcond=None)[0]

    A = np.eye(3)
    A[0] = x

    return A


def transform_left(pts_l, pts_r, R_right, fundamental, epi) -> np.array:
    """
    Left image rectification matrix
    # !if translate was used on R_right, then R_left would be translated the same!
    :param pts_l: key points of the left image
    :param pts_r: key points of the right image
    :param R_right: right image rectification matrix
    :param fundamental: fundamental matrix
    :param epi: epipole to the image
    :return: left transformation matrix
    """

    epi_x = get_cross_matrix(epi)
    M = R_right @ epi_x @ fundamental

    M_orto = M.copy()
    cross_23 = np.cross(M[1], M[2])
    M_orto[0] = cross_23 / np.sqrt(cross_23.dot(cross_23))

    A = minimize_abc(M_orto, pts_l, R_right, pts_r)
    R_left = A @ M_orto

    return R_left


def custom_rectification(img_l, img_r, fundamental, pts_l, pts_r, show=False) -> tuple:
    """
    Rectifies left and right images using Hartley method in opencv
    :param img_l: left image
    :param img_r: right image
    :param fundamental: fundamental matrix
    :param pts_l: points on the left image
    :param pts_r: points on the right image
    :param show: show rectified images or not
    :return: two matrices for rectification
    """

    h1, w1, _ = img_l.shape
    h2, w2, _ = img_r.shape

    epipole = epipole_point(fundamental, 'right')
    Rr = transform_right(epipole, (w2, h2))  # translate is the same on Rl
    Rl = transform_left(pts_l, pts_r, Rr, fundamental, epipole)

    img_l_rect = cv.warpPerspective(img_l, Rl, (w1, h1))
    img_r_rect = cv.warpPerspective(img_r, Rr, (w2, h2))

    if show:
        imshow(img_l_rect)  # , 'data/mess/left_custom.png')
        imshow(img_r_rect)  # , 'data/mess/right_custom.png')

    return Rl, Rr


if __name__ == '__main__':
    path_l, path_r = 'data/mouse/right.png', 'data/mouse/left.png'
    image_l, image_r = cv.imread(path_l, cv.IMREAD_COLOR), cv.imread(path_r, cv.IMREAD_COLOR)

    F, points_l, points_r = get_fundamental(image_l, image_r, return_points=True)

    im_l, im_r = draw_epilines(image_l, image_r, F, points_l[:100], points_r[:100])

    L_cv, R_cv = opencv_rectification(im_l, im_r, F, points_l, points_r, True)
    L, R = custom_rectification(im_l, im_r, F, points_l, points_r, True)

    print(f"Rl mse error: {np.mean((L_cv / L_cv[2, 2] - L / L[2, 2]) ** 2)},",
          f"\nRr mse error: {np.mean((R_cv / R_cv[2, 2] - R / R[2, 2]) ** 2)}")
