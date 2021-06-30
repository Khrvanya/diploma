import time
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve
from scipy.fft import rfftn, irfftn
from scipy.fft import next_fast_len
# from pyfftw.builders import rfftn, irfftn

np.random.seed(47)


def log_image(image):
    """Log image array"""

    image[image < 0] = 0  # if some rotated images have smaller error values
                                         # than the original kernel
    return np.log(image + 1)


def float_to_int_image(image):
    """Transform image to float"""

    image[image > 1] = 1   # for very white pixels to become white
    image[image < 0] = 0   # shouldn't be used, but however

    return (image * 255).astype(np.uint8)


def int_to_float_image(image):
    """Transform image to int"""

    image[image > 255] = 255  # for very white pixels to become white
    image[image < 0] = 0      # shouldn't be used, but however

    return image.astype(np.float32) / 255


def postprocess_images(images):
    """
    Scaling using only first image
    First add image[0].min to all, then take log from all
    Finally divide on image[0].max all
    Then use float_to_int_image (if image > 1 them image = 1)
    """

    images = [image - images[0].min() for image in images]

    log_images = [log_image(image) for image in images]
    normed_images = [image / log_images[0].max() for image in log_images]

    final_images = [float_to_int_image(image) for image in normed_images]

    return final_images


def imshow(image, save=''):
    """Show image"""

    plt.imshow(image, 'gray')
    plt.show()

    if save:
        cv.imwrite(f'data/{save}', image)


def get_centered_image(image, new_shape: tuple):
    """
    Returns new centered sliced image
    :param image: original image
    :param new_shape: shape of the returned image
    :return: centered image with new_shape
    """

    new_shape, curr_shape = np.asarray(new_shape), np.array(image.shape)
    start_idx = (curr_shape - new_shape) // 2
    end_idx = start_idx + new_shape
    img_slice = [slice(start_idx[k], end_idx[k]) for k in range(len(end_idx))]

    return image[tuple(img_slice)]


def preprocess_image(image_path, verbose=0):
    """
    Read image, make it gray, process it, transform to float
    :param image_path: path to the image that will be preprocessed
    :param verbose: verbose
    :return: processed gray image of float in (0,1) range
    """

    img = cv.imread(image_path, cv.IMREAD_COLOR)
    gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # gauss = cv.GaussianBlur(gray_img, (1, 1), 0)
    # _, thresh = cv.threshold(gauss, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    image = int_to_float_image(gray_img)

    if verbose:
        imshow(image)

    return image


def preprocess_kernel(kernel_path, verbose=0):
    """
    Read kernel, make it gray, process it, transform to float
    :param kernel_path: path to the kernel that will be preprocessed
    :param verbose: verbose
    :return: processed gray kernel of float in (0,1) range
    """

    krnl = cv.imread(kernel_path, cv.IMREAD_COLOR)
    gray_krnl = cv.cvtColor(krnl, cv.COLOR_RGB2GRAY)

    gauss = cv.GaussianBlur(gray_krnl, (1, 1), 0)
    # _, thresh = cv.threshold(gauss, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    erode_krnl = cv.erode(gauss, np.ones((2, 2), np.int32))

    kernel = int_to_float_image(erode_krnl)

    if verbose:
        imshow(kernel)

    return kernel


def rescale_image(image, scale: tuple):
    """
    Squeezes or extends image to the scale shape
    :param image: gray image which we rescale
    :param scale: shape of the sub_image, which we need to get after squeezing or extending
    :return: rescaled image
    """

    return image


def rotate_image(image, angle: int):
    """
    Rotates the image to the right on angle
    :param image: gray image which we rotate
    :param angle: angle of right rotation
    :return: rotated image
    """

    return image


def construct_loss_func(image, kernel, method):
    """
    Changes the loss function in convolution
    :param image: main image
    :param kernel: convolutional kernel
    :param method: method of convolution "direct" or "fft"
    :return: Two vectors for convolution (vectors for fft method)
    """

    if method == 'direct':
        kernel = np.flipud(np.fliplr(kernel))
    kernel_vect = np.ones((2, *kernel.shape))
    kernel_vect[1, :, :] = -2 * kernel

    image_vect = np.ones((2, *image.shape))
    image_vect[0, :, :] = image ** 2
    image_vect[1, :, :] = image

    return image_vect, kernel_vect


def convolve_images(image, kernel, mode='full', method='direct'):
    """
    Convolves image and kernel using square differences sums
    only 'full' mode is available
    :param image: gray image where we detect the kernel
    :param kernel: gray image which we detect in the image
    :param mode: mode of convolution "valid" or "full" (just for fft)
    :param method: method of convolution "direct" or "fft"
    :return: gray image, where black color shows the overlap
    """

    image_vect, kernel_vect = construct_loss_func(image, kernel, method)

    shape = tuple([image.shape[i] + kernel.shape[i] - 1 for i in range(len(image.shape))])
    conv_shape = tuple([image.shape[i] - kernel.shape[i] + 1 for i in range(len(image.shape))])
    # print(f"shape: {shape}\nconv_shape: {conv_shape}\n")

    if method == "direct":
        if mode == 'valid':
            convolution = np.zeros(conv_shape)
        else:
            assert False, "!!!No such mode available!!!"

        for y in range(image_vect.shape[2]):
            if y > image_vect.shape[2] - kernel.shape[1]:
                break

            for x in range(image_vect.shape[1]):
                if x > image_vect.shape[1] - kernel.shape[0]:
                    break

                curr_image_vect = image_vect[:, x: x + kernel.shape[0], y: y + kernel.shape[1]]
                convolution[x, y] = (curr_image_vect * kernel_vect).sum()

    elif method == "fft":
        full_shape = tuple([next_fast_len(shape[i], True) for i in range(len(image.shape))])
        convolution = np.zeros(full_shape)

        for idx in range(image_vect.shape[0]):
            fft_image = rfftn(image_vect[idx, :, :], full_shape)
            fft_kernel = rfftn(kernel_vect[idx, :, :], full_shape)
            conv = irfftn(fft_image * fft_kernel, full_shape)
            convolution += conv

        conv_slice = tuple([slice(sz) for sz in shape])  # to get back to shape
        convolution = convolution[conv_slice]  # needed full_shape for speed

        if mode == 'valid':
            convolution = get_centered_image(convolution, conv_shape)

    else:
        assert False, "!!!No such method for convolution!!!"

    return convolution


def get_convolution(image, kernel, mode: str, method: str):  # , angle: int, scale: tuple):
    """
    Convolves image and kernel, after scaling and rotating (to the right) the kernel
    :param image: gray image where we detect the kernel
    :param kernel: gray image which we detect in the image
    :param angle: angle of rotation of the kernel to the right
    :param scale: shape of the kernel, which we need to get after squeezing or extending
    :param mode: mode of convolution "valid" or "full" (just for fft)
    :param method: method of convolution "direct" ot "fft"
    :return: gray images, where black color shows the overlap
    """

    # kernel = rescale_image(kernel, scale)
    # kernel = rotate_image(kernel, angle)

    raw_conv = convolve_images(preprocess_image('data/stave.jpg'), kernel, mode, method)
    raw_conv90 = convolve_images(preprocess_image('data/stave90.jpg'), kernel, mode, method)

    convolutions = postprocess_images([raw_conv, raw_conv90])

    return convolutions


if __name__ == '__main__':
    stave = preprocess_image('data/stave.jpg')
    clef = preprocess_kernel('data/clef.jpg')

    convs = get_convolution(stave.copy(), clef.copy(), 'valid', 'fft')
    imshow(convs[0], save='result_my_fft.jpg')
    imshow(convs[1], save='result_my_fft90.jpg')
