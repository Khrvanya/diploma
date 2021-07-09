import cv2 as cv
import numpy as np
from scipy.ndimage import rotate
from scipy.fft import rfft2, irfft2, next_fast_len
# from pyfftw.builders import rfftn, irfftn

from utils import float_to_int_image, int_to_float_image
from utils import imshow, get_sliced_image, get_padded_image
from utils import norm_image


def preprocess_image(image, verbose=0):
    """
    Make image gray, process it, transform to float, add color axis
    :param image: np.array
    :param verbose: verbose
    :return: processed gray image of float in (0,1) range
    """

    gray_img = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    image = int_to_float_image(gray_img)
    image = image[..., np.newaxis]

    if verbose:
        imshow(image)

    return image


def preprocess_kernel(kernel, verbose=0):
    """
    Make image gray, process it, transform to float, add color axis
    :param kernel: np.array
    :param verbose: verbose
    :return: processed gray kernel of float in (0,1) range
    """

    gray_krnl = cv.cvtColor(kernel, cv.COLOR_RGB2GRAY)

    gauss = cv.GaussianBlur(gray_krnl, (1, 1), 0)
    erode_krnl = cv.erode(gauss, np.ones((2, 2), np.int32))

    kernel = int_to_float_image(erode_krnl)
    kernel = kernel[..., np.newaxis]

    if verbose:
        imshow(kernel)

    return kernel


def postprocess_images(images):
    """
    Only last three shapes are considered as images
    Scaling images, then float_to_int_image
    """

    scaled_image = norm_image(images)
    final_images = float_to_int_image(scaled_image)

    return final_images


def construct_image_vector(image):
    """
    Makes the image vector for convolution, that changes the loss function
    :param image: last shape is the color
    :return: the vector with image for convolution
    """

    image_square = np.sum(image ** 2, axis=-1)  # sums the color channel, we need one value
    image_square = image_square[..., np.newaxis]  # for concatenation
    image_vect = np.concatenate([image_square, image], axis=-1)

    return image_vect


def construct_kernel_vector(kernel):
    """
    Makes the kernel vector for convolution, that changes the loss function
    kernel must have -1 values if you want to use the padding mask
    :param kernel: np.array
    :return: the vector with kernel for convolution
    """

    kernel = kernel[..., ::-1, ::-1, :]  # invert the kernel

    addition_to_color = ~(kernel.mean(axis=-1) == -1) * 1     # -1 is special number to detect the padding mask
    addition_to_color = addition_to_color[..., np.newaxis]  # for concatenation
    kernel[kernel < 0] = 0                                  # to make them back to 0
    kernel_vect = np.concatenate([addition_to_color, -2 * kernel], axis=-1)

    return kernel_vect


def rotation_images(image, number: int, pad_const=-1):
    """
    Rotates the image and concatenates all the rotations
    :param image: image which we rotate
    :param number: number of rotated images, new shape of output
    :param pad_const: to make padding with constant
    :return: rotated images concatenated, adds one shape (number, image.shape)
    """

    padding_shape = tuple([int(np.sqrt(np.sum(np.array(image.shape) ** 2))) + 1] * 2)
    images = np.zeros((number, *padding_shape, image.shape[-1]))
    rotation_angle = 360 / number

    angle = 0
    for idx in range(number):
        rotated_img = rotate(image, angle, cval=pad_const)
        angle += rotation_angle

        padded_img = get_padded_image(rotated_img, padding_shape, pad_const)
        images[idx] = padded_img

    return images


def convolve_images(image, kernel, mode='valid', method='fft'):
    """
    Convolves image and kernel
    Only last three axes are considered as an image,
    last axis is a vector (with color channels)
    :param image: images where we detect the kernel
    :param kernel: images which we detect in the image
    :param mode: mode of convolution "valid" or "full" (just for fft)
    :param method: method of convolution "direct" or "fft"
    :return: convolutions of the image and kernel
    """

    ish, ksh = image.shape, kernel.shape
    shape = (*tuple(ksh[:-3]), *tuple([ish[i] + ksh[i] - 1 for i in [-3, -2]]))
    conv_shape = (*tuple(ksh[:-3]), *tuple([ish[i] - ksh[i] + 1 for i in [-3, -2]]))
    # print(f"shape: {shape}\nconv_shape: {conv_shape}\n")

    if method == "direct":
        if mode == 'valid':
            convolutions = np.zeros(conv_shape)
        else:
            assert False, "!!!No such mode available!!!"

        for y in range(image.shape[-2]):
            if y > image.shape[-2] - kernel.shape[-2]:
                break

            for x in range(image.shape[-3]):
                if x > image.shape[-3] - kernel.shape[-3]:
                    break

                curr_image = image[..., x: x + kernel.shape[-3], y: y + kernel.shape[-2], :]
                convolutions[..., x, y] = (curr_image * kernel).sum(axis=-2).sum(axis=-1).sum(axis=0)

    elif method == "fft":
        full_shape = (*shape[:-2], *tuple([next_fast_len(shape[i], True) for i in [-2, -1]]))
        convolutions = np.zeros(full_shape)

        for idx in range(ish[-1]):  # image.shape[-1] == kernel.shape[-1]
            fft_image = rfft2(image[..., idx], full_shape[-2:])[np.newaxis, ...]
            fft_kernel = rfft2(kernel[..., idx], full_shape[-2:])
            conv = irfft2(fft_image * fft_kernel, full_shape[-2:])
            convolutions += conv

        conv_slice = tuple([slice(sz) for sz in shape])  # to get back to shape
        convolutions = convolutions[conv_slice]  # needed full_shape for speed

        if mode == 'valid':
            convolutions = get_sliced_image(convolutions, conv_shape)

    else:
        assert False, "!!!No such method for convolution!!!"

    return convolutions[..., np.newaxis]     # add color channel


def convolve_mse_images(image, kernel, mode='valid', method='fft'):
    """
    Convolves image and kernel using mse error function
    All info is in convolve_images function
    """

    image_vector = construct_image_vector(image)
    kernel_vector = construct_kernel_vector(kernel)

    raw_convolutions = convolve_images(image_vector, kernel_vector, mode, method)

    return raw_convolutions


def convolve_mse_rotate_images(image, kernel, kernels_number, mode, method):
    """
    Convolves image and kernels using mse error function
    kernels are created using rotation of origin kernel
    We rotate kernel on certain angle and get kernels_number of kernels
    All other parameters are described in convolve_images
    """

    stave = preprocess_image(image)
    clef = preprocess_kernel(kernel)

    clefs = rotation_images(clef, kernels_number, pad_const=-1)  # padding_mask in construct_kernel_vector

    convs_raw = convolve_mse_images(stave, clefs, mode, method)
    convs = postprocess_images(convs_raw)

    return convs
