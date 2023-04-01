import numpy as np
from utils import read_img, write_img


def padding(img, padding_size, type):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """

    x, y = img.shape
    padding_img = np.zeros((x + 2 * padding_size, y + 2 * padding_size))
    padding_img[padding_size:x + padding_size,
                padding_size:y + padding_size] = img
    if type == "zeroPadding":
        return padding_img
    elif type == "replicatePadding":
        padding_img[0:padding_size, 0:padding_size] = img[0][0]
        padding_img[x + padding_size:x + 2 * padding_size,
                    0:padding_size] = img[x - 1][0]
        padding_img[0:padding_size,
                    y + padding_size:y + 2 * padding_size] = img[0][y - 1]
        padding_img[x + padding_size:x + 2 * padding_size,
                    y + padding_size:y + 2 * padding_size] = img[x - 1][y - 1]
        padding_img[0:padding_size, padding_size:y + padding_size] = img[0,
                                                                         0:y]
        padding_img[x + padding_size:x + 2 * padding_size,
                    padding_size:y + padding_size] = img[x - 1, 0:y]
        padding_img[padding_size:x + padding_size,
                    0:padding_size] = img[0:x, 0].reshape(x, 1)
        padding_img[padding_size:x + padding_size, y + padding_size:y +
                    2 * padding_size] = img[0:x, y - 1].reshape(x, 1)
        return padding_img


def convol_with_Toeplitz_matrix(img, kernel):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """

    #zero padding
    pad_img = padding(img, 1, "zeroPadding")
    #build the Toeplitz matrix and compute convolution
    kernel = kernel.reshape(9)

    tmp_arr1 = [kernel[0], kernel[1], kernel[2]]
    tmp_arr2 = [kernel[3], kernel[4], kernel[5]]
    tmp_arr3 = [kernel[6], kernel[7], kernel[8]]
    tmp_arr4 = [0] * 5
    tmp_arr5 = [0] * 46

    arr1 = tmp_arr1 + tmp_arr4 + tmp_arr2 + tmp_arr4 + tmp_arr3 + tmp_arr5
    arr2 = arr1 * 6 + [0] * 2
    arr3 = arr2 * 5 + arr1 * 5 + tmp_arr1 + tmp_arr4 + tmp_arr1 + tmp_arr4 + tmp_arr3

    toeplitz_mat = np.array(arr3).reshape(36, 64)
    img_arr = pad_img.reshape(64, 1)
    output = (toeplitz_mat @ img_arr).reshape(6, 6)
    return output


def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float)
        Outputs:
            output: array(float)
    """

    #build the sliding-window convolution here

    h = img.shape[0]
    w = img.shape[1]
    k = kernel.shape[0]

    i0 = np.repeat(np.arange(k), k)
    i1 = np.repeat(np.arange(h - k + 1), w - k + 1)
    j0 = np.tile(np.arange(k), k)
    j1 = np.tile(np.arange(w - k + 1), h - k + 1)
    idx_i = (i0.reshape(-1, 1) + i1.reshape(1, -1)).T.reshape(-1)
    idx_j = (j0.reshape(-1, 1) + j1.reshape(1, -1)).T.reshape(-1)

    window_mat = img[idx_i, idx_j].reshape((h - k + 1) * (w - k + 1), k * k)
    kernel = kernel.reshape(k * k, 1)
    output = window_mat @ kernel
    output = output.reshape((h - k + 1), (w - k + 1))

    return output


def Gaussian_filter(img):
    padding_img = padding(img, 1, "replicatePadding")
    gaussian_kernel = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8],
                                [1 / 16, 1 / 8, 1 / 16]])
    output = convolve(padding_img, gaussian_kernel)
    return output


def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output


def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output


if __name__ == "__main__":

    np.random.seed(111)
    input_array = np.random.rand(6, 6)
    input_kernel = np.random.rand(3, 3)

    # task1: padding
    zero_pad = padding(input_array, 1, "zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt", zero_pad)

    replicate_pad = padding(input_array, 1, "replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt", replicate_pad)

    #task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    #task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)

    #task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("lenna.png") / 255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x * 255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y * 255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur * 255)
