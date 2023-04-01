import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y
from utils import read_img, write_img


def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float) 
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """

    magnitude_grad = np.sqrt(x_grad * x_grad + y_grad * y_grad)
    direction_grad = np.arctan2(y_grad, x_grad)
    return magnitude_grad, direction_grad


def non_maximal_suppressor(grad_mag, grad_dir):
    """
        The function you need to implement for Q2 b).
        Inputs:
            grad_mag: array(float) 
            grad_dir: array(float)
        Outputs:
            output: array(float)
    """
    len_x, len_y = grad_mag.shape
    target_x = []
    target_y = []

    grad_dir = grad_dir * 180 / np.pi
    # case1：比较上下
    tmp_x, tmp_y = np.where(((67.5 < grad_dir) & (grad_dir <= 112.5))
                            | ((-112.5 < grad_dir) & (grad_dir <= -67.5)))
    grad_tmp = grad_mag[tmp_x, tmp_y]

    neighbor_x_1 = tmp_x - 1
    neighbor_x_1 = np.where(neighbor_x_1 < 0, 0, neighbor_x_1)
    neighbor_y_1 = tmp_y
    grad_neighbor_1 = grad_mag[neighbor_x_1, neighbor_y_1]

    neighbor_x_2 = tmp_x + 1
    neighbor_x_2 = np.where(neighbor_x_2 >= len_x, len_x - 1, neighbor_x_2)
    neighbor_y_2 = tmp_y
    grad_neighbor_2 = grad_mag[neighbor_x_2, neighbor_y_2]

    tmp_idx = np.where((grad_tmp > grad_neighbor_1)
                       & (grad_tmp > grad_neighbor_2))

    target_x = target_x + list(tmp_x[tmp_idx])
    target_y = target_y + list(tmp_y[tmp_idx])

    # case2：比较左下右上
    tmp_x, tmp_y = np.where(((112.5 < grad_dir) & (grad_dir <= 157.5))
                            | ((-67.5 < grad_dir) & (grad_dir <= -22.5)))
    grad_tmp = grad_mag[tmp_x, tmp_y]

    neighbor_x_1 = tmp_x + 1
    neighbor_x_1 = np.where(neighbor_x_1 >= len_x, len_x - 1, neighbor_x_1)
    neighbor_y_1 = tmp_y - 1
    neighbor_y_1 = np.where(neighbor_y_1 < 0, 0, neighbor_y_1)
    grad_neighbor_1 = grad_mag[neighbor_x_1, neighbor_y_1]

    neighbor_x_2 = tmp_x - 1
    neighbor_x_2 = np.where(neighbor_x_2 < 0, 0, neighbor_x_2)
    neighbor_y_2 = tmp_y + 1
    neighbor_y_2 = np.where(neighbor_y_2 >= len_y, len_y - 1, neighbor_y_2)
    grad_neighbor_2 = grad_mag[neighbor_x_2, neighbor_y_2]

    tmp_idx = np.where((grad_tmp > grad_neighbor_1)
                       & (grad_tmp > grad_neighbor_2))

    target_x = target_x + list(tmp_x[tmp_idx])
    target_y = target_y + list(tmp_y[tmp_idx])

    # case3：比较左右
    tmp_x, tmp_y = np.where(((-22.5 < grad_dir) & (grad_dir <= 22.5))
                            | ((157.5 < grad_dir) | (grad_dir <= -157.5)))
    grad_tmp = grad_mag[tmp_x, tmp_y]

    neighbor_x_1 = tmp_x
    neighbor_y_1 = tmp_y + 1
    neighbor_y_1 = np.where(neighbor_y_1 >= len_y, len_y - 1, neighbor_y_1)
    grad_neighbor_1 = grad_mag[neighbor_x_1, neighbor_y_1]

    neighbor_x_2 = tmp_x
    neighbor_y_2 = tmp_y - 1
    neighbor_y_2 = np.where(neighbor_y_2 < 0, 0, neighbor_y_2)
    grad_neighbor_2 = grad_mag[neighbor_x_2, neighbor_y_2]

    tmp_idx = np.where((grad_tmp > grad_neighbor_1)
                       & (grad_tmp > grad_neighbor_2))

    target_x = target_x + list(tmp_x[tmp_idx])
    target_y = target_y + list(tmp_y[tmp_idx])

    # case4：比较左上右下
    tmp_x, tmp_y = np.where(((22.5 < grad_dir) & (grad_dir <= 67.5))
                            | ((-157.5 < grad_dir) & (grad_dir <= -112.5)))
    grad_tmp = grad_mag[tmp_x, tmp_y]

    neighbor_x_1 = tmp_x + 1
    neighbor_x_1 = np.where(neighbor_x_1 >= len_x, len_x - 1, neighbor_x_1)
    neighbor_y_1 = tmp_y + 1
    neighbor_y_1 = np.where(neighbor_y_1 >= len_y, len_y - 1, neighbor_y_1)
    grad_neighbor_1 = grad_mag[neighbor_x_1, neighbor_y_1]

    neighbor_x_2 = tmp_x - 1
    neighbor_x_2 = np.where(neighbor_x_2 < 0, 0, neighbor_x_2)
    neighbor_y_2 = tmp_y - 1
    neighbor_y_2 = np.where(neighbor_y_2 < 0, 0, neighbor_y_2)
    grad_neighbor_2 = grad_mag[neighbor_x_2, neighbor_y_2]

    tmp_idx = np.where((grad_tmp > grad_neighbor_1)
                       & (grad_tmp > grad_neighbor_2))

    target_x = target_x + list(tmp_x[tmp_idx])
    target_y = target_y + list(tmp_y[tmp_idx])

    NMS_output = np.zeros_like(grad_mag)
    NMS_output[target_x, target_y] = grad_mag[target_x, target_y]

    return NMS_output


def hysteresis_thresholding(img):
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float) 
        Outputs:
            output: array(float)
    """

    #you can adjust the parameters to fit your own implementation
    low_threshold = 0.1
    high_threshold = 0.3

    output = np.zeros_like(img)
    x, y = img.shape

    for i in range(x):
        for j in range(y):
            if img[i][j] > high_threshold:
                output[i][j] = 1

    while True:
        flag = False
        for i in range(1, x - 1):
            for j in range(1, y - 1):
                if output[i][j] == 1:
                    for k in range(i - 1, i + 2):
                        for l in range(j - 1, j + 2):
                            if output[k][l] == 0 and img[k][l] > low_threshold:
                                output[k][l] = 1
                                flag = True
        if flag == False:
            break
    return output


if __name__ == "__main__":

    #Load the input images
    input_img = read_img("lenna.png") / 255

    #Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)

    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)

    #Compute the magnitude and the direction of gradient
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(
        x_grad, y_grad)

    #NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)

    #Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output)

    write_img("result/HM1_Canny_result.png", output_img * 255)
