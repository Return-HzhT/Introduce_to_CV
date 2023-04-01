import numpy as np
from utils import read_img, draw_corner
from HM1_Convolve import convolve, Sobel_filter_x, Sobel_filter_y, padding


def corner_response_function(input_img, window_size, alpha, threshold):
    """
        The function you need to implement for Q3.
        Inputs:
            input_img: array(float)
            window_size: int
            alpha: float
            threshold: float
        Outputs:
            corner_list: list
    """

    # please solve the corner_response_function of each window,
    # and keep windows with theta > threshold.
    # you can use several functions from HM1_Convolve to get
    # I_xx, I_yy, I_xy as well as the convolution result.
    # for detials of corner_response_function, please refer to the slides.
    w = np.ones((window_size, window_size))
    I_x = Sobel_filter_x(input_img)
    I_y = Sobel_filter_y(input_img)
    I_xx = I_x * I_x
    I_yy = I_y * I_y
    I_xy = I_x * I_y

    I_xx = padding(I_xx, window_size // 2, "zeroPadding")
    I_yy = padding(I_yy, window_size // 2, "zeroPadding")
    I_xy = padding(I_xy, window_size // 2, "zeroPadding")

    I_xx = convolve(I_xx, w)
    I_yy = convolve(I_yy, w)
    I_xy = convolve(I_xy, w)

    theta = (I_xx * I_yy - I_xy * I_xy) - alpha * (I_xx + I_yy) * (I_xx + I_yy)
    idx_x, idx_y = np.where(theta > threshold)
    tmp_tuple = np.column_stack((idx_x, idx_y, theta[idx_x, idx_y]))
    corner_list = np.array(tmp_tuple)

    return corner_list  # the corners in corne_list: a tuple of (index of rows, index of cols, theta)


if __name__ == "__main__":

    #Load the input images
    input_img = read_img("hand_writting.png") / 255.

    #you can adjust the parameters to fit your own implementation
    window_size = 3
    alpha = 0.04
    threshold = 10

    corner_list = corner_response_function(input_img, window_size, alpha,
                                           threshold)

    # NMS
    corner_list_sorted = sorted(corner_list, key=lambda x: x[2], reverse=True)
    NML_selected = []
    NML_selected.append(corner_list_sorted[0][:-1])
    dis = 10
    for i in corner_list_sorted:
        for j in NML_selected:
            if (abs(i[0] - j[0]) <= dis and abs(i[1] - j[1]) <= dis):
                break
        else:
            NML_selected.append(i[:-1])

    #save results
    draw_corner("hand_writting.png", "result/HM1_HarrisCorner.png",
                NML_selected)
