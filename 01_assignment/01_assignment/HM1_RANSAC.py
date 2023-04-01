import numpy as np
from utils import draw_save_plane_with_points
from math import log, floor

if __name__ == "__main__":

    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    noise_points = np.loadtxt("HM1_ransac_points.txt")

    # RANSAC
    # we recommend you to formulate the palnace function as:  A*x+B*y+C*z+D=0
    # more than 99.9% probability at least one hypothesis does not contain any outliers
    pos_outliers = 1 - ((100 / 130)**3)
    sample_time = floor(log(0.001, pos_outliers)) + 1
    distance_threshold = 0.05

    # sample points group
    sample_points = np.random.randint(130, size=(sample_time, 3))

    # estimate the plane with sampled points group
    sample_planes = np.zeros((sample_time, 4))
    x1 = noise_points[sample_points[:, 0], 0]
    y1 = noise_points[sample_points[:, 0], 1]
    z1 = noise_points[sample_points[:, 0], 2]
    x2 = noise_points[sample_points[:, 1], 0]
    y2 = noise_points[sample_points[:, 1], 1]
    z2 = noise_points[sample_points[:, 1], 2]
    x3 = noise_points[sample_points[:, 2], 0]
    y3 = noise_points[sample_points[:, 2], 1]
    z3 = noise_points[sample_points[:, 2], 2]

    a = (y3 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
    b = (x3 - x1) * (z2 - z1) - (x2 - x1) * (z3 - z1)
    c = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    d = -(a * x1 + b * y1 + c * z1)

    tmp_a = np.tile(a, 130).reshape(130, sample_time).T
    tmp_b = np.tile(b, 130).reshape(130, sample_time).T
    tmp_c = np.tile(c, 130).reshape(130, sample_time).T
    tmp_d = np.tile(d, 130).reshape(130, sample_time).T

    # evaluate inliers (with point-to-plance distance < distance_threshold)
    x = noise_points[:, 0]
    y = noise_points[:, 1]
    z = noise_points[:, 2]
    dis = np.abs(tmp_a * x + tmp_b * y + tmp_c * z +
                 tmp_d) / np.sqrt(tmp_a * tmp_a + tmp_b * tmp_b +
                                  tmp_c * tmp_c)
    inliers_of_planes = (dis < distance_threshold)
    inliers_count = inliers_of_planes.sum(axis=1)
    plane_idx = np.argsort(inliers_count)[-1]

    # minimize the sum of squared perpendicular distances of all inliers with least-squared method
    target_inliers = np.where(dis[plane_idx] < distance_threshold)[0]
    target_points = noise_points[target_inliers]

    x = target_points[:, 0]
    y = target_points[:, 1]
    z = target_points[:, 2]
    ones = np.ones_like(x)

    A = np.stack([x, y, z, ones]).T
    U, D, V_T = np.linalg.svd(A, full_matrices=1)
    pf = V_T[-1]

    # draw the estimated plane with points and save the results
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0
    draw_save_plane_with_points(pf, noise_points, "result/HM1_RANSAC_fig.png")
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)
