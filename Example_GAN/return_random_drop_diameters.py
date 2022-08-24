import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


def return_random_drop_diameters(d_32_hist, N_drops):

    # Calculating CDF
    N_points = d_32_hist.shape[0]
    cdf_x = np.linspace(0.0,1.0,num=N_points, endpoint=True)
    cdf_y=[0.0]
    for k in range(1,N_points):
        cdf_y.append(np.sum(d_32_hist[:k,1]))
    cdf_y = np.array(cdf_y)
    cdf_y /= np.nanmax(cdf_y)

    # Approximating it by a Cubic Spline
    cdf_spline = CubicSpline(d_32_hist[:,0], cdf_y)
    cdf_x_sp = np.linspace(np.min(d_32_hist[:,0]), np.max(d_32_hist[:,0]), 1000)
    cdf_y_sp = cdf_spline(cdf_x_sp)

    # Generating random uniform [0.0,1.0] distribution
    N_RANDOM = N_drops
    points = np.random.uniform(low=0.01, high=1.0, size=N_RANDOM)

    # Sampling from the Cubic Spline through the random uniform distribution
    diams = np.zeros_like(points)
    for k in range(len(points)):
        pnt = points[k]
        arg_x = np.argmin(np.abs(pnt - cdf_y_sp))

        arg_1 = max(0           , arg_x - 1)
        arg_2 = min(N_points - 1, arg_x + 1)

        Y_1 = cdf_y_sp[arg_1]
        Y_2 = cdf_y_sp[arg_2]
        d_1 = cdf_x_sp[arg_1]
        d_2 = cdf_x_sp[arg_2]

        dx = ((pnt - Y_1) / (Y_2 - Y_1)) * (d_2 - d_1) + d_1
        diams[k] = dx

    # Sorting the diameters by size
    diams = np.sort(diams)

    return diams
