#!/usr/bin/env python3

"""
Generate noisy data, smooth with a moving average of size wlen. Then optimize
smooth_sigma of a Gaussian filter to be close to the moving average result
(which is still more noisy). Result: use smooth_sigma = wlen/3.
"""

from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import convolve, gaussian_filter
from scipy.optimize import minimize


def func(sigma, *, y, y_target, mode):
    d = y_target - gaussian_filter(y, sigma[0], mode=mode)

    # What gaussian_filter() basically does, just as an exercise.
    ##kern = scipy.signal.windows.gaussian(int(6*sigma[0]), sigma[0])
    ##d = y_target - convolve(y, kern/kern.sum(), mode=mode)
    return np.dot(d, d)


if __name__ == "__main__":
    rng = np.random.default_rng(123)

    y = rng.normal(size=1000)
    mode = "nearest"

    wlen = 30
    kern = np.ones(wlen)
    y_mov_avg = convolve(y, kern / kern.sum(), mode=mode)

    opt = minimize(partial(func, y=y, y_target=y_mov_avg, mode=mode), x0=10)
    ##print(opt)
    sigma_opt = opt.x[0]
    print(f"{wlen=} {sigma_opt=}")

    fig, ax = plt.subplots()
    ax.plot(y, ".", alpha=0.2)
    ax.plot(y_mov_avg, label="moving average")
    ax.plot(gaussian_filter(y, sigma_opt, mode=mode), label="gauss")
    ax.set_ylim(-1, 1)
    ax.legend()

    plt.show()
