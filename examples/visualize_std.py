#!/usr/bin/env python3

"""
Simulate the online setting by iterating through a history, apply
standardization, plot the transformed history so far, i.e. what

    check(std_f(z))

sees in each iteration i=0...len(y)-1

    z := y[:(i+1)]
    std_f(z) = (z - std_avg(z)) / std(z)

The examples here show that the found convergence points are the same as we
progress, even though std(z) is a function of the iteration count. This works
because in all convergence check rules, such as last + tol > prev, which are
applied at each iteration, std(z) cancels. See also the related tests, where we
show that the method is invariant to an affine transform y' = y * s + c (when
using a properly scaled atol, of course). Real-world data of different
scales cannot usually be constructed by a simple transform, so atol values are
only approximately transferable to other histories.

The std="smad" case shows numerical pathologies in the noise-free case. Note
that in certain (constructed) cases, MAD and IQR can be zero (we use a small
EPS value in the code of course, but still). For instance:

    >>> std([1,1,1,1,3])
    0.8
    >>> scipy.stats.iqr([1,1,1,1,3])
    0.0
    >>> scipy.stats.median_abs_deviation([1,1,1,1,3])
    0.0

This may lead to numerical issues for certain histories. Therefore, we
recommend sticking to std="std" in production.
"""

from matplotlib import pyplot as plt, colormaps
import numpy as np

from nmbx.convergence import SlopeZero
from nmbx import utils


def get_fig_axs(nrows, ncols):
    return plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows),
        constrained_layout=True,
    )


def get_exp_noise_y():
    rng = np.random.default_rng(123)
    x = np.linspace(0, 10, 600)
    y = rng.normal(loc=0, scale=1, size=len(x))
    # make data slightly asymmetric
    msk = y >= 0
    y[msk] *= np.exp(-0.8 * x[msk])
    msk = y < 0
    y[msk] *= np.exp(-1.1 * x[msk])
    return y * 100 + 100


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 11, "legend.fontsize": 11})

    ##std_avg_lst = [np.mean]
    ##std_lst = ["std"]
    std_avg_lst = [np.mean, np.median]
    std_lst = ["std", "siqr", "smad"]

    cmap = colormaps.get_cmap("plasma")
    nrows = len(std_avg_lst)
    ncols = len(std_lst)

    cases = [
        dict(
            name="noise-free loss",
            y=utils.generate_history_data(noise_scale=0)[1],
            kwds=dict(
                atol=1e-2, wlen=10, wait=5, smooth_sigma=None, mode="min"
            ),
        ),
        dict(
            name="noisy loss with Gauss filter",
            y=utils.generate_history_data(noise_scale=0.01)[1],
            kwds=dict(atol=1e-2, wlen=10, wait=5, smooth_sigma=20, mode="min"),
        ),
        dict(
            name="damped noisy signal",
            y=get_exp_noise_y(),
            kwds=dict(
                atol=1e-2, wlen=20, wait=5, smooth_sigma=None, mode="abs"
            ),
        ),
        dict(
            name="damped noisy signal with Gauss filter",
            y=get_exp_noise_y(),
            kwds=dict(atol=1e-2, wlen=10, wait=5, smooth_sigma=10, mode="abs"),
        ),
    ]

    for case_dct in cases:
        y = case_dct["y"]
        kwds = case_dct["kwds"]
        step = len(y) // 10
        assert len(y) % step == 0
        ii_stop_range = range(step, len(y) + step, step)
        colors = cmap(np.linspace(0, 0.8, len(ii_stop_range)))

        fig, axs = get_fig_axs(nrows=nrows, ncols=ncols)
        axs = np.atleast_2d(axs)
        for irow, std_avg in enumerate(std_avg_lst):
            for icol, std in enumerate(std_lst):
                conv = SlopeZero(std=std, std_avg=std_avg, **kwds)
                ax = axs[irow, icol]
                ax.plot(y, ".", alpha=0.2)
                ax2 = ax.twinx()
                for icurve, (ii_stop, cc) in enumerate(
                    zip(ii_stop_range, colors)
                ):
                    y_current = y[:ii_stop]
                    y_plot = conv.preprocess_history(y_current)
                    ax2.plot(
                        y_plot,
                        alpha=0.5,
                        label=f"n={ii_stop}",
                        color=cc,
                    )
                    idx = conv.check_first(y_current)
                    if idx is not None:
                        idx -= kwds.get("wait", 1) - 1
                        ax2.plot(idx, y_plot[idx], "o", ms=10, color=cc)

                ax.set_title(f"std_avg={std_avg.__name__} {std=}")

                if irow == 0 and icol == 0:
                    ax2.legend(loc="upper right")
                if irow == (nrows - 1):
                    ax.set_xlabel("iteration")
                if icol == (ncols - 1):
                    ax2.set_ylabel("(y - std_avg(y)) / std(y)")
        fig.suptitle(case_dct["name"])
    plt.show()
