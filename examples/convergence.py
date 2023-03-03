#!/usr/bin/env python3


import copy

import numpy as np
from matplotlib import pyplot as plt


from nmbx.convergence import SlopeRise, SlopeZero


def func(x, y0=1.0):
    if x <= 0:
        return y0 + x**2
    elif x <= 0.5:
        return y0
    else:
        return y0 + (x - 0.5) ** 2.0 / 10.0


func = np.frompyfunc(func, 1, 1)

const = {
    "zero": dict(wlen=15, tol=0.01, wait=None, reduction=np.mean),
    "rise": dict(wlen=15, tol=0.01, wait=None, reduction=np.mean),
}

vary = dict(wlen=[1, 15, 50], tol=[0.001, 0.01, 0.05, 0.1], wait=[None, 5, 10])

method_map = dict(zero=SlopeZero, rise=SlopeRise)


dx = 0.01
y_offset = 0.05
x = np.arange(-1, 1.5, dx)

nrows = len(const)
ncols = len(vary)

rng = np.random.default_rng(seed=123)

for name, noise in [("no_noise", 0), ("noise", 0.05)]:
    y = func(x) + rng.normal(scale=noise, size=len(x))
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 4 * nrows),
        tight_layout=True,
    )
    fig.suptitle(f"{noise=}")

    for irow, (method_name, const_dct) in enumerate(const.items()):
        ##ic(irow, method_name, const_dct)
        for icol, (vary_key, vary_vals) in enumerate(vary.items()):
            ax = axs[irow, icol]
            ax.plot(y, ".", alpha=0.2)
            ax.set_title(f"{method_map[method_name].__name__} vary={vary_key}")
            for i_vary, vary_val in enumerate(vary_vals):
                kwds = copy.copy(const_dct)
                kwds[vary_key] = vary_val
                ##ic(kwds)
                res = np.zeros_like(x).astype(bool)
                conv = method_map[method_name](**kwds)

                for ii in range(1, len(x)):
                    y_cur = y[:ii]
                    res[ii] = conv.check(y_cur)

                label = " ".join(
                    f"{k}={v}" for k, v in kwds.items() if k != "reduction"
                )
                x_detect = x[res]
                x_detect_plot = np.arange(len(x))[res]
                y_detect = func(x_detect) + y_offset * (i_vary + 1)
                (line,) = ax.plot(
                    x_detect_plot,
                    y_detect,
                    ".",
                    label=label,
                )
                ylo = 0.8
                ax.set_ylim(bottom=ylo)
                if len(x_detect) > 0:
                    ax.vlines(
                        x_detect_plot[0],
                        ylo,
                        y_detect[0],
                        colors=line.get_color(),
                        linestyles="--",
                    )

    for ax in axs.flat:
        ax.legend()
        ax.set_xlabel("iteration")

    ##fig.savefig(f"conv_{name}.png")

plt.show()
