#!/usr/bin/env python3

import copy
import textwrap

import numpy as np
from matplotlib import pyplot as plt

from nmbx.convergence import (
    SlopeRise,
    SlopeZeroAbs,
    SlopeZeroMin,
    smooth_gauss,
)
from nmbx.utils import generate_history_data

plt.rcParams.update({"font.size": 11, "legend.fontsize": 11})

const = {
    "zero_abs": dict(wlen=15, wait=5, std="std", delay=0, smooth_sigma=None),
    "zero_min": dict(wlen=15, wait=5, std="std", delay=0, smooth_sigma=None),
    "rise": dict(wlen=15, wait=5, std="std", delay=0, smooth_sigma=None),
}

vary = dict(
    atol=[0.01, 0.05, 0.1],
    wlen=[1, 15, 30],
    wait=[1, 5, 10],
    delay=[0, 200],
    std=["std", "smad", "siqr"],
    smooth_sigma=[None, 10, 30],
    ##wlen_avg=[np.mean, np.median],
    ##std_avg=[np.mean, np.median],
)

method_map = dict(
    zero_abs=SlopeZeroAbs,
    zero_min=SlopeZeroMin,
    rise=SlopeRise,
)

legend_map = dict(
    zero_min="SlopeZero(mode='min')",
    zero_abs="SlopeZero(mode='abs')",
    rise="SlopeRise",
)
nrows = len(const)
ncols = len(vary)

for name, noise, atol in [("no_noise", 0, 0.05), ("noise", 0.03, 0.05)]:
    x, y, func = generate_history_data(noise)
    ymin = y.min()
    ymax = y.max()
    yspan = ymax - ymin
    ylo = ymin - 0.1 * yspan
    y_offset = 0.05 * yspan
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 4 * nrows),
        tight_layout=True,
    )
    fig.suptitle(f"{noise=}")

    for irow, (method_name, const_dct) in enumerate(const.items()):
        for icol, (vary_key, vary_vals) in enumerate(vary.items()):
            ax = axs[irow, icol]
            ax.plot(y, ".", alpha=0.2)
            for i_vary, vary_val in enumerate(vary_vals):
                kwds = copy.copy(const_dct)
                # In case we want to use a different atol for no_noise and
                # noise, we need to set it here.
                kwds["atol"] = atol
                kwds[vary_key] = vary_val

                res = method_map[method_name](**kwds).check_all(y)

                title = " ".join(
                    f"{k}={v}" for k, v in kwds.items() if k != vary_key
                )
                x_detect = x[res]
                x_detect_plot = np.arange(len(x))[res]
                y_detect = func(x_detect) + y_offset * (i_vary + 1)
                (line,) = ax.plot(
                    x_detect_plot,
                    y_detect,
                    ".",
                    label=f"{vary_key}={kwds[vary_key]}",
                )
                ax.set_ylim(bottom=ylo)
                if irow == 0:
                    ax.set_title("\n".join(textwrap.wrap(title, width=20)))
                if irow == nrows - 1:
                    ax.set_xlabel("iteration")
                if len(x_detect) > 0:
                    ax.vlines(
                        x_detect_plot[0],
                        ylo,
                        y_detect[0],
                        colors=line.get_color(),
                        linestyles="--",
                    )
                if kwds["smooth_sigma"] is not None:
                    ax.plot(
                        smooth_gauss(y, kwds["smooth_sigma"]),
                        ls="-",
                        color=line.get_color(),
                    )
            ax.legend(title=legend_map[method_name], loc="upper right")

    fig.savefig(f"conv_{name}.png")

plt.show()
