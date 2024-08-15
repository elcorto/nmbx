#!/usr/bin/env python3

"""
Use Optuna to find optimal parameters for SlopeRise and SlopeZero using
synthetic noisy history data.
"""

import copy
from functools import partial
from pprint import pprint

import optuna
from matplotlib import pyplot as plt
import numpy as np

from nmbx.convergence import SlopeRise, SlopeZero
from nmbx.utils import generate_history_data


def call_check_first(params, *, cls, const_kwds):
    kwds = copy.copy(const_kwds)
    kwds.update(params)
    conv = cls(**kwds)
    check_result = conv.check_first(y)
    out = (
        check_result - kwds.get("wait", 1) - 1
        if check_result is not None
        else None
    )
    return out, kwds, conv


def objective(trial, *, y_loss_fac, target, cls, const_kwds):
    # Do NOT use constant values here since they won't show up in
    # study.best_trial.params . Instead add them to const.
    params = dict(
        atol=trial.suggest_float("atol", 1e-3, 10, log=True),
        wlen=trial.suggest_int("wlen", 1, 20),
        wait=trial.suggest_int("wait", 1, 10),
        ##std=trial.suggest_categorical("std", ["std", "smad", "siqr"]),
        # Anything below sigma=1 is basically the same as not using the filter.
        # By that we avoid having a super long list of integer values:
        #   trail.suggest_categorical(..., [None] + np.unique(linspace(...).astype(int)))
        smooth_sigma=trial.suggest_float("smooth_sigma", 0.1, len(x) // 4),
        ##wlen_avg=trial.suggest_categorical("wlen_avg", [np.mean, np.median]),
        ##std_avg=trial.suggest_categorical("std_avg", [np.mean, np.median]),
        ##delay=trial.suggest_int("delay", 0, len(x) // 2),
    )

    idx, _, conv = call_check_first(params, cls=cls, const_kwds=const_kwds)

    # How to treat the case when idx is None (no conv detected = bad
    # params). We don't want to just return "some big number" as a hack.
    if idx is None:
        return 10

    # We are close to the change point.
    x_loss = abs(target - x[idx])

    if y_loss_fac > 0:
        # Smoothed y is close to noise-free ground truth. The idea here is to
        # discourage (i) large smooth_sigma which make the history basically
        # flat and detect a correct change point for the wrong reasons (by
        # adjusting the other params to absurd values) and (ii) discourage very
        # small smooth_sigma that basically disable the Gaussian filter.
        y_loss = np.mean(np.abs(y_no_noise - conv.smooth_f(y)))
    else:
        y_loss = 0.0

    # Relaxed loss. Instead of rewarding finding the "correct" change point, we
    # are OK with finding one nearby.
    target_delta_x = 0.2
    lo = target - target_delta_x
    hi = target + target_delta_x
    x_loss = 0 if lo <= x[idx] <= hi else x_loss

    print(f"{x_loss=} {y_loss=}")
    return x_loss + y_loss_fac * y_loss


if __name__ == "__main__":
    x_fall_stop = 1.5
    x_rise_start = 4
    x, y, data_func = generate_history_data(
        noise_scale=0.03,
        dx=0.01,
        x0=0,
        x1=6,
        x_fall_stop=x_fall_stop,
        x_rise_start=x_rise_start,
        y0=1.0,
    )

    y_no_noise = data_func(x)
    method_map = dict(zero=partial(SlopeZero, mode="min"), rise=SlopeRise)
    target_map = dict(zero=x_fall_stop - 0.3, rise=x_rise_start + 0.2)

    const = {
        "zero": dict(wlen=1, wait=1, std="std", delay=0, smooth_sigma=None),
        "rise": dict(wlen=1, wait=1, std="std", delay=0, smooth_sigma=None),
    }

    method_name = "zero"
    ##method_name = "rise"
    cls = method_map[method_name]
    const_kwds = const[method_name]
    target = target_map[method_name]

    use_plotly = False

    # Mixing factor of x_loss and y_loss. Of course both have different units
    # (numerical scales) so one cannot, in general "just mix them" without
    # accounting for that. But in the test data we use here they are about the same
    # scale.
    y_loss_fac = 0.5

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        ##sampler=optuna.samplers.NSGAIIISampler(population_size=500),
        ##sampler=optuna.samplers.GPSampler(),
        # For optuna-dashboard, but writing this is slow, seems like nothing is
        # cached and there is a write after each trial. Better use
        # optuna.visualization.plot_foo(study).
        ##storage="sqlite:///db.sqlite3",
    )
    study.optimize(
        partial(
            objective,
            y_loss_fac=y_loss_fac,
            target=target,
            const_kwds=const_kwds,
            cls=cls,
        ),
        n_trials=100,
    )
    pprint(study.best_trial)
    pprint(study.best_trial.params)

    ncols = 3
    fig, axs = plt.subplots(
        ncols=ncols, figsize=(5 * ncols, 5), tight_layout=True
    )
    idx, kwds, conv = call_check_first(
        study.best_trial.params,
        cls=cls,
        const_kwds=const_kwds,
    )
    axs[0].plot(y, ".", alpha=0.2)
    ylim = axs[0].get_ylim()
    axs[0].vlines(idx, *ylim, label="conv point")
    axs[0].vlines(
        np.argmin(np.abs(x - target)),
        *ylim,
        ls="--",
        label="target conv point",
    )
    axs[0].plot(conv.smooth_f(y))
    axs[0].legend()

    # Slow for many trials, see
    # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.importance.get_param_importances.html.
    # But using another evaluator (w/o tuning *its* params) results in very
    # different importance scores :)
    imp = optuna.importance.get_param_importances(
        study,
        ##evaluator=optuna.importance.PedAnovaImportanceEvaluator()
    )
    axs[1].bar(
        x=range(len(imp)), height=imp.values(), tick_label=list(imp.keys())
    )
    axs[1].tick_params(axis="x", labelrotation=45)
    axs[1].set_title("feature importance")

    df = study.trials_dataframe().sort_values("datetime_start")
    loss = df["value"].values
    number = df["number"].values
    lo_loss = [loss[0]]
    lo_number = [0]
    for number_i, loss_i in zip(number, loss):
        if loss_i < lo_loss[-1]:
            lo_loss.append(loss_i)
            lo_number.append(number_i)
    axs[2].plot(number, loss, ".", color="tab:blue", alpha=0.3)
    axs[2].plot(lo_number, lo_loss, color="tab:red", lw=2, label="best trial")
    axs[2].set_xlabel("trial")
    axs[2].set_ylabel("loss by trial")
    axs[2].set_title("study progress")
    axs[2].legend()

    # If you have plotly installed, then this will open some plots in the browser.
    # Still better than optuna-dashboard, since there certain plots only let you
    # select 2 params at a time (e.g. plot_rank()), whereas the plot functions show
    # all combos.
    #
    # rank and slice fail with
    #   TypeError: Object of type _ArrayFunctionDispatcher is not JSON serializable
    #
    # with optuna 3.6.1, plotly 5.20.0, Python 3.12. Bummer.
    if use_plotly:
        f_names = [
            "contour",
            "optimization_history",
            "parallel_coordinate",
            ##"rank",
            "param_importances",
            ##"slice",
        ]

        # We need to keep the returned plotly figs around for plots to show.
        obs = []
        for name in f_names:
            obs.append(getattr(optuna.visualization, f"plot_{name}")(study))
            obs[-1].show()

    plt.show()
