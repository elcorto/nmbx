# nmbx

nmbx, NuMBoX, n_umb_ox, $n_umb_ox$

A box of tools that deal with numbers.

# Content

## Convergence tests

In `nmbx.convergence` we have tools for testing convergence in a sequence of
numbers, such as the loss when training machine learning models or any other
iterative process with a converging metric.

`SlopeZero` detects a flat plateau ("zero slope"), which is a general-purpose
method. `SlopeRise` detects a rise in the history after a flat plateau.

The idea is to call either in a training loop, passing a history of loss
values.

### Example

```py
from nmbx.convergence import SlopeZero

# Detect convergence with wait=10 iterations of "patience", an absolute tolerance
# of 0.01 and a moving average window of wlen=25 points. Start checking not before
# 100 iterations have been performed.
conv = SlopeZero(wlen=25, atol=0.01, wait=10, delay=100)

history = []
while True:
    history.append(compute_loss(model, data))
    if conv.check(history):
        print("converged")
        break
```

### Methods

`SlopeZero` implements the same logic as found in [Keras'][keras_es] or
[Lightning's][liight_es] `EarlyStopping(mode=...)` with `mode="min"` or
`"max"`. In addition we provide `mode="abs"` (detect convergence not assuming a
direction).

Since we only work with a given list of numbers $y_i$ in the history, we have
$\Delta x=1$ in the slope $\Delta y/\Delta x$. Therefore, the `atol` and `rtol`
parameters are to be understood w.r.t. $y$. Please check the doc strings for
what `tol` does in each method, where `tol = atol` or `tol = rtol * abs(prev)`. In
short

* `SlopeRise`: `last - tol > prev`
* `SlopeZero`:
  * `mode="abs": |last - prev| < tol`
  * `mode="min": last + tol > prev`
  * `mode="max": last - tol < prev`

`last` and `prev` are the mean/median/... (see `wlen_avg`) over the last and
previous non-overlapping windows of `wlen` points each. This means that the
earliest convergence point can be detected after `2 * wlen` iterations. With
`delay`, the first possible convergence point is after `2 * wlen + delay`
iterations.


### Settings

We implement several options that can make convergence checks more robust and
versatile than vanilla "early stopping".

* Noise filtering (smoothing): Histories are often noisy (e.g. when using
  stochastic optimizers). In vanilla early stopping, the only counter measure is
  using "patience". We have the option to smooth the history using
  * a Gaussian filter (set `smooth_sigma`) and/or
  * a moving reduction of window size `wlen` (reduction = mean/median/..., see
    `wlen_avg`). `wlen=1` means a window of one, so no noise filtering of this
    kind. You can still use the Gaussian filter by setting `smooth_sigma`.

* You may use some `delay` to make sure to run at least this many iterations
  before checking for convergence. This can help to avoid early false positive
  convergence detection.

Can we get "transferable" tolerances? Well, kind of.

* Absolute (`atol`) or relative (`rtol`) tolerances: If you know the unit of
  the history and can say something like "we call changes below 0.01
  converged", then use `atol`. Else, try to use a relative tolerance `rtol`, in
  which case we use `tol = rtol * abs(prev)`.
  * Pro: This will be invariant to scaling $y' = y s$.
  * Con: Will not be invariant to a shift $y' = y + c$.

* Standardization: You can standardize `history`, for example using a z-score
  (set `std="std"` and `std_avg=np.mean`) to zero mean and unity standard
  deviation such that, at each iteration $i$, `atol` will be
  in units of $\sigma_i$. Now the convergence criterion is "stop if changes are
  below `atol` standard deviations".
  * Pro: This is helpful for histories of very different numerical scale
    but similar "shape" and where you don't know or care about the unit of $y$.
    More precisely, you can apply the same `atol` to all histories which
    differ from $y$ by an affine transform $y' = y s + c$.
  * Con: Since `check()` is an online method, the standardization is performed
    for each iteration, using all history values provided so far. Therefore
    $\sigma_i$, and thus the unit of `atol`, will change which makes the effect
    of standardization more difficult to interpret. There are corner cases
    where this method doesn't work (for example a noise-free constant history
    where $\sigma$ is zero). Also some experimentation is needed to find good
    `atol` value. Check `examples/convergence/visualize_std.py` and all
    ``test_atol_std*`` tests.

Here are results from a parameter study in
`examples/convergence/param_study.py` with noise-free and noisy histories,
where we explore the above parameters. Blue points are the histories. The other
points indicate when `check()` is True. The points marked with vertical dashed
lines are the *first* points where the check is True, i.e. where you would
break out of the training loop. If no colored points show up, then this means
that the corresponding parameter setting leads to no convergence detection.

![](doc/pics/conv_no_noise.png)
![](doc/pics/conv_noise.png)

We observe that `SlopeZero` is pretty robust against noise, while `SlopeRise`
is more tricky, i.e. it is not clear what the right parameters are in this
case.

### Recommendations for parameter settings

* Use `wlen > 1` (smoothing by moving reduction) and `wait > 1` ("patience").
  Typical values are `wlen=10...20` (but that depends very much on your data),
  and `wait=5..10`. Using the Gaussian filter  (`smooth_sigma`) is a more
  effective smoothing option, but still some `wlen` can help.
* When data is noisy, it *can* help to raise `tol` to prevent too early / false
  convergence detection. But having a good setting for `wlen` or `smooth_sigma` is
  preferable.
* To find a good `smooth_sigma` value, start with `smooth_sigma = wlen/3` where
  `wlen` is the value you would use for smoothing instead (e.g. `wlen=15 ->
  smooth_sigma=3`). See `examples/convergence/find_sigma_from_wlen.py`.
* Use `delay` if you know that you need to run at least this many iterations.
  This also helps too avoid early false positives.
* When using standardization (e.g. `std="std"`), start with `atol=0.01`, so
  "stop when things fluctuate by less than 0.01 standard deviations".
* Take the ball park numbers used in `examples/convergence/` as starting point.

### Finding good parameters

One can frame this as an optimization problem. We have an example using Optuna
in `examples/convergence/param_opt.py`. You can use this if you have a
representative history recorded for your application and plan to use
convergence detection for many similar runs.

### Other packages

* https://github.com/JuliaAI/EarlyStopping.jl

[keras_es]: https://keras.io/api/callbacks/early_stopping
[liight_es]: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.pytorch.callbacks.EarlyStopping
