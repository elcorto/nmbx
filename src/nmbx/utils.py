import numpy as np


def generate_history_data(
    noise_scale=0, dx=0.01, x0=0, x1=6, x_fall_stop=1.5, x_rise_start=4, y0=1.0
):
    def func_single_point(x):
        lamb = -5
        if x <= x_fall_stop:
            return y0 + np.exp(lamb * x) - np.exp(lamb * x_fall_stop)
        elif x <= x_rise_start:
            return y0
        else:
            return y0 + (x - x_rise_start) ** 2.0 / 15.0

    # frompyfunc returns PyObject arrays, that's why the extra lambda
    _np_func = np.frompyfunc(func_single_point, 1, 1)
    np_func = lambda x: _np_func(x).astype(np.float64)

    x = np.arange(x0, x1, dx)

    # Shift and scale to show effect of standardization.
    y_scale = 20
    y_shift = 100
    func = lambda x: np_func(x) * y_scale + y_shift
    rng = np.random.default_rng(seed=123)
    if noise_scale > 0:
        y = func(x) + rng.normal(scale=noise_scale * y_scale, size=len(x))
    else:
        y = func(x)
    return x, y, func
