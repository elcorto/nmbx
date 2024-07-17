from functools import partial
import numpy as np
import pytest
import scipy.stats
import scipy.optimize

from nmbx.convergence import (
    Base,
    SlopeRise,
    SlopeZero,
    SlopeZeroMin,
    SlopeZeroAbs,
    SlopeZeroMax,
    smooth_gauss,
)

STD_FUNCS = Base.get_std_funcs()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


class XyFactory:
    def __init__(self, func):
        self.func = func

    def __call__(self, noise_scale=0):
        x = np.linspace(0.1, 10, 100)
        y = self.func(x)
        if noise_scale > 0:
            rng = np.random.default_rng(123)
            return x, y + rng.normal(scale=noise_scale, size=len(x))
        else:
            return x, y


get_xy_fall_to_flat = XyFactory(lambda x: np.exp(-x))
get_xy_rise_to_flat = XyFactory(lambda x: 1 - np.exp(-x))
get_xy_rise_increasing = XyFactory(np.exp)


class FakeCheck(Base):
    # Pass in atol only to make Base.__init__ happy, it's not going to be used
    # in _check_once() below.
    def __init__(self, *args, **kwds):
        super().__init__(*args, atol=0, **kwds)

    # Fake _check_once: pass y = [0,0,1,0,...] which are results of an actual
    # check. We could also let y be bool and return that but passing a sequence
    # of check results in tests below is more instructive.
    def _check_once(self, y):
        return bool(y[-1])


def get_f_std(std):
    if std is None:
        f_std = lambda h: 1
    elif isinstance(std, str):
        f_std = lambda h: STD_FUNCS[std](h)
    else:
        f_std = std
    return f_std


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_fall_to_flat():
    x, y = get_xy_fall_to_flat()
    y += 1.0

    conv = SlopeZeroAbs(atol=1e-3)
    assert conv.check(y)

    conv = SlopeZeroMin(atol=1e-3)
    assert conv.check(y)

    # In max mode, all tests are true ("convergence").
    conv = SlopeZeroMax(atol=1e-3)
    assert all(
        conv.check_all(y) == np.array([0] + [1] * (len(y) - 1), dtype=bool)
    )

    conv = SlopeRise(atol=1e-3)
    assert not conv.check(y)


def test_rise_to_flat():
    x, y = get_xy_rise_to_flat()
    y += 1.0

    conv = SlopeZeroAbs(atol=1e-3)
    assert conv.check(y)

    conv = SlopeZeroMax(atol=1e-3)
    assert conv.check(y)

    # In min mode, all tests are true ("convergence").
    conv = SlopeZeroMin(atol=1e-3)
    assert all(
        conv.check_all(y) == np.array([0] + [1] * (len(y) - 1), dtype=bool)
    )

    conv = SlopeRise(atol=1e-3)
    assert not conv.check(y)


def test_rise_increasing():
    x, y = get_xy_rise_increasing()
    y += 1.0

    conv = SlopeRise(atol=1e-3)
    assert conv.check(y)

    conv = SlopeZeroAbs(atol=1e-3)
    assert not conv.check(y)

    # In min mode, all tests are true ("convergence").
    conv = SlopeZeroMin(atol=1e-3)
    assert all(
        conv.check_all(y) == np.array([0] + [1] * (len(y) - 1), dtype=bool)
    )

    conv = SlopeZeroMax(atol=1e-3)
    assert not conv.check(y)


@pytest.mark.parametrize("mode", ["abs", "min"])
def test_zero_fall_modes(mode):
    history = np.array([20, 10, 5, 2, 1, 1, 1, 6, 6.2])
    bool_arrs = dict(
        abs=np.array([0, 0, 0, 0, 1, 1, 1, 0, 1], dtype=bool),
        min=np.array([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=bool),
    )
    pos_idx = 4
    conv = SlopeZero(atol=1.1, mode=mode)
    assert conv.check_first(history) == pos_idx
    assert (conv.check_all(history) == bool_arrs[mode]).all()


@pytest.mark.parametrize("mode", ["abs", "max"])
def test_zero_rise_modes(mode):
    history = np.array([1, 2, 3, 4, 4.2, 4.3, 4.3, 3.5])
    bool_arrs = dict(
        abs=np.array([0, 0, 0, 0, 1, 1, 1, 0], dtype=bool),
        max=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=bool),
    )
    pos_idx = 4
    conv = SlopeZero(atol=0.5, mode=mode)
    assert conv.check_first(history) == pos_idx
    assert (conv.check_all(history) == bool_arrs[mode]).all()


def test_rise():
    history = np.array([20, 10, 5, 2, 1, 1, 1, 6, 6.2])
    bool_arr = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=bool)
    pos_idx = 7
    conv = SlopeRise(atol=1.1)
    assert conv.check_first(history) == pos_idx
    assert (conv.check_all(history) == bool_arr).all()


def _gen_combos_std_scale_offset(
    std_lst=[
        "std",
        "siqr",
        "smad",
        scipy.stats.iqr,
        scipy.stats.median_abs_deviation,
    ],
):
    """Generate combos for parametrize."""
    # For std=None only scale=1 and offset=0 work.
    combos = [(None, 1, 0)]
    for std in std_lst:
        assert std is not None
        for scale in [1, 100]:
            for offset in [0, 100]:
                combos.append((std, scale, offset))
    return combos


@pytest.mark.parametrize("std, scale, offset", _gen_combos_std_scale_offset())
def test_atol_std(std, scale, offset):
    """For the manually created data below, all std methods work (same pos_idx
    and bool_arr for all scale and offset).
    """
    history = np.array([20, 10, 5, 2, 1, 1, 1, 6, 6.2])
    arr_zero_abs = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1], dtype=bool)
    arr_zero_min = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=bool)
    arr_rise = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=bool)

    f_std = get_f_std(std)

    for cls, pos_idx, bool_arr in [
        (SlopeZeroAbs, 4, arr_zero_abs),
        (SlopeZeroMin, 4, arr_zero_min),
        (SlopeRise, 7, arr_rise),
    ]:
        # atol = 1.1 is for the raw history values (scale=1, offset=0). atol =
        # 1.1 / std(y) and using std != None makes atol invariant to affine
        # transforms y * scale + offset. This is because in standardization, (y
        # - std_avg(y)) / std(y) removes the offset and scales y to be in units
        # of std(y).
        atol = 1.1 / f_std(history)
        conv = cls(atol=atol, std=std)
        assert conv.check_first(history * scale + offset) == pos_idx
        assert (conv.check_all(history * scale + offset) == bool_arr).all()


@pytest.mark.parametrize(
    "xy_func, cls, mode",
    [
        (get_xy_fall_to_flat, SlopeZero, "min"),
        (get_xy_fall_to_flat, SlopeZero, "abs"),
        (get_xy_rise_to_flat, SlopeZero, "max"),
        (get_xy_rise_to_flat, SlopeZero, "abs"),
    ],
)
@pytest.mark.parametrize(
    "std, scale, offset", _gen_combos_std_scale_offset(std_lst=["std"])
)
@pytest.mark.parametrize("std_avg", [np.mean, np.median])
def test_atol_std_xy_to_flat(xy_func, cls, mode, std, scale, offset, std_avg):
    """For generated xy data based on exp(), only std="std" works. This is also
    reflected in the behavior of IQR and MAD in examples/visualize_std.py. So
    better stick to std="std" in production.
    """
    f_std = get_f_std(std)
    _, history = xy_func()

    # Get reference pos_idx and bool_arr.
    atol_ref = 0.03
    wlen = 1
    kwds = dict(mode=mode, wlen=wlen, std_avg=std_avg)
    conv = cls(atol=atol_ref, std=None, **kwds)
    pos_idx_ref = conv.check_first(history)
    bool_arr_ref = conv.check_all(history)
    assert 0 < pos_idx_ref < len(history)

    conv = cls(atol=atol_ref / f_std(history), std=std, **kwds)
    assert conv.check_first(history * scale + offset) == pos_idx_ref
    assert (conv.check_all(history * scale + offset) == bool_arr_ref).all()


@pytest.mark.parametrize("xy_func, cls", [(get_xy_rise_increasing, SlopeRise)])
@pytest.mark.parametrize("std, scale, offset", _gen_combos_std_scale_offset())
@pytest.mark.parametrize("std_avg", [np.mean, np.median])
def test_atol_std_xy_rise_increasing(
    xy_func, cls, std, scale, offset, std_avg
):
    f_std = get_f_std(std)
    _, history = xy_func()

    # Get reference pos_idx and bool_arr.
    atol_ref = 0.03
    wlen = 1
    kwds = dict(wlen=wlen, std_avg=std_avg)
    conv = cls(atol=atol_ref, std=None, **kwds)
    pos_idx_ref = conv.check_first(history)
    bool_arr_ref = conv.check_all(history)
    assert 0 < pos_idx_ref < len(history)

    conv = cls(atol=atol_ref / f_std(history), std=std, **kwds)
    assert conv.check_first(history * scale + offset) == pos_idx_ref
    assert (conv.check_all(history * scale + offset) == bool_arr_ref).all()


@pytest.mark.parametrize(
    "wait", [1, None, pytest.param(0, marks=pytest.mark.xfail)]
)
def test_no_wait(wait):
    conv = FakeCheck(wlen=1, wait=wait)

    y = [0, 0, 0]
    assert not conv.check(y)

    y = [0, 0, 0, 1]
    assert conv.check(y)


def test_wait():
    conv = FakeCheck(wlen=1, wait=3, std=None)

    y = [0, 0, 0]
    assert not conv.check(y)

    y = [0, 0, 0, 1]
    assert not conv.check(y)
    assert conv._wait_counter == 1

    y = [0, 0, 0, 1, 1]
    assert not conv.check(y)
    assert conv._wait_counter == 2

    y = [0, 0, 0, 1, 1, 1]
    assert conv.check(y)
    assert conv._wait_counter == 3

    y = [0, 0, 0, 1, 1, 1, 0]
    assert not conv.check(y)
    assert conv._wait_counter == 1

    y = [0, 0, 0, 1, 1, 1, 0, 1]
    assert not conv.check(y)
    assert conv._wait_counter == 1


def test_delay():
    history = [1] * 10

    for delay in [0, 8]:
        conv = SlopeZero(wait=1, wlen=1, delay=delay, atol=0.1)
        assert conv.check(history)

    for delay in [9, 10]:
        conv = SlopeZero(wait=1, wlen=1, delay=delay, atol=0.1)
        assert not conv.check(history)


@pytest.mark.parametrize("mode", ["abs", "min"])
def test_gaussian_filter(mode):
    gen = get_xy_fall_to_flat
    x, y_noise = gen(noise_scale=0.2)
    y_gt = gen.func(x)

    # Find the best smooth_sigma :)
    def func(s, *, y_noise, y_gt):
        d = y_gt - smooth_gauss(y_noise, sigma=s)
        return np.dot(d, d)

    opt = scipy.optimize.minimize(
        partial(func, y_noise=y_noise, y_gt=y_gt), x0=5
    )
    print(opt)
    s_opt = opt.x

    wlen = 5

    # Make sure that the check fails for noisy data when we don't use
    # smoothing.
    conv = SlopeZero(atol=0.05, mode=mode, smooth_sigma=None, wlen=wlen)
    assert conv.check(y_gt)
    assert not conv.check(y_noise)

    conv = SlopeZero(atol=0.05, mode=mode, smooth_sigma=s_opt, wlen=wlen)
    assert conv.check(y_gt)
    assert conv.check(y_noise)

    conv = SlopeZero(atol=0.001, mode=mode, smooth_sigma=s_opt, wlen=wlen)
    assert conv.check(y_gt)
    if mode == "abs":
        assert not conv.check(y_noise)
    elif mode == "min":
        assert conv.check(y_noise)
    else:
        raise ValueError("mode!? {mode=}")
