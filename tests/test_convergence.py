import numpy as np
import pytest

from nmbx.convergence import Base, SlopeRise, SlopeZero

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def get_xy_fall():
    x = np.linspace(0.1, 10, 100)
    y = np.exp(-x)
    return x, y


def get_xy_rise():
    x = np.linspace(0.1, 10, 100)
    y = np.exp(x)
    return x, y


class FakeCheck(Base):
    # Fake _check_once: pass y = [0,0,1,0,...] which are results of an actual
    # check. We could also let y be bool and return that but passing a sequence
    # of check results in tests below is more instructive.
    def _check_once(self, y):
        return bool(y[-1])


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_fall_only():
    x, y = get_xy_fall()
    y += 1.0

    conv = SlopeZero(tol=1e-3)
    assert conv.check(y)

    conv = SlopeRise(tol=0)
    assert not conv.check(y)


def test_rise_only():
    x, y = get_xy_rise()
    y += 1.0

    conv = SlopeZero(tol=1e-3)
    assert not conv.check(y)

    conv = SlopeRise(tol=0)
    assert conv.check(y)


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
    conv = FakeCheck(wlen=1, wait=3, standardize=False)

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
        conv = SlopeZero(wait=1, wlen=1, delay=delay, tol=0.1)
        assert conv.check(history)

    for delay in [9, 10]:
        conv = SlopeZero(wait=1, wlen=1, delay=delay, tol=0.1)
        assert not conv.check(history)
