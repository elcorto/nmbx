from typing import Sequence, Callable
import numpy as np


class Base:
    """Detect convergence of the numbers in `history` passed to check().
    _check_once() implements the condition that must be True in order to be
    treated as converged. `wait` (a.k.a. "patience") requires the condition to
    be True `wait` times in a row in consecutive check(history) calls, where we
    assume that new points are appended to `history` in between calls.

    To filter noise in `history` we compare means (default, see `reduction`) of
    `wlen` entries in `history`. When using `wait`, then we have a moving
    average filter. Use `wlen=1` to disable smoothing.
    """

    def __init__(
        self,
        wlen: int = 1,
        wait: int = None,
        tol: float = None,
        reduction: Callable = np.mean,
    ):
        self.tol = tol
        self.wait = wait
        self.wlen = wlen
        self.reduction = reduction

        self._wait_counter = 1
        self._prev_check_result = False

    def check(self, history: Sequence[float]):
        if len(history) < (2 * self.wlen):
            return False
        result = self._check_once(history)
        if self.wait is None:
            return result
        if result and self._prev_check_result:
            self._wait_counter += 1
        else:
            self._wait_counter = 1
        self._prev_check_result = result
        return self._wait_counter >= self.wait

    def _get_prev_last(self, history):
        prev = self.reduction(np.array(history[-2 * self.wlen : -self.wlen]))
        last = self.reduction(np.array(history[-self.wlen :]))
        return prev, last

    def _check_once(self, *args, **kwds):
        raise NotImplementedError


class SlopeRise(Base):
    """Detect rise in history (a.k.a. "early stopping" in ML speak).

    last - tol > prev

    Assumes falling history.
    """

    def _check_once(self, history: Sequence[float]):
        prev, last = self._get_prev_last(history)
        return last - self.tol > prev


class SlopeZero(Base):
    """Check convergence of difference (i.e. slope approaching zero).

    |last - prev| < tol

    Falling or rising history works.
    """

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        assert self.tol > 0, "tol must be > 0 for flat slope detection"

    def _check_once(self, history: Sequence[float]):
        prev, last = self._get_prev_last(history)
        return abs(last - prev) < self.tol


EarlyStopping = SlopeRise
