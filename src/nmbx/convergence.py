from typing import Sequence, Callable
import numpy as np


class Base:
    """Detect convergence of the numbers in `history` passed to check().
    _check_once() implements the condition that must be True in order to be
    treated as converged. `wait` (a.k.a. "patience") requires the condition to
    be True `wait` times in a row in consecutive check(history) calls,
    where we assume that new points are appended to `history` in between calls.

    To filter noise in `history` we compare means (default, see `reduction`) of
    `wlen` entries in `history`. When using `wait`, then we have a moving
    average filter. Use `wlen=1` to disable smoothing.
    """

    def __init__(
        self,
        wlen: int = 1,
        wait: int = 1,
        tol: float = None,
        reduction: Callable = np.mean,
        standardize: bool = True,
        std_eps: float = 1e-16,
        std_reduction: Callable = np.median,
        verbose: bool = False,
    ):
        self.tol = tol
        self.wait = 1 if wait is None else wait
        self.wlen = wlen
        self.reduction = reduction
        self.standardize = standardize
        self.std_eps = std_eps
        self.std_reduction = std_reduction
        self.verbose = verbose

        self._wait_counter = 1
        self._prev_check_result = False

        assert self.wlen >= 1, "wlen must be >= 1"
        assert self.wait >= 1, "wait must be >= 1"

    def check(self, history: Sequence[float]):
        if len(history) < (2 * self.wlen):
            return False
        if self.standardize:
            hist = (history - self.std_reduction(history)) / (
                np.std(history) + self.std_eps
            )
        else:
            hist = history
        result = self._check_once(hist)
        if self.wait == 1:
            return result
        if result and self._prev_check_result:
            self._wait_counter += 1
        else:
            self._wait_counter = 1
        self._prev_check_result = result
        return self._wait_counter >= self.wait

    def _get_prev_last(self, hist):
        prev = self.reduction(np.array(hist[-2 * self.wlen : -self.wlen]))
        last = self.reduction(np.array(hist[-self.wlen :]))
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
        val = last - self.tol
        cond = val > prev
        if self.verbose:
            op = ">" if cond else "<"
            print(f"{prev=} - tol={self.tol} = {val} {op} {prev=}")
        return cond


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
        val = abs(last - prev)
        cond = val < self.tol
        if self.verbose:
            op = "<" if cond else ">"
            print(f"|{last=} - {prev=}| = {val} {op} tol={self.tol}")
        return cond


EarlyStopping = SlopeRise
