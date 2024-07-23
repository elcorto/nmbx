from typing import Callable, Sequence
from functools import partial
import warnings

import numpy as np
import scipy.stats
from scipy.ndimage import gaussian_filter

# ~2.2e-16
EPS = np.finfo(np.float64).eps


def smooth_gauss(y, sigma):
    # "nearest" makes the least assumptions about how the signal would extend
    # beyond the borders.
    return gaussian_filter(y, sigma=sigma, mode="nearest")


class Base:
    def __init__(
        self,
        wlen: int = 1,
        wait: int = 1,
        delay: int = 0,
        atol: float = None,
        rtol: float = None,
        wlen_avg: Callable = np.median,
        std: Callable | str = None,
        std_avg: Callable = np.median,
        smooth_sigma: float = None,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        wlen
            Window length over which to take reductions. To filter noise in
            `history` we compare the last and previous median/mean/.. (see
            `wlen_avg`) of `wlen` entries in `history` such that
            ``last=wlen_avg(history[-wlen:])``,
            ``prev=wlen_avg(history[-2*wlen:-wlen])``. Use `wlen=1` to disable
            smoothing. Then ``last=history[-1]`` and ``prev=history[-2]``.
        wait
            (a.k.a. "patience") requires the convergence condition to be True
            `wait` times in a row in consecutive check(history) calls, where we
            assume that new points are appended to `history` in between calls.
            Use 1 or None to disable.
        delay
            Start checking for convergence only after this many iterations.
        atol
            Absolute tolerance for convergence check. The meaning is defined in
            derived classes' ``_check_once()`` methods. The value also depends on
            whether standardization is used. If yes, then history values are in
            units of the spread calculated by the `std` method.
        rtol
            Relative tolerance, used as ``tol = rtol * prev`. Use this or
            `atol.`
        wlen_avg
            Something like np.mean or np.median, used with `wlen`. The median
            is more robust against outliers, which are are often encountered
            with stochastic optimizers.
        std : string {'std', 'smad', 'siqr'}, None or a Callable
            The method to calculate the history spread for standardization. For
            MAD and IQR, we scale the spread to that of a standard normal
            distribution for better comparison of `tol` with "std", hence the
            "s" prefix. If you don't want that, then supply a callable such as
            ``std=scipy.stats.iqr`` or ``scipy.stats.median_abs_deviation``.
            Use ``std=None`` to disable standardization.
        std_avg
            Callable to calculate the center of the history if `std` is used.
            If ``std="std"`` (or ``std=np.std``) and ``std_avg=np.mean`` then
            a z-score is used for standardization.
        smooth_sigma
            In addition to `wlen`, apply a Gaussian filter to reduce noise. As
            a rule of thumb, sigma = 1/5 M where M is the kernel length of a
            Hann window, as in ``scipy.signal.convolve(y, hann(M), "same")``
            but w/o the edge effect handling in
            ``scipy.ndimage.gaussian_filter()``. Use None to disable.
        verbose
            Print diagnostics

        Notes
        -----
        wait:

        check() is an online method and cannot, by construction, correct
        for wait (i.e. subtract patience). The user needs to take care of that.
        check_all() and check_first() are offline methods and could correct for
        `wait`. But we don't do this in order to be consistent with check().
        """
        self.atol = atol
        self.rtol = rtol
        self.wait = 1 if wait is None else wait
        self.wlen = wlen
        self.delay = delay
        self.wlen_avg = wlen_avg
        self.verbose = verbose
        self.smooth_sigma = smooth_sigma

        assert self.wlen >= 1, "wlen must be >= 1"
        assert self.wait >= 1, "wait must be >= 1"

        assert [rtol, atol].count(None) == 1, "Use either rtol or atol"
        if rtol is not None:
            assert std is None, "rtol given, use std only with atol"

        self._wait_counter = 1
        self._prev_check_result = False

        if isinstance(std, str):
            std_funcs = self.get_std_funcs()
            assert std in (
                lst := list(std_funcs.keys())
            ), f"{std=} not one of {lst}"
            self.std_f = lambda y: (y - std_avg(y)) / (EPS + std_funcs[std](y))
        elif isinstance(std, Callable):
            self.std_f = lambda y: (y - std_avg(y)) / (EPS + std(y))
        elif std is None:
            self.std_f = lambda x: x
        else:
            raise ValueError("Illegal type for std.")

        if self.smooth_sigma is None:
            self.smooth_f = lambda x: x
        else:
            self.smooth_f = lambda x: smooth_gauss(x, self.smooth_sigma)

    @staticmethod
    def get_std_funcs():
        # CDF^-1(3/4) of the standard normal
        norm_ppf_3_4 = scipy.stats.norm(loc=0, scale=1).ppf(0.75)
        # MAD is < sigma, scale=1.4826
        mad_scale = 1.0 / norm_ppf_3_4
        # IQR is > sigma, scale=0.7413
        iqr_scale = 0.5 * mad_scale
        return dict(
            std=np.std,
            siqr=lambda x: iqr_scale * scipy.stats.iqr(x),
            smad=lambda x: mad_scale * scipy.stats.median_abs_deviation(x),
        )

    def preprocess_history(self, history):
        return self.std_f(self.smooth_f(history))

    def check(self, history: Sequence[float]) -> bool:
        """Run one check on history's end (normal use case in a loop). Return
        bool.
        """
        if len(history) < (2 * self.wlen + self.delay):
            return False

        result = self._check_once(self.preprocess_history(history))

        if self.wait == 1:
            return result
        if result and self._prev_check_result:
            self._wait_counter += 1
        else:
            self._wait_counter = 1
        self._prev_check_result = result
        return self._wait_counter >= self.wait

    def check_all(self, history) -> Sequence[bool]:
        """Iterate over history and run all len(history) checks.

        Only useful for analyzing the effect of settings (tol, std, ...) on the
        method's performance, given a recorded history.
        """
        return np.array(
            [self.check(history[: (ii + 1)]) for ii in range(len(history))],
            dtype=bool,
        )

    def check_first(self, history) -> int | None:
        """Iterate over history and return the array index of the first
        positive check. If nothing is found, return None.

        Useful helper for optimization of settings, given a recorded history.
        """
        for ii in range(2, len(history)):
            if self.check(history[:ii]):
                return ii - 1

    def _get_prev_last_tol(self, history):
        prev = self.wlen_avg(np.array(history[-2 * self.wlen : -self.wlen]))
        last = self.wlen_avg(np.array(history[-self.wlen :]))
        if self.rtol is not None:
            tol = self.rtol * abs(prev)
            if tol < 2 * EPS:
                warnings.warn(f"rtol: {tol=} < {2*EPS=}")
        else:
            tol = self.atol
        return prev, last, tol

    def _check_once(self, *args, **kwds):
        """Implements the convergence condition.

        Called in check().
        """
        raise NotImplementedError


class SlopeRise(Base):
    """Detect rise in history.

    last - tol > prev
    """

    def _check_once(self, history: Sequence[float]):
        prev, last, tol = self._get_prev_last_tol(history)
        val = last - tol
        cond = val > prev
        if self.verbose:
            op = ">" if cond else "<="
            print(f"{last=} - {tol=} = {val} {op} {prev=}")
        return cond


class SlopeZero(Base):
    """Check for slope approaching zero.

    mode="abs": falling or rising history
        |last - prev| < tol

    mode="min": falling history
        last + tol > prev

    mode="max": rising history
        last - tol < prev

    Notes
    -----
    Modes min and max is what keras' and lightning's EarlyStopping implement
    (if ``std=None`` and ``atol`` is used). However note that they implement a
    test for "is improvement", while we test the opposite, i.e. swap the
    comparison operation.
    """

    def __init__(self, *args, mode="abs", **kwds):
        super().__init__(*args, **kwds)
        self.mode = mode

    def _check_once(self, history: Sequence[float]):
        prev, last, tol = self._get_prev_last_tol(history)
        if self.mode == "abs":
            val = abs(last - prev)
            cond = val < tol
            if self.verbose:
                op = "<" if cond else ">="
                print(f"|{last=} - {prev=}| = {val} {op} {tol=}")
        elif self.mode == "min":
            val = last + tol
            cond = val > prev
            if self.verbose:
                op = ">" if cond else "<="
                print(f"{last=} + {tol=} = {val} {op} {prev=}")
        elif self.mode == "max":
            val = last - tol
            cond = val < prev
            if self.verbose:
                op = "<" if cond else ">="
                print(f"{last=} - {tol=} = {val} {op} {prev=}")
        else:
            raise ValueError(f"Unknown mode {self.mode=}")
        return cond


class MultiCheck:
    """Apply a check to multiple histories. Return True when all checks are
    True.
    """

    def __init__(self, cls: Base, names: Sequence[str], **kwds):
        self.checkers = {n: cls(**kwds) for n in names}
        self.names = names
        self._set_names = set(names)
        self.verbose = kwds.get("verbose", False)

    def check(self, histories: dict[str, Sequence[float]]):
        assert set(histories.keys()) == self._set_names
        all_results = True
        for name, hist in histories.items():
            this_result = self.checkers[name].check(hist)
            if self.verbose and this_result:
                print(f"{name} converged at iter={len(hist)}")
            all_results = all_results and this_result
        return all_results


# Aliases mainly for tests and to save some typing in examples.
SlopeZeroAbs = partial(SlopeZero, mode="abs")
SlopeZeroMin = partial(SlopeZero, mode="min")
SlopeZeroMax = partial(SlopeZero, mode="max")
