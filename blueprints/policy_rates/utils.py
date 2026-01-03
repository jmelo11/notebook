import numpy as np
import pandas as pd


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fritsch-Carlson monotone slopes for piecewise cubic Hermite (PCHIP).
    Returns m[i] ~ dy/dx at each knot x[i].
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("x and y must be numpy arrays")
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1D arrays of equal length")

    n = x.size
    if n < 2:
        return np.zeros_like(x)

    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("tenors must be strictly increasing")

    delta = np.diff(y) / h
    m = np.zeros(n, dtype=y.dtype)

    if n == 2:
        m[:] = delta[0]
        return m

    # Endpoint estimates + limiting
    m0 = ((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    mn = ((2 * h[-1] + h[-2]) * delta[-1] -
          h[-1] * delta[-2]) / (h[-1] + h[-2])

    def _limit_endpoint(me, de):
        if me * de <= 0:
            return 0.0 * me
        if np.abs(me) > 3 * np.abs(de):
            return 3 * de
        return me

    m[0] = _limit_endpoint(m0, delta[0])
    m[-1] = _limit_endpoint(mn, delta[-1])

    # Interior points (weighted harmonic mean, with sign/flat checks)
    for k in range(1, n - 1):
        if delta[k - 1] == 0 or delta[k] == 0 or (delta[k - 1] * delta[k] < 0):
            m[k] = 0.0 * delta[k]
        else:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    return m


def _hermite_eval_and_deriv(
    x: np.ndarray, y: np.ndarray, m: np.ndarray, xq
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate y(xq) and dy/dx(xq) for piecewise cubic Hermite interpolation.
    Linear extrapolation using endpoint slopes.
    """
    if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(m, np.ndarray)):
        raise TypeError("x, y, m must be numpy arrays")
    if x.ndim != 1 or y.ndim != 1 or m.ndim != 1 or x.size != y.size or x.size != m.size:
        raise ValueError("x, y, m must be 1D arrays of equal length")

    xq_arr = xq if isinstance(xq, np.ndarray) else np.array(xq)

    yq = np.empty_like(xq_arr, dtype=y.dtype)
    dyq = np.empty_like(xq_arr, dtype=y.dtype)

    left = xq_arr <= x[0]
    right = xq_arr >= x[-1]
    mid = ~(left | right)

    # Linear extrapolation
    if np.any(left):
        dx = xq_arr[left] - x[0]
        yq[left] = y[0] + m[0] * dx
        dyq[left] = m[0]

    if np.any(right):
        dx = xq_arr[right] - x[-1]
        yq[right] = y[-1] + m[-1] * dx
        dyq[right] = m[-1]

    # Hermite on each interval
    if np.any(mid):
        i = np.searchsorted(x, xq_arr[mid], side="right") - 1
        i = np.clip(i, 0, x.size - 2)

        x0 = x[i]
        x1 = x[i + 1]
        h = x1 - x0
        s = (xq_arr[mid] - x0) / h  # in [0,1]

        y0 = y[i]
        y1 = y[i + 1]
        m0 = m[i]
        m1 = m[i + 1]

        s2 = s * s
        s3 = s2 * s

        h00 = 2 * s3 - 3 * s2 + 1
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2

        yq[mid] = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1

        dh00 = 6 * s2 - 6 * s
        dh10 = 3 * s2 - 4 * s + 1
        dh01 = -6 * s2 + 6 * s
        dh11 = 3 * s2 - 2 * s

        dyq[mid] = (dh00 * y0 + dh10 * h * m0 + dh01 * y1 + dh11 * h * m1) / h

    return yq, dyq


class Curve:
    """
    Discount curve with PCHIP interpolation on log(discount factors).

    Key fix vs your version:
      - Anchor at T=0 with DF=1, so fwd_rate(0, 1D) is well-defined and consistent.
    """

    def __init__(self, tenors: np.ndarray, dfs: np.ndarray, *, copy: bool = True):
        if not isinstance(tenors, np.ndarray) or not isinstance(dfs, np.ndarray):
            raise TypeError("tenors and dfs must be numpy arrays")
        if tenors.ndim != 1 or dfs.ndim != 1 or tenors.size != dfs.size:
            raise ValueError(
                "tenors and dfs must be 1D arrays of equal length")
        if tenors.size < 2:
            raise ValueError("need at least 2 curve points")
        if np.any(dfs <= 0):
            raise ValueError("dfs must be > 0")

        # Sort by tenor
        idx = np.argsort(tenors)
        t = tenors[idx].astype(float)
        df = dfs[idx].astype(float)

        if copy:
            t = t.copy()
            df = df.copy()

        # ---- IMPORTANT: anchor at time 0 ----
        # Ensure DF(0)=1 so log_discount(0)=0 and forwards from 0 are correct.
        if t[0] > 0.0:
            t = np.concatenate(([0.0], t))
            df = np.concatenate(([1.0], df))
        else:
            # If you have an explicit 0 tenor, force it to DF=1.
            # (If there are multiple zeros, keep the first.)
            zero_idx = np.where(np.isclose(t, 0.0))[0]
            if zero_idx.size > 0:
                df[zero_idx[0]] = 1.0

        if np.any(np.diff(t) <= 0):
            raise ValueError(
                "tenors must be strictly increasing (after anchoring)")

        log_df = np.log(df)
        dlog_df_dt = _pchip_slopes(t, log_df)

        self.tenors = t
        self.dfs = df
        self._log_dfs = log_df
        self._dlog_df_dt = dlog_df_dt

    def log_discount(self, T):
        y, _ = _hermite_eval_and_deriv(
            self.tenors, self._log_dfs, self._dlog_df_dt, T)
        return y

    def discount(self, T):
        return np.exp(self.log_discount(T))

    def inst_fwd(self, T):
        _, dy = _hermite_eval_and_deriv(
            self.tenors, self._log_dfs, self._dlog_df_dt, T)
        return -dy

    def fwd_rate(self, T1, T2):
        """
        Continuously-compounded average forward rate over [T1, T2]:
            r = - (log P(T2) - log P(T1)) / (T2 - T1)
        """
        if T2 <= T1:
            return np.nan
        log_df1 = self.log_discount(T1)
        log_df2 = self.log_discount(T2)
        return -(log_df2 - log_df1) / (T2 - T1)

    


def yearfrac(d0, d1, dc=360):
    return (pd.Timestamp(d1) - pd.Timestamp(d0)).days / dc
