from typing import Callable, Literal, Dict
import torch
from typing import List, Tuple, Optional, Sequence
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Iterable, Sequence, Tuple, List, Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch import nn


def weight_init_xavier(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def str_tenor_to_days(tenor: str) -> int:
    if tenor.endswith('d'):
        return int(tenor[:-1])
    elif tenor.endswith('y'):
        return int(tenor[:-1])*360
    else:
        raise ValueError("Unknown tenor format")


class Swap:
    def __init__(self, tenor: str, rate: float | None = None, dtype: torch.dtype = torch.float64, device: torch.device = torch.device('cpu')) -> None:
        self.tenor_days = str_tenor_to_days(tenor)
        self.rate = rate
        self.maturity = self.tenor_days / 360.0
        if self.tenor_days < 720:
            self.frequency = self.maturity
        else:
            self.frequency = 0.5  # semi-annual payments for tenors > 2 years
        self.yfs = torch.arange(
            0, self.maturity+self.frequency, self.frequency, dtype=dtype, device=device)

    def fair_rate(self, model: nn.Module) -> torch.Tensor:
        discounts = model.discounts(self.yfs)[1:]
        pv01 = torch.sum(discounts * self.frequency)
        fixed_leg = 1.0 - discounts[-1]
        fair_rate = fixed_leg / pv01
        return fair_rate.squeeze()


class CurveModel(ABC):
    """
    Abstract interface that both:
      - DeepONet direct DF model
      - NeuralODE+DeepONet model
    should implement.

    Required methods:
      - set_curve: set conditioning information (rates, tenors, possibly masks)
      - discounts: discount factors for a time grid
      - fwd_rates: forward rates on a grid (helper default provided, but you can override)
    """

    @abstractmethod
    def set_curve(self, rates: torch.Tensor, tenors: torch.Tensor, **kwargs) -> None:
        """
        rates : (n_tenors,)
        tenors: (n_tenors,) in years
        kwargs can include mask, etc. (ignored for full-pillar case).
        """
        raise NotImplementedError

    @abstractmethod
    def discounts(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (n_times,) in years
        returns: (n_times, 1) or (n_times,) discount factors
        """
        raise NotImplementedError

    def fwd_rates(self, pillars: torch.Tensor) -> torch.Tensor:
        """
        Default forward-rate computation on a grid:
          f_i = - (log DF(t_{i+1}) - log DF(t_i)) / (t_{i+1} - t_i)

        pillars: (n_pillars,) increasing times in years
        returns: (n_pillars-1,) forward rates on each interval
        """
        pillars = pillars.reshape(-1)
        dfs = self.discounts(pillars)
        dfs = dfs.squeeze(-1) if dfs.ndim == 2 else dfs

        dt = pillars[1:] - pillars[:-1]
        if torch.any(dt <= 0):
            raise ValueError("pillars must be strictly increasing")

        fwds = -(torch.log(dfs[1:]) - torch.log(dfs[:-1])) / dt
        return fwds


# =============================================================================
# 2) Training sample generation
# =============================================================================

@dataclass(frozen=True)
class CurvePairBatch:
    """
    Convenience container for training pairs.
    """
    rates_prev: torch.Tensor  # (B, n_tenors)
    rates_next: torch.Tensor  # (B, n_tenors)
    tenors: Optional[torch.Tensor] = None  # (n_tenors,)


# utils.py


@dataclass(frozen=True)
class CurvePairBatch:
    rates_prev: torch.Tensor  # (B, n_tenors)
    rates_next: torch.Tensor  # (B, n_tenors)
    # Optional but very useful for plotting/debugging:
    dates_prev: Optional[List[pd.Timestamp]] = None
    dates_next: Optional[List[pd.Timestamp]] = None


def get_prev_date(df: pd.DataFrame, date: pd.Timestamp) -> pd.Timestamp:
    """
    Given a date in df.index, return the previous available date in the index.

    Raises if date is the first date or not found.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a DatetimeIndex")

    df = df.sort_index()
    date = pd.Timestamp(date)

    if date not in df.index:
        raise KeyError(f"date {date} not found in index")

    loc = df.index.get_loc(date)
    # get_loc can return slice/array if duplicates; we take the last occurrence safely
    if isinstance(loc, slice):
        idx = loc.start
    elif isinstance(loc, (np.ndarray, list)):
        idx = int(loc[-1])
    else:
        idx = int(loc)

    if idx == 0:
        raise ValueError(f"date {date} has no previous date in the data")

    return df.index[idx - 1]


def sample_dates(df: pd.DataFrame, n: int, seed: Optional[int] = None) -> List[pd.Timestamp]:
    """
    Sample n random dates from df.index (without replacement).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a DatetimeIndex")
    if len(df) < n:
        raise ValueError(f"Need at least {n} rows to sample {n} dates")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    pick = rng.choice(idx, size=n, replace=False)
    pick.sort()
    return [df.index[i] for i in pick]


def get_date_pairs(
    market_rates: pd.DataFrame,
    n_pairs: int,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> List[Tuple[pd.Series, pd.Series]]:
    """
    Sample (prev_row, next_row) adjacent-in-time pairs from a DataFrame sorted by date index.
    If shuffle=True, the returned list is shuffled (pairing preserved).
    """
    if len(market_rates) < 2:
        raise ValueError("Need at least 2 rows to create adjacent pairs.")

    df = market_rates.sort_index()
    rng = np.random.default_rng(seed)

    max_pairs = len(df) - 1
    n_pairs = min(n_pairs, max_pairs)

    next_ix = rng.choice(np.arange(1, len(df)), size=n_pairs, replace=False)
    prev_ix = next_ix - 1

    pairs = [(df.iloc[pi], df.iloc[ni]) for pi, ni in zip(prev_ix, next_ix)]

    if shuffle:
        pairs = [pairs[i] for i in rng.permutation(len(pairs))]

    return pairs


def gen_training_set(
    market_rates: pd.DataFrame,
    n_pairs: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> "CurvePairBatch":
    """
    Generates a tensor batch of adjacent-day curve pairs, optionally shuffled.
    """
    pairs = get_date_pairs(market_rates, n_pairs=n_pairs,
                           seed=seed, shuffle=shuffle)

    r_prev, r_next = [], []
    for prev_row, next_row in pairs:
        r_prev.append(torch.tensor(prev_row.values,
                      device=device, dtype=dtype))
        r_next.append(torch.tensor(next_row.values,
                      device=device, dtype=dtype))

    rates_prev = torch.stack(r_prev, dim=0)
    rates_next = torch.stack(r_next, dim=0)
    return CurvePairBatch(rates_prev=rates_prev, rates_next=rates_next)


def t0_loss(
    model: CurveModel,
    rates: torch.Tensor,
    tenors: torch.Tensor,
    criterion: nn.Module = nn.MSELoss(),
    set_curve_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    set_curve_kwargs = set_curve_kwargs or {}
    model.set_curve(rates, tenors, **set_curve_kwargs)

    t0 = torch.zeros((1,), device=tenors.device, dtype=tenors.dtype)  # (1,)
    df0 = model.discounts(t0)

    # force df0 -> (1,1)
    if df0.ndim == 0:
        df0 = df0.view(1, 1)
    elif df0.ndim == 1:
        df0 = df0.view(-1, 1)
    elif df0.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected df0 shape: {tuple(df0.shape)}")

    if df0.shape != (1, 1):
        raise ValueError(f"Expected df0 shape (1,1), got {tuple(df0.shape)}")

    target = torch.ones((1, 1), device=tenors.device, dtype=tenors.dtype)
    return criterion(df0, target)


def swap_rate_loss(
    model: CurveModel,
    swaps: Sequence[Swap],
    obs_rates: torch.Tensor,
    tenors: torch.Tensor,
    set_curve_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Loss between observed pillar swap rates and model-implied swap rates computed via Swap.fair_rate(model).
    """
    set_curve_kwargs = set_curve_kwargs or {}
    model.set_curve(obs_rates, tenors, **set_curve_kwargs)

    preds = torch.stack([s.fair_rate(model) for s in swaps])  # (n_tenors,)
    if preds.shape != obs_rates.shape:
        raise ValueError(
            f"Shape mismatch preds={preds.shape} vs obs={obs_rates.shape}")

    return (preds-obs_rates).pow(2)


def fwd_dist_loss(
    model: CurveModel,
    rates_prev: torch.Tensor,
    rates_next: torch.Tensor,
    tenors: torch.Tensor,
    pillars: torch.Tensor,
    criterion: nn.Module = nn.MSELoss(),
    set_curve_kwargs_prev: Optional[Dict[str, Any]] = None,
    set_curve_kwargs_next: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Forward-rate consistency loss between consecutive days:
      MSE( fwd_next(pillars), fwd_prev(pillars) )

    pillars controls where you compute forward rates.
    """
    set_curve_kwargs_prev = set_curve_kwargs_prev or {}
    set_curve_kwargs_next = set_curve_kwargs_next or {}

    model.set_curve(rates_next, tenors, **set_curve_kwargs_next)
    f_next = model.fwd_rates(pillars)

    model.set_curve(rates_prev, tenors, **set_curve_kwargs_prev)
    f_prev = model.fwd_rates(pillars)

    return criterion(f_next, f_prev)


def long_end_loss(
    model: CurveModel,
    rates: torch.Tensor,                  # (n_tenors,)
    tenors: torch.Tensor,                 # (n_tenors,)
    t_last: float,
    horizon: float = 20.0,
    n_grid: int = 80,
    mode: str = "flat",                   # "flat" or "ufr"
    ufr: Optional[float] = None,          # required if mode="ufr"        
    set_curve_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Long-term behaviour regularizer on the forward curve beyond the last tenor.

    Tail grid: [t_last, t_last+horizon]

    mode:
      - "flat": penalize variance of forward rates in tail (encourages flat tail)
      - "ufr":  penalize deviation of tail forwards to ufr

    Also includes optional DF monotonicity penalty (no increasing DFs in tail).
    """
    set_curve_kwargs = set_curve_kwargs or {}
    device = tenors.device
    dtype = tenors.dtype

    model.set_curve(rates, tenors, **set_curve_kwargs)

    t_tail = torch.linspace(
        float(t_last),
        float(t_last) + float(horizon),
        int(n_grid),
        device=device,
        dtype=dtype,
    )

    t_tail.requires_grad_(True)
    log_dfs = torch.log(model.discounts(t_tail))

    fwds = -torch.autograd.grad(log_dfs, t_tail,
                                grad_outputs=torch.ones_like(log_dfs), create_graph=True)[0]

    if mode == "flat":
        tail_pen = (fwds - fwds.mean()).pow(2)
    elif mode == "ufr":
        if ufr is None:
            raise ValueError("ufr must be provided when mode='ufr'")
        target = torch.full_like(fwds, float(ufr))
        tail_pen = (fwds-target).pow(2)
    else:
        raise ValueError("mode must be 'flat' or 'ufr'")

    return tail_pen


# def convexity_loss(
#     model: CurveModel,
#     rates: torch.Tensor,                  # (n_tenors,)
#     tenors: torch.Tensor,     
#     set_curve_kwargs: Optional[Dict[str, Any]] = None,
# ) -> torch.Tensor:
#     """
#     Penalize forward rates with high second order derivative.
#     """
#     set_curve_kwargs = set_curve_kwargs or {}
#     device = tenors.device
#     dtype = tenors.dtype

#     model.set_curve(rates, tenors, **set_curve_kwargs)

#     t_short = torch.linspace(1/360, 1.0, 100, device=device, dtype=dtype)
#     t_long  = torch.linspace(1.0, float(tenors.max()), 80, device=device, dtype=dtype)
#     t = torch.cat([t_short, t_long[1:]])

#     t.requires_grad_(True)
#     log_dfs = torch.log(model.discounts(t))

#     fwds = -torch.autograd.grad(log_dfs, t,
#                                 grad_outputs=torch.ones_like(log_dfs), create_graph=True)[0]

#     dfwds = torch.autograd.grad(fwds, t,
#                                 grad_outputs=torch.ones_like(fwds), create_graph=True)[0]

#     freq_pen = dfwds.pow(2)
#     return freq_pen

def spike_loss(
    model: CurveModel,
    rates: torch.Tensor,
    tenors: torch.Tensor,
    n_grid: int = 200,
    weight: float = 1.0,
    t_min: float = 1/360,
    t_max: Optional[float] = None,
    cap: float = 0.50,          # threshold for |f''| (tuned)
    huber_delta: float = 0.10,  # smooth robustification
    set_curve_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Penalize *spikes* in instantaneous forwards by targeting large second differences of f(t).

    Steps:
      1) compute f(t) = - d/dt log DF(t) with autograd
      2) approximate f'' with discrete second difference
      3) apply a robust penalty only beyond a 'cap'

    This allows broad curvature (humps) but discourages sharp kinks/oscillations.
    """
    set_curve_kwargs = set_curve_kwargs or {}
    device, dtype = tenors.device, tenors.dtype

    model.set_curve(rates, tenors, **set_curve_kwargs)

    if t_max is None:
        t_max = float(tenors.max().detach().cpu().numpy())

    t = torch.linspace(float(t_min), float(t_max), int(n_grid), device=device, dtype=dtype, requires_grad=True)

    log_df = torch.log(model.discounts(t).squeeze(-1))

    f = -torch.autograd.grad(
        log_df, t, grad_outputs=torch.ones_like(log_df), create_graph=True
    )[0]

    dt = t[1] - t[0]
    f2 = (f[2:] - 2.0 * f[1:-1] + f[:-2]) / (dt * dt)

    excess = torch.relu(torch.abs(f2) - cap)

    a = torch.abs(excess)
    huber = torch.where(a <= huber_delta, 0.5 * a * a, huber_delta * (a - 0.5 * huber_delta))

    return weight * huber.mean()



def monotonicity_loss(
    model: CurveModel,
    rates: torch.Tensor,                  # (n_tenors,)
    tenors: torch.Tensor,                 # (n_tenors,)
    n_grid: int = 80,
    weight: float = 1.0,
    set_curve_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Penalize discount factors that are not monotonic decreasing.
    """
    set_curve_kwargs = set_curve_kwargs or {}
    device = tenors.device
    dtype = tenors.dtype

    model.set_curve(rates, tenors, **set_curve_kwargs)

    t = torch.linspace(
        float(1/360),
        float(tenors.max()),
        int(n_grid),
        device=device,
        dtype=dtype,
    )

    dfs = model.discounts(t).squeeze(-1)
    diffs = dfs[:-1] - dfs[1:]  # should be >= 0 for monotonicity

    penalties = torch.relu(-diffs)  # only penalize negative differences
    mono_pen = penalties.pow(2).mean()
    return mono_pen * weight


# =============================================================================
# 4) Interpolation utilities (article examples)
# =============================================================================


InterpMethod = Literal[
    "linear_df",
    "log_linear_df",
    "linear_zero",
    "cubic_zero",
    "monotone_cubic_zero",
]


def dfs_from_zeros(t: np.ndarray, z: np.ndarray) -> np.ndarray:
    """DF(t) = exp(-z(t)*t) with continuous compounding."""
    t = np.asarray(t, float)
    z = np.asarray(z, float)
    return np.exp(-z * t)


def zeros_from_dfs(t: np.ndarray, df: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    """z(t) = -log(DF(t))/t (continuous-compounded zero). t must be > 0."""
    t = np.asarray(t, float)
    df = np.asarray(df, float)
    if np.any(t <= 0):
        raise ValueError("zeros_from_dfs requires t > 0.")
    df = np.maximum(df, eps)
    return -np.log(df) / t


def instantaneous_fwd_from_df_grid(t: np.ndarray, df: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    """
    Approx instantaneous forward:
      f(t) = - d/dt log DF(t)
    computed via central differences on a grid (returns same length as t, with edge one-sided diffs).
    """
    t = np.asarray(t, float)
    df = np.asarray(df, float)
    if t.ndim != 1 or df.ndim != 1 or t.size != df.size:
        raise ValueError("t and df must be 1D arrays with the same length.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t grid must be strictly increasing.")

    df = np.maximum(df, eps)
    logdf = np.log(df)
    f = np.empty_like(t)

    # forward diff at start
    f[0] = -(logdf[1] - logdf[0]) / (t[1] - t[0])
    # central diffs
    dt = (t[2:] - t[:-2])
    f[1:-1] = -(logdf[2:] - logdf[:-2]) / dt
    # backward diff at end
    f[-1] = -(logdf[-1] - logdf[-2]) / (t[-1] - t[-2])
    return f

# -------------------------
# Natural cubic spline (no SciPy)
# -------------------------


def _natural_cubic_spline_coeffs(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Returns coefficients for natural cubic spline on (x,y).
    S_i(t) = a_i + b_i*(t-x_i) + c_i*(t-x_i)^2 + d_i*(t-x_i)^3 for t in [x_i, x_{i+1}]
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    if n < 3:
        raise ValueError("Need at least 3 points for cubic spline.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing.")

    h = np.diff(x)
    # set up tridiagonal system for c
    alpha = np.zeros(n)
    alpha[1:-1] = (3 / h[1:]) * (y[2:] - y[1:-1]) - \
        (3 / h[:-1]) * (y[1:-1] - y[:-2])

    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)

    for i in range(1, n - 1):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    # natural boundary: c[0]=c[n-1]=0
    c = np.zeros(n)
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)
    a = y[:-1].copy()

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        if j < n - 1:
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (2 * c[j] + c[j + 1]) / 3
            d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    return {"x": x, "a": a, "b": b, "c": c[:-1], "d": d}


def _natural_cubic_spline_eval(coeffs: Dict[str, np.ndarray], x_new: np.ndarray) -> np.ndarray:
    x = coeffs["x"]
    a, b, c, d = coeffs["a"], coeffs["b"], coeffs["c"], coeffs["d"]
    x_new = np.asarray(x_new, float)

    # find interval indices
    idx = np.searchsorted(x, x_new, side="right") - 1
    idx = np.clip(idx, 0, x.size - 2)

    dx = x_new - x[idx]
    return a[idx] + b[idx] * dx + c[idx] * dx**2 + d[idx] * dx**3

# -------------------------
# Monotone cubic Hermite (Fritsch–Carlson)
# -------------------------


def _monotone_cubic_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    h = np.diff(x)
    delta = np.diff(y) / h

    m = np.zeros(n)
    m[0] = delta[0]
    m[-1] = delta[-1]

    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    # additional limiting to avoid overshoot
    for i in range(n - 1):
        if delta[i] == 0.0:
            m[i] = 0.0
            m[i + 1] = 0.0
        else:
            a = m[i] / delta[i]
            b = m[i + 1] / delta[i]
            s = a * a + b * b
            if s > 9.0:
                tau = 3.0 / np.sqrt(s)
                m[i] = tau * a * delta[i]
                m[i + 1] = tau * b * delta[i]
    return m


def _monotone_cubic_eval(x: np.ndarray, y: np.ndarray, m: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.asarray(m, float)
    x_new = np.asarray(x_new, float)

    idx = np.searchsorted(x, x_new, side="right") - 1
    idx = np.clip(idx, 0, x.size - 2)

    h = x[idx + 1] - x[idx]
    s = (x_new - x[idx]) / h

    h00 = 2*s**3 - 3*s**2 + 1
    h10 = s**3 - 2*s**2 + s
    h01 = -2*s**3 + 3*s**2
    h11 = s**3 - s**2

    return h00*y[idx] + h10*h*m[idx] + h01*y[idx+1] + h11*h*m[idx+1]

# -------------------------
# Main DF interpolation API
# -------------------------

def interpolate_discount_curve(
    t_pillars: np.ndarray,
    df_pillars: np.ndarray,
    method: InterpMethod = "log_linear_df",
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build an interpolated discount curve DF(t) from pillar times and pillar DFs.

    Methods:
      - linear_df          : linear interpolation on DF (can break positivity if used carelessly)
      - log_linear_df      : linear interpolation on log DF (guarantees DF>0)
      - linear_zero        : linear interpolation on zero rates z(t); DF=exp(-z(t)t)
      - cubic_zero         : natural cubic spline on zero rates (smooth but can overshoot)
      - monotone_cubic_zero: monotone cubic on zero rates (shape-preserving, less overshoot)
    """
    t = np.asarray(t_pillars, float)
    df = np.asarray(df_pillars, float)

    if t.ndim != 1 or df.ndim != 1 or t.size != df.size:
        raise ValueError(
            "t_pillars and df_pillars must be 1D and same length.")
    if np.any(t <= 0):
        raise ValueError(
            "Use pillar times strictly > 0 for these interpolators (avoid t=0 in zero-rate transforms).")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t_pillars must be strictly increasing.")
    if np.any(df <= 0):
        raise ValueError("df_pillars must be strictly positive.")

    if method == "linear_df":
        def f(x_new: np.ndarray) -> np.ndarray:
            x_new = np.asarray(x_new, float)
            return np.interp(x_new, t, df)
        return f

    if method == "log_linear_df":
        logdf = np.log(df)

        def f(x_new: np.ndarray) -> np.ndarray:
            x_new = np.asarray(x_new, float)
            return np.exp(np.interp(x_new, t, logdf))
        return f

    # zero-rate based methods
    z = zeros_from_dfs(t, df)

    if method == "linear_zero":
        def f(x_new: np.ndarray) -> np.ndarray:
            x_new = np.asarray(x_new, float)
            z_new = np.interp(x_new, t, z)
            return np.exp(-z_new * x_new)
        return f

    if method == "cubic_zero":
        coeffs = _natural_cubic_spline_coeffs(t, z)

        def f(x_new: np.ndarray) -> np.ndarray:
            x_new = np.asarray(x_new, float)
            z_new = _natural_cubic_spline_eval(coeffs, x_new)
            return np.exp(-z_new * x_new)
        return f

    if method == "monotone_cubic_zero":
        m = _monotone_cubic_slopes(t, z)

        def f(x_new: np.ndarray) -> np.ndarray:
            x_new = np.asarray(x_new, float)
            z_new = _monotone_cubic_eval(t, z, m, x_new)
            return np.exp(-z_new * x_new)
        return f

    raise ValueError(f"Unknown method: {method}")


def toy_df_pillars_from_par_swaps(t_pillars: np.ndarray, par_swap_rates: np.ndarray) -> np.ndarray:
    """
    Toy transform used ONLY for the interpolation demo plots:
      Treat par swap rate as a continuous-compounded zero at that maturity:
        DF(T) = exp(-S(T)*T)

    This is not a proper swap bootstrap (it ignores coupon structure),
    but it is good enough to demonstrate how *interpolation* choices change
    zeros/forwards while matching the same pillars.
    """
    t = np.asarray(t_pillars, float)
    s = np.asarray(par_swap_rates, float)
    return np.exp(-s * t)
