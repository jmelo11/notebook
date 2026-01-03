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
    criterion: nn.Module = nn.MSELoss(),
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

    return criterion(preds, obs_rates)


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
    criterion: nn.Module = nn.MSELoss(),
    tail_weight: float = 1.0,
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

    dfs = model.discounts(t_tail)
    dfs = dfs.squeeze(-1) if dfs.ndim == 2 else dfs

    # Tail forwards
    dt = t_tail[1:] - t_tail[:-1]
    fwds = -(torch.log(dfs[1:]) - torch.log(dfs[:-1])) / dt

    if mode == "flat":
        tail_pen = (fwds - fwds.mean()).pow(2).mean()
    elif mode == "ufr":
        if ufr is None:
            raise ValueError("ufr must be provided when mode='ufr'")
        target = torch.full_like(fwds, float(ufr))
        tail_pen = criterion(fwds, target)
    else:
        raise ValueError("mode must be 'flat' or 'ufr'")

    return tail_weight * tail_pen
