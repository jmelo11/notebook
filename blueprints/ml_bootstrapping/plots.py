from utils import (
    interpolate_discount_curve,
    toy_df_pillars_from_par_swaps,
    zeros_from_dfs,
    instantaneous_fwd_from_df_grid,
)
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import Swap, CurveModel, str_tenor_to_days


_DEFAULT_TEMPLATE = "plotly_white"
_DEFAULT_FONT = dict(family="Arial", size=12)
_PLOT_WIDTH = None
_PLOT_HEIGHT = 500

def apply_style(
    fig: go.Figure,
    title: str,
    x_title: str,
    y_title: str,
    *,
    height: Optional[int] = None,
    width: Optional[int] = None,
    compact: bool = False,
    legend_orientation: str = "h",
    legend_y: Optional[float] = None,
) -> go.Figure:
    """Default styling + a compact mode that plays nicer in narrow columns."""
    font = dict(_DEFAULT_FONT)
    if compact:
        font["size"] = 11

    if legend_y is None:
        legend_y = -0.18 if compact else -0.15

    fig.update_layout(
        template=_DEFAULT_TEMPLATE,
        font=font,
        title=dict(text=title, xanchor="left"),
        legend=dict(
            orientation=legend_orientation,
            yanchor="top",
            y=legend_y,
            xanchor="left",
            x=0.0,
        ),
        autosize=True,
        margin=dict(l=55, r=25, t=60, b=70) if compact else dict(l=60, r=30, t=70, b=70),
        width=_PLOT_WIDTH if width is None else width,
        height=_PLOT_HEIGHT if height is None else height,
    )
    fig.update_xaxes(title=x_title, showgrid=True, zeroline=False, automargin=True)
    fig.update_yaxes(title=y_title, showgrid=True, zeroline=False, automargin=True)
    return fig


def _nice_tick_subset(vals: np.ndarray, labels: List[str], max_ticks: int = 10):
    """
    Downsample ticks (keep order) so narrow columns don't get unreadable.
    """
    n = len(vals)
    if n <= max_ticks:
        return vals.tolist(), labels
    step = int(np.ceil(n / max_ticks))
    idx = np.arange(0, n, step)
    # Ensure last tick is included
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return vals[idx].tolist(), [labels[i] for i in idx]



def _device_dtype_from_model(model: torch.nn.Module) -> Tuple[torch.device, torch.dtype]:
    p = next(model.parameters(), None)
    if p is None:
        return torch.device("cpu"), torch.float64
    return p.device, p.dtype


def _tenors_tensor_from_df(curves_df: pd.DataFrame, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tenors_years = np.array([str_tenor_to_days(c)
                            for c in curves_df.columns], dtype=float)/360
    return torch.tensor(tenors_years, device=device, dtype=dtype)


def _sample_dates(curves_df: pd.DataFrame, n: int = 3, seed: Optional[int] = None) -> List[pd.Timestamp]:
    rng = np.random.default_rng(seed)
    idx = curves_df.index.to_numpy()
    if len(idx) < n:
        raise ValueError(f"Need at least {n} rows to sample {n} dates.")
    pick = rng.choice(idx, size=n, replace=False)
    pick = np.sort(pick)
    return [pd.Timestamp(x) for x in pick]


def _discount_curve_on_grid(model, rates_row: torch.Tensor, tenors: torch.Tensor, t_grid: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    model.set_curve(rates_row, tenors)
    with torch.no_grad():
        dfs = model.discounts(t_grid)
        dfs = dfs.squeeze(-1) if dfs.ndim == 2 else dfs
    return t_grid.detach().cpu().numpy(), dfs.detach().cpu().numpy()


def _instantaneous_forward_from_df(t: torch.Tensor, df: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    logdf = torch.log(df)

    t_np = t.detach().cpu().numpy()
    f = torch.empty((t.numel() - 2,), dtype=t.dtype, device=t.device)

    for i in range(1, t.numel() - 1):
        dt = t[i + 1] - t[i - 1]
        f[i - 1] = -(logdf[i + 1] - logdf[i - 1]) / dt

    return t_np[1:-1], f.detach().cpu().numpy()


def _implied_swap_rates(model: CurveModel, rates_row: torch.Tensor, tenors: torch.Tensor, swap_labels: List[str], mask) -> np.ndarray:
    swaps = [Swap(lbl) for lbl in swap_labels]
    model.set_curve(rates_row, tenors, mask=mask)
    with torch.no_grad():
        preds = torch.stack([s.fair_rate(model) for s in swaps])
    return preds.detach().cpu().numpy()


def plot_discount_factors(
    model,
    curves_df: pd.DataFrame,
    n_samples: int = 3,
    seed: Optional[int] = None,
    t_max: Optional[float] = None,
    n_grid: int = 301,
) -> go.Figure:
    device, dtype = _device_dtype_from_model(model)
    tenors = _tenors_tensor_from_df(curves_df, device=device, dtype=dtype)

    dates = _sample_dates(curves_df, n=n_samples, seed=seed)
    max_t = float(np.max(tenors.detach().cpu().numpy())
                  ) if t_max is None else float(t_max)

    t_grid = torch.linspace(0.0, max_t, n_grid, device=device, dtype=dtype)
    t_grid[0] = torch.tensor(0.0, device=device, dtype=dtype)

    fig = go.Figure()
    for d in dates:
        rates_row = torch.tensor(
            curves_df.loc[d].values, device=device, dtype=dtype)
        t_np, df_np = _discount_curve_on_grid(model, rates_row, tenors, t_grid)
        fig.add_trace(go.Scatter(x=t_np, y=df_np,
                      mode="lines", name=str(d.date())))

    apply_style(fig, "Discount factors (DF)",
                 "Maturity (years)", "DF(t)")
    return fig


def plot_instant_fwds(
    model: CurveModel,
    curves_df: pd.DataFrame,
    n_samples: int = 3,
    seed: Optional[int] = None,
    t_max: Optional[float] = None,
    n_grid: int = 601,
    dates: Optional[List[pd.Timestamp]] = None,
    show_tenor_labels: bool = True,
) -> go.Figure:
    device, dtype = _device_dtype_from_model(model)
    tenors = _tenors_tensor_from_df(curves_df, device=device, dtype=dtype)

    if dates is None:
        dates = _sample_dates(curves_df, n=n_samples, seed=seed)

    pillar_labels = list(curves_df.columns)
    pillar_years = np.array([str_tenor_to_days(c) for c in pillar_labels], dtype=float) / 360.0
    last_pillar = float(np.max(pillar_years))

    max_t = last_pillar if t_max is None else float(t_max)

    t_grid = torch.linspace(1e-6, max_t, n_grid, device=device, dtype=dtype)

    fig = go.Figure()
    for d in dates:
        rates_row = torch.tensor(
            curves_df.loc[d].values, device=device, dtype=dtype)
        model.set_curve(rates_row, tenors)
        with torch.no_grad():
            df = model.discounts(t_grid)
            df = df.squeeze(-1) if df.ndim == 2 else df

        t_f, f = _instantaneous_forward_from_df(t_grid, df)
        fig.add_trace(go.Scatter(x=t_f, y=f, mode="lines", name=str(d.date())))

    apply_style(fig, "Instantaneous forward rates",
                 "Maturity (years)", "f(t)")
    fig.update_yaxes(tickformat=".2%")

    for x in pillar_years:
        fig.add_vline(
            x=float(x),
            line_width=1,
            line_color="rgba(0,0,0,0.18)",
            layer="above",
        )

    if show_tenor_labels:
        fig.update_xaxes(
            tickmode="array",
            tickvals=pillar_years.tolist(),
            ticktext=pillar_labels,
            tickangle=-90,
            tickfont=dict(size=12),
            automargin=True,
        )

    if max_t > last_pillar + 1e-12:
        fig.add_vrect(
            x0=last_pillar, x1=max_t,
            fillcolor="LightSalmon",
            opacity=0.22,
            layer="below",
            line_width=0,
            annotation_text="Extrapolation region",
            annotation_position="top right",
        )

        fig.add_vline(
            x=last_pillar,
            line_width=2,
            line_color="rgba(255,100,50,0.65)",
            layer="above",
        )

    return fig

def plot_market_swap_rates(
    market_rates_df: pd.DataFrame,
    *,
    dates: Optional[List[pd.Timestamp]] = None,
    n_samples: int = 3,
    seed: Optional[int] = 1,
    max_xticks: int = 10,
    column_layout: bool = True,
) -> Tuple[go.Figure, List[pd.Timestamp]]:
    """
    Plot market par swap rates for particular dates (sampled or provided),
    designed to remain readable in a narrow column.
    """
    if dates is None:
        dates = _sample_dates(market_rates_df, n=n_samples, seed=seed)
    else:
        dates = [pd.Timestamp(d) for d in dates]

    pillar_labels = list(market_rates_df.columns)
    tenors_years = np.array([str_tenor_to_days(c) for c in pillar_labels], dtype=float) / 360.0

    tickvals, ticktext = _nice_tick_subset(tenors_years, pillar_labels, max_ticks=max_xticks)

    fig = go.Figure()
    for d in dates:
        row = market_rates_df.loc[d].astype(float)
        fig.add_trace(
            go.Scatter(
                x=tenors_years,
                y=row.values,
                mode="lines+markers",
                name=str(pd.Timestamp(d).date()),
            )
        )

    apply_style(
        fig,
        title="Market Swap Rates - Camara/Fix",
        x_title="Maturity (years)",
        y_title="Swap Rate",
        compact=column_layout,
        height=420 if column_layout else None,
    )
    fig.update_yaxes(tickformat=".2%")
    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        tickangle=-90,
        tickfont=dict(size=11 if column_layout else 12),
        automargin=True,
    )
    return fig, dates


def plot_swap_rates(
    model,
    curves_df: pd.DataFrame,
    n_samples: int = 3,
    seed: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
    *,
    column_layout: bool = True,
) -> Tuple[go.Figure, List[pd.Timestamp]]:
    device, dtype = _device_dtype_from_model(model)
    tenors = _tenors_tensor_from_df(curves_df, device=device, dtype=dtype)

    dates = _sample_dates(curves_df, n=n_samples, seed=seed)
    swap_labels = list(curves_df.columns)
    x_cat = swap_labels  # categorical axis

    if column_layout:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.42, 0.58],
            vertical_spacing=0.14,
            subplot_titles=["Model error (bp)", "Market vs model par swap rates"],
        )
        err_row, err_col = 1, 1
        fit_row, fit_col = 2, 1
    else:
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.38, 0.62],
            horizontal_spacing=0.08,
            subplot_titles=["Model error (bp)", "Market vs model par swap rates"],
        )
        err_row, err_col = 1, 1
        fit_row, fit_col = 1, 2

    colorway = go.Figure().layout.colorway or [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
    ]

    date_names = [str(d.date()) for d in dates]
    date_colors = {dn: colorway[i % len(colorway)] for i, dn in enumerate(date_names)}

    for d in dates:
        obs = curves_df.loc[d].values.astype(float)
        rates_row = torch.tensor(obs, device=device, dtype=dtype)
        pred = _implied_swap_rates(model, rates_row, tenors, swap_labels, mask)

        err_bp = (pred - obs) * 1e4  # bp
        date_name = str(d.date())
        c = date_colors[date_name]

        # Error panel
        fig.add_trace(
            go.Bar(
                x=x_cat,
                y=err_bp,
                name=f"{date_name}",
                legendgroup=date_name,
                marker=dict(color=c),
            ),
            row=err_row, col=err_col,
        )

        # Fit panel: market + model
        fig.add_trace(
            go.Scatter(
                x=x_cat, y=obs,
                mode="markers+lines",
                name=f"{date_name} (Market)",
                legendgroup=date_name,
                showlegend=False,
                marker=dict(color=c),
                line=dict(color=c),
            ),
            row=fit_row, col=fit_col,
        )
        fig.add_trace(
            go.Scatter(
                x=x_cat, y=pred,
                mode="lines",
                name=f"{date_name} (Model)",
                legendgroup=date_name,
                showlegend=False,
                line=dict(color=c, dash="dash"),
                opacity=0.65,
            ),
            row=fit_row, col=fit_col,
        )

    # Axes formatting
    # Only show x tick labels on bottom subplot when stacked
    if column_layout:
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        x_tick_row = 2
        height = max(680, 240 + 220 * 2)  # stable “card” size
        legend_y = -0.20
    else:
        x_tick_row = 1
        height = _PLOT_HEIGHT
        legend_y = -0.20

    for (r, c) in [(err_row, err_col), (fit_row, fit_col)]:
        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=swap_labels,
            tickangle=-90,
            tickfont=dict(size=11),
            automargin=True,
            title_text="Tenor" if (r == x_tick_row) else None,
            row=r, col=c,
        )

    fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.35)", row=err_row, col=err_col)

    fig.update_yaxes(title_text="Error (bp)", row=err_row, col=err_col, zeroline=False)
    fig.update_yaxes(title_text="Par swap rate", row=fit_row, col=fit_col, tickformat=".2%")

    fig.update_layout(
        template=_DEFAULT_TEMPLATE,
        font=dict(_DEFAULT_FONT, size=11 if column_layout else _DEFAULT_FONT["size"]),
        title=dict(text="Par swap fit and errors", xanchor="left"),
        barmode="group",
        bargap=0.15,
        bargroupgap=0.0,
        legend=dict(orientation="h", yanchor="top", y=legend_y, xanchor="left", x=0.0),
        margin=dict(l=55, r=25, t=60, b=90),
        width=_PLOT_WIDTH,
        height=height,
    )

    return fig, dates


def _jacobian_zero_wrt_rates(
    model,
    rates_row: torch.Tensor,
    tenors: torch.Tensor,
    t_grid: torch.Tensor,
    bump_bp: float = 1.0,
    eps_df: float = 1e-12,
) -> np.ndarray:
    device, dtype = tenors.device, tenors.dtype

    r = rates_row.detach().clone().to(device=device, dtype=dtype).requires_grad_(True)
    model.set_curve(r, tenors)

    t = t_grid.to(device=device, dtype=dtype).reshape(-1)

    df = model.discounts(t).squeeze(-1)
    df = torch.clamp(df, min=eps_df, max=1.0)

    z = -torch.log(df) / t  

    n_t, n_in = z.numel(), r.numel()
    J = torch.empty((n_t, n_in), device=device, dtype=dtype)

    for k in range(n_t):
        (g,) = torch.autograd.grad(z[k], r, retain_graph=True)
        J[k, :] = g 

    bump = torch.tensor(bump_bp * 1e-4, device=device, dtype=dtype)
    J_bp = J * bump * 1e4  

    return J_bp.detach().cpu().numpy()


def plot_jacobian(
    model,
    curves_df: pd.DataFrame,
    n_samples: int = 3,
    seed: Optional[int] = None,
    t_grid: Optional[np.ndarray] = None,
    dates: Optional[List[pd.Timestamp]] = None,
    *,
    column_layout: bool = True,
) -> go.Figure:
    device, dtype = _device_dtype_from_model(model)
    tenors = _tenors_tensor_from_df(curves_df, device=device, dtype=dtype)

    if dates is None:
        dates = _sample_dates(curves_df, n=n_samples, seed=seed)
    n = len(dates)

    pillar_labels = list(curves_df.columns)
    x_cat = pillar_labels

    if t_grid is None:
        t_vals = tenors.detach().cpu().numpy()
    else:
        t_vals = np.asarray(t_grid, dtype=float)

    t_vals = np.maximum(t_vals, 1/360)
    t_torch = torch.tensor(t_vals, device=device, dtype=dtype)

    Js, all_vals = [], []
    for d in dates:
        rates_row = torch.tensor(curves_df.loc[d].values, device=device, dtype=dtype)
        J = _jacobian_zero_wrt_rates(model, rates_row, tenors, t_torch, bump_bp=1.0)
        Js.append(J)
        all_vals.append(J.reshape(-1))
    all_vals = np.concatenate(all_vals)

    zlim = float(np.nanmax(np.abs(all_vals)))
    if not np.isfinite(zlim) or zlim == 0.0:
        zlim = 1.0

    if column_layout:
        fig = make_subplots(
            rows=n, cols=1,
            subplot_titles=[str(d.date()) for d in dates],
            vertical_spacing=0.10,
        )
        # Column-friendly height
        height = max(340, 220 * n + 110)
    else:
        fig = make_subplots(
            rows=1, cols=n,
            subplot_titles=[str(d.date()) for d in dates],
            horizontal_spacing=0.06,
        )
        height = _PLOT_HEIGHT

    for i, (d, J) in enumerate(zip(dates, Js), start=1):
        r, c = (i, 1) if column_layout else (1, i)

        fig.add_trace(
            go.Heatmap(
                x=x_cat,
                y=t_vals,
                z=J,
                zmin=-zlim,
                zmax=+zlim,
                zmid=0.0,
                coloraxis="coloraxis",
                zsmooth="best",
                hovertemplate=(
                    "t=%{y:.3f}y<br>"
                    "pillar=%{x}<br>"
                    "Δz(t)=%{z:.3f} bp (per +1bp bump)<extra></extra>"
                ),
            ),
            row=r, col=c
        )

        # Hide x tick labels for all but the bottom panel when stacked
        show_ticks = (not column_layout) or (i == n)

        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=pillar_labels,
            tickangle=-90,
            tickfont=dict(size=11),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            automargin=True,
            title_text="Input tenor" if show_ticks else None,
            showticklabels=show_ticks,
            row=r, col=c,
        )
        fig.update_yaxes(
            title_text="Output maturity t (years)",
            row=r, col=c
        )

    fig.update_layout(
        template=_DEFAULT_TEMPLATE,
        font=dict(_DEFAULT_FONT, size=11 if column_layout else _DEFAULT_FONT["size"]),
        margin=dict(l=60, r=35, t=60, b=70),
        title="Jacobian: Δ zero-rate (bp) for +1bp bumps to pillar quotes",
        coloraxis=dict(
            colorscale="RdBu",
            cmin=-zlim,
            cmax=+zlim,
            cmid=0.0,
            colorbar=dict(title="Δz(t) (bp)"),
        ),
        width=_PLOT_WIDTH,
        height=height,
    )
    return fig


def plot_zero_sensitivity_surface_one_day(model, curves_df, date, t_vals):
    device, dtype = _device_dtype_from_model(model)
    tenors = _tenors_tensor_from_df(curves_df, device=device, dtype=dtype)

    rates_row = torch.tensor(
        curves_df.loc[date].values, device=device, dtype=dtype)
    t_torch = torch.tensor(t_vals, device=device, dtype=dtype)

    J = _jacobian_zero_wrt_rates(
        model, rates_row, tenors, t_torch, bump_bp=1.0)  

    x_out = np.asarray(t_vals, float)                    
    y_in = tenors.detach().cpu().numpy()

    Z = J.T  

    fig = go.Figure(data=go.Surface(x=x_out, y=y_in, z=Z,
                    colorbar=dict(title="Δzero (bp)")))
    fig.update_layout(
        title=f"Δ zero-rate (bp) for +1bp bump — {pd.Timestamp(date).date()}",
        scene=dict(
            xaxis_title="Output maturity t (years)",
            yaxis_title="Input pillar maturity (years)",
            zaxis_title="Δz(t) (bp)",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        width=_PLOT_WIDTH,
        height=_PLOT_HEIGHT,
    )
    return fig


def influence_share(J: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    A = np.abs(J)
    denom = A.sum(axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return A / denom


def signed_influence_share(J: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    denom = np.abs(J).sum(axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return J / denom


def plot_influence_map(
    model,
    curves_df: pd.DataFrame,
    n_samples: int = 3,
    seed: Optional[int] = None,
    t_grid: Optional[np.ndarray] = None,
    bump_bp: float = 1.0,
    signed: bool = False,
):
   
    device, dtype = _device_dtype_from_model(model)
    tenors = _tenors_tensor_from_df(curves_df, device=device, dtype=dtype)
    dates = _sample_dates(curves_df, n=n_samples, seed=seed)
    pillar_labels = list(curves_df.columns)

    if t_grid is None:
        t_vals = np.linspace(
            1/360, float(tenors.max().detach().cpu().numpy()), 201)
    else:
        t_vals = np.asarray(t_grid, dtype=float)
        if np.any(t_vals <= 0):
            t_vals = np.maximum(t_vals, 1/360)

    t_torch = torch.tensor(t_vals, device=device, dtype=dtype)

    x_years = np.array([str_tenor_to_days(c)
                       for c in pillar_labels], dtype=float) / 360.0

    fig = make_subplots(
        rows=1, cols=n_samples,
        subplot_titles=[str(pd.Timestamp(d).date()) for d in dates],
        horizontal_spacing=0.06,
    )

    for col, d in enumerate(dates, start=1):
        rates_row = torch.tensor(
            curves_df.loc[d].values, device=device, dtype=dtype)
        J = _jacobian_zero_wrt_rates(model, rates_row, tenors, t_torch,
                                     bump_bp=bump_bp)  

        Z = signed_influence_share(J) if signed else influence_share(J)

        if signed:
            zmin, zmax = -1.0, 1.0
            cbar_title = "Signed share"
            hover = (
                "t=%{y:.3f}y<br>"
                "pillar=%{x:.3f}y<br>"
                "signed share=%{z:.4f}<extra></extra>"
            )
        else:
            zmin, zmax = 0.0, 1.0
            cbar_title = "Share of |sens|"
            hover = (
                "t=%{y:.3f}y<br>"
                "pillar=%{x:.3f}y<br>"
                "share=%{z:.4f}<extra></extra>"
            )

        fig.add_trace(
            go.Heatmap(
                x=x_years,
                y=t_vals,
                z=Z,
                zmin=zmin,
                zmax=zmax,
                zsmooth="best",
                coloraxis="coloraxis",
                hovertemplate=hover,
            ),
            row=1, col=col
        )
        fig.update_xaxes(
            title_text="Input pillar maturity (years)", row=1, col=col)
        fig.update_yaxes(
            title_text="Output maturity t (years)", row=1, col=col)

    title = "Influence map (who drives each maturity)"
    if signed:
        title += " — signed"
    else:
        title += " — absolute share"

    fig.update_layout(
        template=_DEFAULT_TEMPLATE,
        font=_DEFAULT_FONT,
        margin=dict(l=60, r=30, t=70, b=55),
        title=title,
        coloraxis=dict(colorbar=dict(title=cbar_title)),
    )

    return fig

def plot_interpolation_examples(
    curves_df: pd.DataFrame,
    date: Optional[pd.Timestamp] = None,
    methods: Optional[List[str]] = None,
    t_max: Optional[float] = None,
    n_grid: int = 800,
) -> go.Figure:
    """
    Demonstrates how different interpolation choices change DF/zero/inst-fwd
    while matching the SAME pillar points.

    Uses a toy transform DF(T)=exp(-S(T)T) from the chosen market curve
    (good for interpolation illustration; not a full swap bootstrap).
    """
    if date is None:
        date = curves_df.index[0]
    date = pd.Timestamp(date)

    pillar_labels = list(curves_df.columns)
    t_pillars = np.array([str_tenor_to_days(c) for c in pillar_labels], dtype=float) / 360.0
    s_pillars = curves_df.loc[date].values.astype(float)

    df_pillars = toy_df_pillars_from_par_swaps(t_pillars, s_pillars)

    if t_max is None:
        t_max = float(t_pillars.max())

    t_grid = np.linspace(1/360, float(t_max), int(n_grid))

    if methods is None:
        methods = ["log_linear_df", "linear_zero", "cubic_zero", "monotone_cubic_zero"]

    pretty = {
        "linear_df": "Linear on DF",
        "log_linear_df": "Log-linear on DF",
        "linear_zero": "Linear on zero",
        "cubic_zero": "Cubic spline on zero",
        "monotone_cubic_zero": "Monotone cubic on zero",
    }

    default_colorway = go.Figure().layout.colorway
    if default_colorway is None:
        default_colorway = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
                            "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]

    method_colors = {m: default_colorway[i % len(default_colorway)] for i, m in enumerate(methods)}

    # --- CHANGED: 3 rows x 1 col instead of 1 row x 3 cols ---
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=[
            "Discount factors DF(t)",
            "Zero rates z(t)",
            "Instantaneous forwards f(t)",
        ],
    )

    # Pillars on the DF panel (top)
    fig.add_trace(
        go.Scatter(
            x=t_pillars, y=df_pillars,
            mode="markers",
            name="Pillars",
            marker=dict(size=7, color="black"),
        ),
        row=1, col=1
    )

    for rank, m in enumerate(methods):
        curve = interpolate_discount_curve(t_pillars, df_pillars, method=m)
        df_grid = curve(t_grid)
        z_grid = zeros_from_dfs(t_grid, df_grid)
        f_grid = instantaneous_fwd_from_df_grid(t_grid, df_grid)

        name = pretty.get(m, m)
        c = method_colors[m]
        group = m

        # DF panel (row 1)
        fig.add_trace(
            go.Scatter(
                x=t_grid, y=df_grid,
                mode="lines",
                name=name,
                legendgroup=group,
                showlegend=True,
                legendrank=10 + rank,
                line=dict(color=c),
            ),
            row=1, col=1
        )

        # Zero panel (row 2)
        fig.add_trace(
            go.Scatter(
                x=t_grid, y=z_grid,
                mode="lines",
                name=name,
                legendgroup=group,
                showlegend=False,
                line=dict(color=c),
            ),
            row=2, col=1
        )

        # Fwd panel (row 3)
        fig.add_trace(
            go.Scatter(
                x=t_grid, y=f_grid,
                mode="lines",
                name=name,
                legendgroup=group,
                showlegend=False,
                line=dict(color=c),
            ),
            row=3, col=1
        )

    fig.update_layout(
        template=_DEFAULT_TEMPLATE,
        font=_DEFAULT_FONT,
        title="Interpolation choices",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,          # a bit tighter since figure is taller now
            xanchor="left",
            x=0.0,
            groupclick="togglegroup",
        ),
        margin=dict(l=60, r=30, t=70, b=80),
        height=820,           # column-friendly (taller)
        width=_PLOT_WIDTH,
    )

    # Shared x-axis: only label the bottom plot to reduce clutter
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="", row=2, col=1)
    fig.update_xaxes(title_text="Maturity (years)", row=3, col=1)

    fig.update_yaxes(title_text="DF(t)", row=1, col=1)
    fig.update_yaxes(title_text="z(t)", tickformat=".2%", row=2, col=1)
    fig.update_yaxes(title_text="f(t)", tickformat=".2%", row=3, col=1)

    return fig
