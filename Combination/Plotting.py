"""
Plotting helpers for portfolio price and structure inspection.

Currently supports:
  - Line plots of all stocks in a DAX/SDAX portfolio (excluding the index)
  - Line plots of the corresponding DAX/SDAX indices
  - Overview plots describing the structure of a portfolio (prices, returns,
    dispersion, and correlation) based on DataStorage Excel files.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Tuple, Union, List, Optional

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PLOTS_DIR = BASE_DIR / "Results" / "plots"
DEFAULT_STORAGE_DIR = BASE_DIR / "DataStorage"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

import matplotlib
from ConfigManager import ConfigManager

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREFERRED_PRICE_FIELDS: List[str] = ["TRDPRC_1", "OPEN_PRC", "CLOSE_PRC"]


def _parse_column_label(label: Any) -> Tuple[str, Optional[str]]:
    """
    Extract instrument and field from a column or sheet label.

    Accepts strings like \"('ABC.DE', 'TRDPRC_1')\" or simple names.
    """
    text = str(label)
    match = re.search(r"\('?(?P<instr>[^',]+)'?,\s*'?(?P<field>[^')]+)'?\)", text)
    if match:
        return match.group("instr").strip(), match.group("field").strip()

    if "," in text:
        parts = [p.strip(" _()") for p in text.split(",")]
        if len(parts) >= 2:
            return parts[0], parts[1]

    clean = text.strip(" _")
    return clean, None


def _load_prices_from_excel(
    excel_path: Path,
    preferred_fields: Optional[List[str]] = None,
    exclude_fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load a combined price DataFrame from an Excel file created by exceltextwriter.

    Picks the best available field per instrument based on `preferred_fields`
    priority (default: TRDPRC_1, OPEN_PRC, CLOSE_PRC).
    """
    preferred_fields = preferred_fields or PREFERRED_PRICE_FIELDS
    exclude_fields = set((exclude_fields or []))
    if not excel_path.exists():
        return pd.DataFrame()

    xls = pd.ExcelFile(excel_path)
    series_by_instrument: Dict[str, Tuple[int, pd.Series]] = {}

    for sheet in xls.sheet_names:
        df_sheet = pd.read_excel(xls, sheet_name=sheet)
        if df_sheet.empty:
            continue

        date_col = next((c for c in df_sheet.columns if str(c).lower() == "date"), None)
        if date_col is None:
            continue

        value_cols = [c for c in df_sheet.columns if c != date_col]
        if not value_cols:
            continue

        value_col = value_cols[-1]
        label_text = str(value_col)
        instrument, field = _parse_column_label(value_col)

        if field and field in exclude_fields:
            continue
        if not field and any(x.upper() in label_text.upper() for x in exclude_fields):
            continue

        if field and field not in preferred_fields:
            continue

        priority = preferred_fields.index(field) if field in preferred_fields else len(preferred_fields)
        frame = df_sheet[[date_col, value_col]].copy()
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame = frame.dropna(subset=[date_col])
        series = frame.set_index(date_col)[value_col]
        series.name = instrument or str(sheet)

        existing = series_by_instrument.get(series.name)
        if existing and existing[0] <= priority:
            continue
        series_by_instrument[series.name] = (priority, series)

    if not series_by_instrument:
        return pd.DataFrame()

    combined = pd.concat([item[1] for item in series_by_instrument.values()], axis=1)
    combined.sort_index(inplace=True)
    return combined


def _build_portfolio_price_series(
    portfolio_name: str,
    period_type: str = "daily",
    config_path: Union[str, Path] = BASE_DIR / "config.yaml",
    exclude_symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a price DataFrame for all stocks in a given portfolio.

    The returned DataFrame contains one column per stock in the configured
    universe (e.g. all DAX or all SDAX stocks we consider), using the best
    available price field per instrument. The index level itself is not
    included, since it is not part of the universe.
    """
    config = ConfigManager(str(config_path))
    universe = config.get(f"data.portfolios.{portfolio_name}.universe", [])
    if not universe:
        return pd.DataFrame()

    excel_path = DEFAULT_STORAGE_DIR / f"{portfolio_name}_{period_type}.xlsx"
    prices_df = _load_prices_from_excel(excel_path)
    if prices_df.empty:
        return pd.DataFrame()

    instruments = [inst for inst in universe if inst in prices_df.columns]
    if exclude_symbols:
        exclude_set = set(exclude_symbols)
        instruments = [inst for inst in instruments if inst not in exclude_set]
    if not instruments:
        return pd.DataFrame()

    return prices_df[instruments]


def _build_index_price_series(
    portfolio_name: str,
    period_type: str = "daily",
    config_path: Union[str, Path] = BASE_DIR / "config.yaml",
) -> pd.Series:
    """
    Build the price series for the index associated with a given portfolio.

    Example:
        - portfolio_name=\"dax\"  -> index \".GDAXI\"
        - portfolio_name=\"sdax\" -> index \".SDAXI\"
    """
    config = ConfigManager(str(config_path))
    index_symbol = config.get(f"data.portfolios.{portfolio_name}.index")
    if not index_symbol:
        return pd.Series(dtype=float)

    excel_path = DEFAULT_STORAGE_DIR / f"{portfolio_name}_{period_type}.xlsx"
    prices_df = _load_prices_from_excel(excel_path, preferred_fields=["TRDPRC_1"])
    if prices_df.empty:
        return pd.Series(dtype=float)

    candidate_cols = [c for c in prices_df.columns if index_symbol in str(c)]
    if not candidate_cols:
        return pd.Series(dtype=float)

    series = prices_df[candidate_cols[0]].copy()
    series.name = index_symbol
    return series


def plot_portfolio_price_line(
    portfolio_name: str,
    period_type: str = "daily",
    output_dir: Union[str, Path] = DEFAULT_PLOTS_DIR / "portfolio_line",
    exclude_symbols: Optional[List[str]] = None,
) -> Optional[Path]:
    """
    Create a line plot for all portfolio stocks' price series.

    Example:
        - plot_portfolio_price_line("dax")                         # all DAX stocks
        - plot_portfolio_price_line("sdax")                        # all SDAX stocks
        - plot_portfolio_price_line("dax", exclude_symbols=["RHMG.DE"])  # DAX excl. Rheinmetall
    """
    prices_df = _build_portfolio_price_series(
        portfolio_name,
        period_type=period_type,
        exclude_symbols=exclude_symbols,
    )
    if prices_df.empty:
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    prices_df.plot(ax=ax, linewidth=1)
    ax.set_title(f"{portfolio_name.upper()} portfolio stocks - prices ({period_type})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    if len(prices_df.columns) > 10:
        ax.legend().remove()
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()

    suffix = ""
    if exclude_symbols:
        cleaned = "_".join(sym.replace(".", "") for sym in sorted(exclude_symbols))
        suffix = f"_excl_{cleaned}"

    out_file = out_dir / f"{portfolio_name}_{period_type}_portfolio_line{suffix}.png"
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def plot_index_price_line(
    portfolio_name: str,
    period_type: str = "daily",
    output_dir: Union[str, Path] = DEFAULT_PLOTS_DIR / "index_line",
) -> Optional[Path]:
    """
    Create a line plot for the index price associated with a portfolio.

    Example:
        - plot_index_price_line("dax")   # DAX index (.GDAXI)
        - plot_index_price_line("sdax")  # SDAX index (.SDAXI)
    """
    series = _build_index_price_series(portfolio_name, period_type=period_type)
    if series.empty:
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series.index, series.values, label=series.name)
    ax.set_title(f"{series.name} index price ({period_type})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()

    out_file = out_dir / f"{portfolio_name}_{period_type}_index_line.png"
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def plot_portfolio_structure_overview(
    portfolio_name: str,
    period_type: str = "daily",
    output_dir: Union[str, Path] = DEFAULT_PLOTS_DIR / "portfolio_structure",
) -> Optional[Path]:
    """
    Create an overview figure that helps describe the data structure
    of a portfolio:

      - Equal-weight portfolio price over time
      - Histogram of equal-weight log returns
      - Boxplot of per-stock log returns
      - Correlation heatmap of per-stock log returns
    """
    prices_df = _build_portfolio_price_series(portfolio_name, period_type=period_type)
    if prices_df.empty:
        return None

    returns_df = np.log(prices_df / prices_df.shift(1)).dropna(how="all")
    if returns_df.empty:
        return None

    ew_price = prices_df.mean(axis=1)
    ew_returns = np.log(ew_price / ew_price.shift(1)).dropna()

    corr = returns_df.corr()

    base_dir = Path(output_dir)
    out_dir = base_dir / period_type
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.0))

    # Top-left: equal-weight portfolio price
    ax_price = axes[0, 0]
    ax_price.plot(ew_price.index, ew_price.values)
    ax_price.set_title(f"{portfolio_name.upper()} portfolio (equal-weight price, {period_type})")
    ax_price.set_xlabel("Date")
    ax_price.set_ylabel("Price")

    # Top-right: histogram of equal-weight log returns
    ax_hist = axes[0, 1]
    ax_hist.hist(ew_returns.values, bins=30, edgecolor="black")
    ax_hist.set_title(f"{portfolio_name.upper()} portfolio log returns (histogram)")
    ax_hist.set_xlabel("Log return")
    ax_hist.set_ylabel("Frequency")

    # Bottom-left: boxplot of per-stock log returns
    ax_box = axes[1, 0]
    ax_box.boxplot(
        [returns_df[col].dropna().values for col in returns_df.columns],
        labels=returns_df.columns,
        vert=True,
        patch_artist=True,
    )
    ax_box.set_title("Per-stock log return dispersion")
    ax_box.set_ylabel("Log return")
    ax_box.tick_params(axis="x", rotation=45)

    # Bottom-right: correlation heatmap of per-stock log returns
    ax_heat = axes[1, 1]
    im = ax_heat.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax_heat.set_title("Correlation of per-stock log returns")
    ax_heat.set_xticks(range(len(corr.columns)))
    ax_heat.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax_heat.set_yticks(range(len(corr.index)))
    ax_heat.set_yticklabels(corr.index)
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    fig.tight_layout()

    overview_file = out_dir / f"{portfolio_name}_{period_type}_structure_overview.png"
    fig.savefig(overview_file)
    plt.close(fig)

    # Additionally: save each component as its own figure

    # 1) Equal-weight portfolio price
    fig_p, ax_p = plt.subplots(figsize=(8.0, 4.0))
    ax_p.plot(ew_price.index, ew_price.values)
    ax_p.set_title(f"{portfolio_name.upper()} portfolio (equal-weight price, {period_type})")
    ax_p.set_xlabel("Date")
    ax_p.set_ylabel("Price")
    fig_p.tight_layout()
    price_file = out_dir / f"{portfolio_name}_{period_type}_structure_price.png"
    fig_p.savefig(price_file)
    plt.close(fig_p)

    # 2) Histogram of equal-weight log returns
    fig_h, ax_h = plt.subplots(figsize=(6.0, 4.0))
    ax_h.hist(ew_returns.values, bins=30, edgecolor="black")
    ax_h.set_title(f"{portfolio_name.upper()} portfolio log returns (histogram)")
    ax_h.set_xlabel("Log return")
    ax_h.set_ylabel("Frequency")
    fig_h.tight_layout()
    hist_file = out_dir / f"{portfolio_name}_{period_type}_structure_ew_returns_hist.png"
    fig_h.savefig(hist_file)
    plt.close(fig_h)

    # 3) Boxplot of per-stock log returns
    fig_b, ax_b = plt.subplots(figsize=(10.0, 4.0))
    ax_b.boxplot(
        [returns_df[col].dropna().values for col in returns_df.columns],
        labels=returns_df.columns,
        vert=True,
        patch_artist=True,
    )
    ax_b.set_title("Per-stock log return dispersion")
    ax_b.set_ylabel("Log return")
    ax_b.tick_params(axis="x", rotation=45)
    fig_b.tight_layout()
    box_file = out_dir / f"{portfolio_name}_{period_type}_structure_stock_returns_box.png"
    fig_b.savefig(box_file)
    plt.close(fig_b)

    # 4) Correlation heatmap of per-stock log returns
    fig_c, ax_c = plt.subplots(figsize=(8.0, 6.0))
    im2 = ax_c.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax_c.set_title("Correlation of per-stock log returns")
    ax_c.set_xticks(range(len(corr.columns)))
    ax_c.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax_c.set_yticks(range(len(corr.index)))
    ax_c.set_yticklabels(corr.index)
    fig_c.colorbar(im2, ax=ax_c, fraction=0.046, pad=0.04)
    fig_c.tight_layout()
    corr_file = out_dir / f"{portfolio_name}_{period_type}_structure_corr_heatmap.png"
    fig_c.savefig(corr_file)
    plt.close(fig_c)

    return overview_file
