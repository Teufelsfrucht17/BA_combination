"""
Plotting helpers for DAX and SDAX raw data.

Generates price trend and return distribution plots from Excel files in DataStorage.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Dataprep import DataPrep

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PLOTS_DIR = BASE_DIR / "Results" / "plots" / "core"
DEFAULT_FEATURE_PLOTS_DIR = BASE_DIR / "Results" / "plots" / "features"
DEFAULT_STORAGE_DIR = BASE_DIR / "DataStorage"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

PREFERRED_PRICE_FIELDS = ["TRDPRC_1", "OPEN_PRC", "CLOSE_PRC"]
DEFAULT_MOMENTUM_WINDOWS = [5, 10, 20]

matplotlib.use("Agg")


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name) or "plot"


def _parse_column_label(label: Any) -> Tuple[str, Optional[str]]:
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
    preferred_fields: Optional[list[str]] = None,
    excluded_instruments: Optional[list[str]] = None,
    include_instruments: Optional[list[str]] = None,
) -> pd.DataFrame:
    preferred_fields = preferred_fields or PREFERRED_PRICE_FIELDS
    excluded = [e.upper() for e in (excluded_instruments or [])]
    include = [i.upper() for i in (include_instruments or [])]
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

        instr_upper = instrument.upper() if instrument else ""
        if include and not any(inc in instr_upper for inc in include):
            continue
        if excluded and any(excl in instr_upper for excl in excluded):
            continue

        if field and field not in preferred_fields:
            continue

        priority = preferred_fields.index(field) if field in preferred_fields else len(preferred_fields)
        frame = df_sheet[[date_col, value_col]].copy()
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame = frame.dropna(subset=[date_col])
        series = frame.set_index(date_col)[value_col]
        series.name = instrument or label_text

        existing = series_by_instrument.get(series.name)
        if existing and existing[0] <= priority:
            continue
        series_by_instrument[series.name] = (priority, series)

    if not series_by_instrument:
        return pd.DataFrame()

    combined = pd.concat([item[1] for item in series_by_instrument.values()], axis=1)
    combined.sort_index(inplace=True)
    return combined


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.drop_duplicates()
    if isinstance(cleaned.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        cleaned = cleaned.sort_index()
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        cleaned[numeric_cols] = cleaned[numeric_cols].apply(
            lambda col: col.clip(lower=col.quantile(0.01), upper=col.quantile(0.99))
        )
    return cleaned


def _plot_price_trend(df: pd.DataFrame, dataset_name: str, output_dir: Path) -> None:
    numeric_df = df.select_dtypes(include=["number"]).dropna(how="all")
    if numeric_df.empty:
        return
    numeric_df = numeric_df.sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    numeric_df.plot(ax=ax, linewidth=1)
    ax.set_title(f"{dataset_name} price development")
    ax.set_xlabel("Date" if isinstance(numeric_df.index, pd.DatetimeIndex) else "Index")
    ax.set_ylabel("Price")
    if len(numeric_df.columns) > 12:
        ax.legend().remove()
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    path = output_dir / f"{_sanitize(dataset_name)}_price.png"
    fig.savefig(path)
    plt.close(fig)


def _plot_momentum_feature(
    momentum_df: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    windows: list[int] = DEFAULT_MOMENTUM_WINDOWS,
) -> None:
    momentum_df = momentum_df.copy().dropna(how="all")
    if momentum_df.empty:
        return
    drop_n = max(windows) if windows else 0
    if drop_n > 0 and len(momentum_df) > drop_n:
        momentum_df = momentum_df.iloc[drop_n:]
    fig, ax = plt.subplots(figsize=(12, 5))
    momentum_df.plot(ax=ax, linewidth=1.2)
    ax.set_title(f"{dataset_name} momentum features")
    ax.set_xlabel("Date" if isinstance(momentum_df.index, pd.DatetimeIndex) else "Index")
    ax.set_ylabel("Momentum")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    path = output_dir / f"{_sanitize(dataset_name)}_momentum.png"
    fig.savefig(path)
    plt.close(fig)


def _plot_return_distribution(df: pd.DataFrame, dataset_name: str, output_dir: Path) -> None:
    numeric_df = df.select_dtypes(include=["number"]).dropna(how="all")
    if numeric_df.empty:
        return
    returns = np.log(numeric_df / numeric_df.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan)
    stacked = returns.stack(dropna=True)
    if stacked.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(stacked, bins=60, density=True, color="#1f77b4", alpha=0.7)
    stacked.plot(kind="kde", ax=ax, color="#d62728", linewidth=1.5)
    ax.set_title(f"{dataset_name} return distribution")
    ax.set_xlabel("Log return")
    ax.set_ylabel("Density")
    fig.tight_layout()
    path = output_dir / f"{_sanitize(dataset_name)}_return_distribution.png"
    fig.savefig(path)
    plt.close(fig)


def _plot_return_box(df: pd.DataFrame, dataset_name: str, output_dir: Path) -> None:
    numeric_df = df.select_dtypes(include=["number"]).dropna(how="all")
    if numeric_df.empty:
        return
    returns = np.log(numeric_df / numeric_df.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if returns.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([returns[col].dropna() for col in returns.columns], labels=[str(c) for c in returns.columns], vert=True)
    ax.set_title(f"{dataset_name} return boxplot")
    ax.set_ylabel("Log return")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    path = output_dir / f"{_sanitize(dataset_name)}_return_box.png"
    fig.savefig(path)
    plt.close(fig)


def _compute_momentum_features(
    dataset_name: str,
    df: pd.DataFrame,
    config_path: Path,
) -> Optional[pd.DataFrame]:
    portfolio_map = {
        "dax_daily": "dax",
        "sdax_daily": "sdax",
    }
    portfolio_name = portfolio_map.get(dataset_name)
    if portfolio_name is None:
        return None
    price_cols = [c for c in df.columns if "TRDPRC_1" in str(c)]
    df_price = df[price_cols].copy() if price_cols else df.copy()
    df_price = df_price.ffill().bfill()
    min_non_null = int(len(df_price.columns) * 0.5) if len(df_price.columns) > 0 else 0
    if min_non_null > 0:
        df_price = df_price[df_price.count(axis=1) >= min_non_null]
    prep = DataPrep(str(config_path))
    features = prep.create_features(df_price, portfolio_name=portfolio_name, period_type="daily")
    momentum_cols = [c for c in features.columns if str(c).startswith("momentum_")]
    if not momentum_cols:
        return None
    return features[momentum_cols]


def _save_features_to_excel(features_df: pd.DataFrame, dataset_name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{_sanitize(dataset_name)}_features.xlsx"
    features_df.to_excel(out_path, index=True)


def _generate_dataset_plots(
    dataset_name: str,
    excel_path: Path,
    output_dir: Path,
    momentum_dir: Path,
    config_path: Path,
    preferred_fields: Optional[list[str]] = None,
    excluded_instruments: Optional[list[str]] = None,
    include_instruments: Optional[list[str]] = None,
    add_boxplot: bool = False,
) -> None:
    df = _load_prices_from_excel(
        excel_path,
        preferred_fields=preferred_fields,
        excluded_instruments=excluded_instruments,
        include_instruments=include_instruments,
    )
    if df.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_price_trend(df, dataset_name, output_dir)
    _plot_return_distribution(df, dataset_name, output_dir)
    if add_boxplot:
        _plot_return_box(df, dataset_name, output_dir)
    momentum_dir.mkdir(parents=True, exist_ok=True)
    momentum_df = _compute_momentum_features(dataset_name, df, config_path)
    if momentum_df is not None:
        _save_features_to_excel(momentum_df, dataset_name, momentum_dir)
        _plot_momentum_feature(momentum_df, dataset_name, momentum_dir)


def plot_datastorage_price_development(
    output_dir: Path = DEFAULT_PLOTS_DIR,
    feature_output_dir: Path = DEFAULT_FEATURE_PLOTS_DIR,
    config_path: Path = BASE_DIR / "config.yaml",
    preferred_fields: Optional[list[str]] = None,
) -> None:
    datasets = {
        "dax_daily": {
            "path": DEFAULT_STORAGE_DIR / "dax_daily.xlsx",
            "exclude": [".GDAXI", "VIXI", "V1XI"],
        },
        "sdax_daily": {
            "path": DEFAULT_STORAGE_DIR / "sdax_daily.xlsx",
            "exclude": [".SDAXI", "VIXI", "V1XI"],
        },
        "gdaxi_index": {
            "path": DEFAULT_STORAGE_DIR / "dax_daily.xlsx",
            "include": [".GDAXI"],
            "boxplot": True,
        },
        "sdaxi_index": {
            "path": DEFAULT_STORAGE_DIR / "sdax_daily.xlsx",
            "include": [".SDAXI"],
            "boxplot": True,
        },
    }
    for name, cfg in datasets.items():
        _generate_dataset_plots(
            name,
            cfg["path"],
            Path(output_dir),
            Path(feature_output_dir),
            Path(config_path),
            preferred_fields=preferred_fields,
            excluded_instruments=cfg.get("exclude"),
            include_instruments=cfg.get("include"),
            add_boxplot=cfg.get("boxplot", False),
        )
