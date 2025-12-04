"""
Plotting helpers for inspecting raw and prepared data.

Creates a line plot and a boxplot for each numeric DataFrame provided.
Includes loaders for plotting price developments directly from the Excel
files stored in DataStorage.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union, List, Optional

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PLOTS_DIR = BASE_DIR / "Results" / "plots"
DEFAULT_STORAGE_DIR = BASE_DIR / "DataStorage"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DataContainer = Union[pd.DataFrame, Dict[str, Any]]
PREFERRED_PRICE_FIELDS: List[str] = ["TRDPRC_1", "OPEN_PRC", "CLOSE_PRC"]
OPEN_CLOSE_FIELDS: List[str] = ["OPEN_PRC", "CLOSE_PRC"]


def _sanitize(name: str) -> str:
    """Create a filesystem-friendly name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name) or "plot"


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


def _flatten_data(data: DataContainer, prefix: str) -> Iterable[Tuple[str, pd.DataFrame]]:
    """
    Yield (name, DataFrame) pairs from nested dicts or a single DataFrame.

    Args:
        data: DataFrame or (nested) dictionary of DataFrames.
        prefix: Name prefix used to build plot labels.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            child_prefix = f"{prefix}_{key}" if prefix else str(key)
            yield from _flatten_data(value, child_prefix)
    elif isinstance(data, pd.DataFrame):
        yield prefix, data
    else:
        return


def _plot_single_dataframe(df: pd.DataFrame, plot_name: str, output_dir: Path) -> None:
    """Create line plot and boxplot for a single DataFrame."""
    if matplotlib is None or plt is None:
        return

    numeric_df = df.select_dtypes(include=["number"]).copy()
    numeric_df = numeric_df.dropna(how="all")

    if numeric_df.empty:
        return

    numeric_df.sort_index(inplace=True)

    safe_name = _sanitize(plot_name)

    fig, ax = plt.subplots(figsize=(12, 5))
    numeric_df.plot(ax=ax, linewidth=1)
    ax.set_title(f"{plot_name} - Line plot")
    ax.set_xlabel("Date" if isinstance(numeric_df.index, pd.DatetimeIndex) else "Index")
    ax.set_ylabel("Value")
    if len(numeric_df.columns) > 10:
        ax.legend().remove()
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    line_path = output_dir / f"{safe_name}_line.png"
    fig.savefig(line_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    numeric_df.plot(kind="box", ax=ax, rot=45)
    ax.set_title(f"{plot_name} - Boxplot")
    ax.set_ylabel("Value distribution")
    fig.tight_layout()
    box_path = output_dir / f"{safe_name}_box.png"
    fig.savefig(box_path)
    plt.close(fig)


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


def plot_datastorage_price_development(
    output_dir: Union[str, Path] = DEFAULT_PLOTS_DIR / "datastorage_prices",
    preferred_fields: Optional[List[str]] = None,
    include_intraday: bool = False,
) -> None:
    """
    Plot price development of DAX and SDAX companies from stored Excel files.

    Reads `dax_daily.xlsx` and `sdax_daily.xlsx` from DataStorage, combines the
    preferred price fields per instrument, and saves line/box plots. Set
    `include_intraday=True` to also plot 30-minute data.
    """
    storage_dir = DEFAULT_STORAGE_DIR
    datasets = {
        "dax_daily_prices": storage_dir / "dax_daily.xlsx",
        "sdax_daily_prices": storage_dir / "sdax_daily.xlsx",
    }
    if include_intraday:
        datasets["dax_intraday_prices"] = storage_dir / "dax_intraday.xlsx"
        datasets["sdax_intraday_prices"] = storage_dir / "sdax_intraday.xlsx"

    preferred_main = preferred_fields or OPEN_CLOSE_FIELDS
    for name, path in datasets.items():
        df_main = _load_prices_from_excel(
            path,
            preferred_fields=preferred_main,
            exclude_fields=["TRDPRC_1"],
        )
        if df_main.empty:
            continue
        else:
            plot_data_overview(df_main, dataset_name=f"{name}_no_trdprc1", output_dir=output_dir)

        df_trd = _load_prices_from_excel(path, preferred_fields=["TRDPRC_1"])
        if df_trd.empty:
            continue
        else:
            trd_dir = Path(output_dir) / "trdprc1_only"
            plot_data_overview(df_trd, dataset_name=f"{name}_trdprc1", output_dir=trd_dir)


def plot_data_overview(
    data: DataContainer,
    dataset_name: str,
    output_dir: Union[str, Path] = DEFAULT_PLOTS_DIR,
) -> None:
    """
    Plot numeric columns of raw or prepared data as line plot and boxplot.

    Args:
        data: DataFrame or nested dictionaries containing DataFrames.
        dataset_name: Name prefix used in plot titles and file names.
        output_dir: Directory where plots are written.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for plot_name, df in _flatten_data(data, dataset_name):
        _plot_single_dataframe(df, plot_name, output_path)
