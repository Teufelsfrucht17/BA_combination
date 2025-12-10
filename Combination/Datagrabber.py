"""
Datagrabber.py - Extended version for BA_combination
Retrieves both daily and 30-minute data based on config.yaml
Supports two modes controlled via config:
  - data.use_lseg_api = True  -> use LSEG API and update Excel cache
  - data.use_lseg_api = False -> skip API calls and load from cached Excels in DataStorage
"""

import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import LSEG as LS
from ConfigManager import ConfigManager


def _clean_column_name(raw_name: object, index_prefix: Optional[str] = None) -> str:
    """
    Normalize column names from Excel sheets.

    This is equivalent to the helper used in the BA_trading_walkthrough notebook
    so that cached Excels created by exceltextwriter can be re-loaded in a robust way.
    """
    col = str(raw_name)
    if col == "TRDPRC_1" and index_prefix:
        return f"{index_prefix}_TRDPRC_1"
    col = col.replace("(", "").replace(")", "").replace("'", "")
    col = col.replace(" ", "").replace(",", "_")
    col = col.replace("__", "_").strip("_")
    return col


def _load_price_excel(excel_path: Path, index_prefix: Optional[str] = None) -> pd.DataFrame:
    """
    Load price data from an Excel file created by exceltextwriter.

    The resulting DataFrame matches the structure expected by Dataprep/FamaFrench.
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"Datei fehlt: {excel_path}")

    sheets = pd.read_excel(excel_path, sheet_name=None)
    merged: Optional[pd.DataFrame] = None

    for _, df in sheets.items():
        if df is None or df.empty:
            continue

        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])

        value_cols = [c for c in df.columns if c != "Date"]
        if not value_cols:
            continue

        for col in value_cols:
            cleaned = _clean_column_name(col, index_prefix=index_prefix)
            series = pd.to_numeric(df[col], errors="coerce")
            tmp = pd.DataFrame({cleaned: series.values}, index=df["Date"])
            merged = tmp if merged is None else merged.join(tmp, how="outer")

    if merged is None:
        raise ValueError(f"Keine gueltigen Daten in {excel_path}")

    merged.index = pd.to_datetime(merged.index)
    return merged.sort_index()


def _load_company_excel(excel_path: Path) -> pd.DataFrame:
    """
    Load company/fundamental data from an Excel file created by exceltextwriter.
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"Datei fehlt: {excel_path}")

    sheets = pd.read_excel(excel_path, sheet_name=None)
    merged: Optional[pd.DataFrame] = None

    for name, df in sheets.items():
        if df is None or df.empty:
            continue

        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])

        value_cols = [c for c in df.columns if c != "Date"]
        if not value_cols:
            continue

        cleaned_cols = {col: _clean_column_name(f"{name}_{col}") for col in value_cols}
        tmp = df[["Date"] + value_cols].rename(columns=cleaned_cols).set_index("Date")
        merged = tmp if merged is None else merged.join(tmp, how="outer")

    if merged is None:
        raise ValueError(f"Keine gueltigen Company-Daten in {excel_path}")

    merged.index = pd.to_datetime(merged.index)
    return merged.sort_index()


class DataGrabber:
    """Data grabber with config support (API or cached Excel)."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize DataGrabber

        Args:
            config_path: Path to the config file
        """
        self.config = ConfigManager(config_path)
        self._cached_prices: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None
        self._cached_companies: Optional[Dict[str, pd.DataFrame]] = None

    def _use_lseg_api(self) -> bool:
        return bool(self.config.get("data.use_lseg_api", True))

    def _get_storage_dir(self) -> Path:
        storage_dir = self.config.get("data.storage_dir", "DataStorage")
        return Path(storage_dir)

    def _load_cached_data(self) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, pd.DataFrame]]:
        """
        Load cached price and company data from Excel files in storage_dir.
        """
        portfolios = self.config.get("data.portfolios", {})
        data_dir = self._get_storage_dir()

        price_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        company_data: Dict[str, pd.DataFrame] = {}

        for p_name, p_cfg in portfolios.items():
            p_cfg = p_cfg or {}
            index_prefix = p_cfg.get("index", "").replace(".", "")
            price_data[p_name] = {}

            for period in ["daily", "intraday"]:
                excel_path = data_dir / f"{p_name}_{period}.xlsx"
                if excel_path.exists():
                    price_data[p_name][period] = _load_price_excel(excel_path, index_prefix=index_prefix)
                else:
                    print(f"WARNUNG: {excel_path} fehlt.")

            comp_path = data_dir / f"{p_name}_company_data.xlsx"
            if comp_path.exists():
                company_data[p_name] = _load_company_excel(comp_path)
            else:
                print(f"Hinweis: {comp_path} nicht gefunden (FFC optional).")

        return price_data, company_data

    def _ensure_cached_loaded(self) -> None:
        if self._cached_prices is None or self._cached_companies is None:
            self._cached_prices, self._cached_companies = self._load_cached_data()

    def fetch_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch data for all portfolios (DAX, SDAX) and both periods (daily, intraday).

        Behaviour depends on config:
          - data.use_lseg_api = True  -> pull via LSEG API and update Excel cache
          - data.use_lseg_api = False -> load from existing Excel cache only
        """
        if not self._use_lseg_api():
            self._ensure_cached_loaded()
            return self._cached_prices or {}

        all_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        portfolios = self.config.get("data.portfolios", {})

        for portfolio_name in portfolios.keys():
            all_data[portfolio_name] = {}

            all_data[portfolio_name]["daily"] = self.fetch_portfolio_data(portfolio_name, "daily")

            all_data[portfolio_name]["intraday"] = self.fetch_portfolio_data(portfolio_name, "intraday")

        return all_data

    def fetch_portfolio_data(self, portfolio_name: str, period_type: str) -> pd.DataFrame:
        """
        Fetch data for a specific portfolio and period using the LSEG API.

        Args:
            portfolio_name: Portfolio name (e.g. "dax", "sdax")
            period_type: "daily" or "intraday"

        Returns:
            DataFrame with portfolio stocks, portfolio index, and shared indices
        """
        portfolio_config = self.config.get(f"data.portfolios.{portfolio_name}")
        period_config = self.config.get(f"data.periods.{period_type}")
        start = datetime.datetime.strptime(period_config["start"], "%Y-%m-%d")
        end = datetime.datetime.strptime(period_config["end"], "%Y-%m-%d")
        interval = period_config["interval"]

        def _prefix_single_universe(df: pd.DataFrame, universe: List[str]) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            if isinstance(df.columns, pd.MultiIndex):
                return df
            if len(universe) == 1:
                instrument = universe[0]
                df = df.copy()
                df.columns = [f"{instrument}_{col}" for col in df.columns]
            return df

        universe = portfolio_config["universe"]
        portfolio_df = LS.getHistoryData(
            universe=universe,
            fields=self.config.get("data.fields"),
            start=start,
            end=end,
            interval=interval,
        )
        portfolio_df = _prefix_single_universe(portfolio_df, universe)

        portfolio_index = portfolio_config.get("index")
        index_df = pd.DataFrame()
        if portfolio_index:
            index_df = LS.getHistoryData(
                universe=[portfolio_index],
                fields=["TRDPRC_1"],
                start=start,
                end=end,
                interval=interval,
            )
            index_df = _prefix_single_universe(index_df, [portfolio_index])

        common_indices = self.config.get("data.common_indices", [])
        common_df = pd.DataFrame()
        if len(common_indices) > 0:
            common_df = LS.getHistoryData(
                universe=common_indices,
                fields=["TRDPRC_1"],
                start=start,
                end=end,
                interval=interval,
            )
            common_df = _prefix_single_universe(common_df, common_indices)
            print(f"Common indices fetched ({len(common_indices)}): {common_df.shape}")
            print(common_df.head(2))

        combined_df = pd.concat([portfolio_df, index_df, common_df], axis=1)

        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

        self.exceltextwriter(combined_df, f"{portfolio_name}_{period_type}")

        return combined_df

    def fetch_company_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch fundamental company data for all portfolios (DAX, SDAX)

        These data are used for Fama-French/Carhart models.
        Creates separate Excel files for each portfolio.

        Behaviour depends on config:
          - data.use_lseg_api = True  -> pull via LSEG API and update Excel cache
          - data.use_lseg_api = False -> load from existing Excel cache only

        Returns:
            Dictionary: {portfolio_name: DataFrame}
            Example: {"dax": df, "sdax": df}

        Raises:
            ValueError: If portfolio is missing
            RuntimeError: If API calls fail
        """
        if not self._use_lseg_api():
            self._ensure_cached_loaded()
            return self._cached_companies or {}

        all_company_data: Dict[str, pd.DataFrame] = {}
        portfolios = self.config.get("data.portfolios", {})

        for portfolio_name in portfolios.keys():
            portfolio_config = self.config.get(f"data.portfolios.{portfolio_name}")
            universe = portfolio_config["universe"]

            period_config = self.config.get("data.periods.daily")
            if period_config:
                company_params = {
                    "Curn": "USD",
                    "SDate": period_config.get("start", "2024-01-01"),
                    "EDate": period_config.get("end", "2025-11-15"),
                    "Frq": "D",
                }
                company_df = LS.getCompanyData(universe=universe, parameters=company_params)
            else:
                company_df = LS.getCompanyData(universe=universe)

            self.exceltextwriter(company_df, f"{portfolio_name}_company_data")

            all_company_data[portfolio_name] = company_df

        return all_company_data

    def fetch_period_data(self, period_type: str) -> pd.DataFrame:
        """
        Fetch data for a specific period (daily/intraday)

        Args:
            period_type: "daily" or "intraday"

        Returns:
            DataFrame with combined portfolio and index data
        """
        period_config = self.config.get(f"data.periods.{period_type}")
        start = datetime.datetime.strptime(period_config["start"], "%Y-%m-%d")
        end = datetime.datetime.strptime(period_config["end"], "%Y-%m-%d")
        interval = period_config["interval"]

        portfolio_df = LS.getHistoryData(
            universe=self.config.get("data.universe"),
            fields=self.config.get("data.fields"),
            start=start,
            end=end,
            interval=interval
        )

        index_df = LS.getHistoryData(
            universe=self.config.get("data.indices"),
            fields=["TRDPRC_1"],
            start=start,
            end=end,
            interval=interval
        )

        combined_df = self.combine_data(portfolio_df, index_df)

        self.exceltextwriter(combined_df, f"combined_{period_type}")

        return combined_df

    def combine_data(self, portfolio_df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine portfolio and index data

        Args:
            portfolio_df: DataFrame with portfolio data
            index_df: DataFrame with index data

        Returns:
            Combined DataFrame
        """
        combined_df = pd.concat([portfolio_df, index_df], axis=1)

        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

        return combined_df

    def exceltextwriter(self, df: pd.DataFrame, name: str) -> None:
        """
        Save a DataFrame as Excel (compatibility helper from version 1)

        Args:
            df: DataFrame to write
            name: Excel file name without extension

        Raises:
            IOError: If Excel file cannot be written
        """
        d = df.copy()
        if isinstance(d.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            d = d.reset_index()
            if "index" in d.columns:
                d = d.rename(columns={"index": "Date"})
            elif d.columns[0] not in ("Date", "Datetime"):
                d = d.rename(columns={d.columns[0]: "Date"})
        else:
            for c in ("Date", "Datetime", "DATE", "timestamp"):
                if c in d.columns:
                    if c != "Date":
                        d = d.rename(columns={c: "Date"})
                    break

        out_dir = self._get_storage_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{name}.xlsx"

        with pd.ExcelWriter(
            out_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm:ss"
        ) as writer:
            for column in df.columns:
                sheet_name = str(column).strip()[:31]
                sheet_name = sheet_name.replace("/", "_").replace("\\", "_").replace("?", "_")
                sheet_name = sheet_name.replace("*", "_").replace("[", "_").replace("]", "_")
                sheet_name = sheet_name.replace(":", "_").replace("'", "_")
                
                if not sheet_name:
                    sheet_name = "sheet"
                
                if "Date" in d.columns:
                    frame = d[["Date", column]].copy()
                else:
                    frame = d[[column]].copy()
                
                if isinstance(frame.columns, pd.MultiIndex):
                    frame.columns = [str(col).replace("/", "_") for col in frame.columns]

                frame.to_excel(writer, index=False, sheet_name=sheet_name)

        return


def createExcel(
    universe: List[str],
    fields: List[str],
    start: datetime.datetime,
    end: datetime.datetime,
    interval: str,
    name: str,
) -> None:
    """
    Fetch data and save as Excel (compatibility helper)

    Args:
        universe: List of instruments
        fields: List of fields
        start: Start date
        end: End date
        interval: Interval (D, 30min, etc.)
        name: Excel file name
    """
    df = LS.getHistoryData(
        universe=universe, fields=fields, start=start, end=end, interval=interval
    )

    grabber = DataGrabber()
    grabber.exceltextwriter(df, name)
    return


if __name__ == "__main__":
    grabber = DataGrabber()
    grabber.fetch_all_data()
    grabber.fetch_company_data()
