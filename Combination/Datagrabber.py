"""
Datagrabber.py - Extended version for BA_combination
Retrieves both daily and 30-minute data based on config.yaml
"""

import datetime
from pathlib import Path
from typing import Dict, List
import pandas as pd
import LSEG as LS
from ConfigManager import ConfigManager


class DataGrabber:
    """Data grabber with config support"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize DataGrabber

        Args:
            config_path: Path to the config file
        """
        self.config = ConfigManager(config_path)

    def fetch_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch data for all portfolios (DAX, SDAX) and both periods (daily, intraday)

        Returns:
            Dictionary: {portfolio_name: {period_type: DataFrame}}
            Example: {"dax": {"daily": df, "intraday": df}, "sdax": {...}}

        Raises:
            ValueError: If portfolio or period is missing
            RuntimeError: If API calls fail
        """
        all_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        portfolios = self.config.get("data.portfolios", {})

        for portfolio_name in portfolios.keys():
            all_data[portfolio_name] = {}

            all_data[portfolio_name]["daily"] = self.fetch_portfolio_data(portfolio_name, "daily")

            all_data[portfolio_name]["intraday"] = self.fetch_portfolio_data(portfolio_name, "intraday")

        return all_data

    def fetch_portfolio_data(self, portfolio_name: str, period_type: str) -> pd.DataFrame:
        """
        Fetch data for a specific portfolio and period

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
            interval=interval
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
                interval=interval
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
                interval=interval
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

        Returns:
            Dictionary: {portfolio_name: DataFrame}
            Example: {"dax": df, "sdax": df}

        Raises:
            ValueError: If portfolio is missing
            RuntimeError: If API calls fail
        """
        all_company_data: Dict[str, pd.DataFrame] = {}
        portfolios = self.config.get("data.portfolios", {})

        for portfolio_name in portfolios.keys():
            portfolio_config = self.config.get(f"data.portfolios.{portfolio_name}")
            universe = portfolio_config["universe"]

            period_config = self.config.get("data.periods.daily")
            if period_config:
                company_params = {
                    'Curn': 'USD',
                    'SDate': period_config.get('start', '2024-01-01'),
                    'EDate': period_config.get('end', '2025-11-15'),
                    'Frq': 'D'
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

        out_dir = Path("DataStorage")
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
