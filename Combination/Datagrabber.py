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
from logger_config import get_logger

logger = get_logger(__name__)


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
        logger.info("="*70)
        logger.info("DATA FETCH STARTED - PORTFOLIO BASED")
        logger.info("="*70)
        print("\n" + "="*70)
        print("DATA FETCH STARTED - PORTFOLIO BASED")
        print("="*70)

        all_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        portfolios = self.config.get("data.portfolios", {})

        if not portfolios:
            logger.warning("No portfolios found in config")
            raise ValueError("No portfolios defined in config")

        for portfolio_name in portfolios.keys():
            portfolio_display = portfolios[portfolio_name].get('name', portfolio_name.upper())
            logger.info("="*70)
            logger.info(f"PORTFOLIO: {portfolio_display}")
            logger.info("="*70)
            print(f"\n{'='*70}")
            print(f"PORTFOLIO: {portfolio_display}")
            print(f"{'='*70}")

            all_data[portfolio_name] = {}

            # Daily data
            logger.info(f"[1/2] Fetching daily data for {portfolio_name.upper()}...")
            print(f"\n[1/2] Fetching daily data for {portfolio_name.upper()}...")
            all_data[portfolio_name]["daily"] = self.fetch_portfolio_data(portfolio_name, "daily")

            # 30-minute data
            logger.info(f"[2/2] Fetching 30-minute data for {portfolio_name.upper()}...")
            print(f"\n[2/2] Fetching 30-minute data for {portfolio_name.upper()}...")
            all_data[portfolio_name]["intraday"] = self.fetch_portfolio_data(portfolio_name, "intraday")

        logger.info("="*70)
        logger.info("DATA FETCH COMPLETE")
        logger.info("="*70)
        print("\n" + "="*70)
        print("DATA FETCH COMPLETE")
        print("="*70)

        # Show summary
        for portfolio_name in all_data:
            logger.info(f"{portfolio_name.upper()}:")
            logger.info(f"  Daily: {all_data[portfolio_name]['daily'].shape}")
            logger.info(f"  Intraday: {all_data[portfolio_name]['intraday'].shape}")
            print(f"\n{portfolio_name.upper()}:")
            print(f"  Daily: {all_data[portfolio_name]['daily'].shape}")
            print(f"  Intraday: {all_data[portfolio_name]['intraday'].shape}")

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
        if portfolio_config is None:
            raise ValueError(f"Portfolio '{portfolio_name}' not found in config")

        period_config = self.config.get(f"data.periods.{period_type}")
        if period_config is None:
            raise ValueError(f"Period '{period_type}' not found in config")

        start = datetime.datetime.strptime(period_config["start"], "%Y-%m-%d")
        end = datetime.datetime.strptime(period_config["end"], "%Y-%m-%d")
        interval = period_config["interval"]

        logger.info(f"Date range: {start.date()} to {end.date()}, Interval: {interval}")
        print(f"  Date range: {start.date()} to {end.date()}")
        print(f"  Interval: {interval}")

        # Portfolio data (equities)
        universe = portfolio_config["universe"]
        if not universe:
            raise ValueError(f"Portfolio '{portfolio_name}' has an empty universe")

        logger.info(f"Fetching portfolio data ({len(universe)} stocks)...")
        print(f"  Fetching portfolio data ({len(universe)} stocks)...")
        portfolio_df = LS.getHistoryData(
            universe=universe,
            fields=self.config.get("data.fields"),
            start=start,
            end=end,
            interval=interval
        )

        # Portfolio-specific index (DAX or SDAX)
        portfolio_index = portfolio_config.get("index")
        index_df = pd.DataFrame()
        if not portfolio_index:
            logger.warning(f"No index defined for portfolio '{portfolio_name}'")
        else:
            logger.info(f"Fetching portfolio index ({portfolio_index})...")
            print(f"  Fetching portfolio index ({portfolio_index})...")
            index_df = LS.getHistoryData(
                universe=[portfolio_index],
                fields=["TRDPRC_1"],
                start=start,
                end=end,
                interval=interval
            )

        # Common indices (VDAX)
        common_indices = self.config.get("data.common_indices", [])
        common_df = pd.DataFrame()
        if len(common_indices) > 0:
            logger.info(f"Fetching common indices ({len(common_indices)})...")
            print(f"  Fetching common indices ({len(common_indices)})...")
            common_df = LS.getHistoryData(
                universe=common_indices,
                fields=["TRDPRC_1"],
                start=start,
                end=end,
                interval=interval
            )

        combined_df = pd.concat([portfolio_df, index_df, common_df], axis=1)

        # Remove duplicate column names if present
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

        # Save as Excel
        logger.debug(f"Writing Excel: {portfolio_name}_{period_type}")
        print("  Writing Excel...")
        self.exceltextwriter(combined_df, f"{portfolio_name}_{period_type}")

        logger.info(f"{portfolio_name.upper()} {period_type} data loaded: {combined_df.shape}")
        print(f"  {portfolio_name.upper()} {period_type} data loaded: {combined_df.shape}")
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
        logger.info("="*70)
        logger.info("COMPANY DATA FETCH STARTED")
        logger.info("="*70)
        print("\n" + "="*70)
        print("COMPANY DATA FETCH STARTED")
        print("="*70)

        all_company_data: Dict[str, pd.DataFrame] = {}
        portfolios = self.config.get("data.portfolios", {})

        if not portfolios:
            logger.warning("No portfolios found in config")
            raise ValueError("No portfolios defined in config")

        for portfolio_name in portfolios.keys():
            portfolio_config = self.config.get(f"data.portfolios.{portfolio_name}")
            if portfolio_config is None:
                logger.error(f"Portfolio '{portfolio_name}' not found in config")
                raise ValueError(f"Portfolio '{portfolio_name}' not found in config")

            portfolio_display = portfolio_config.get('name', portfolio_name.upper())
            logger.info("="*70)
            logger.info(f"PORTFOLIO: {portfolio_display} - COMPANY DATA")
            logger.info("="*70)
            print(f"\n{'='*70}")
            print(f"PORTFOLIO: {portfolio_display} - COMPANY DATA")
            print(f"{'='*70}")

            # Universe (equities) for this portfolio
            universe = portfolio_config["universe"]
            if not universe:
                logger.warning(f"Portfolio '{portfolio_name}' has an empty universe, skipping...")
                continue

            logger.info(f"Fetching company data for {len(universe)} stocks...")
            print(f"  Fetching company data ({len(universe)} stocks)...")

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

            if company_df is None or company_df.empty:
                raise ValueError("Empty API response")

            logger.info(f"Writing Excel: {portfolio_name}_company_data")
            print("  Writing Excel...")
            self.exceltextwriter(company_df, f"{portfolio_name}_company_data")

            all_company_data[portfolio_name] = company_df

            logger.info(f"{portfolio_name.upper()} company data loaded: {company_df.shape}")
            print(f"  {portfolio_name.upper()} company data loaded: {company_df.shape}")

        logger.info("="*70)
        logger.info("COMPANY DATA FETCH COMPLETE")
        logger.info("="*70)
        print("\n" + "="*70)
        print("COMPANY DATA FETCH COMPLETE")
        print("="*70)

        # Show summary
        if all_company_data:
            for portfolio_name, df in all_company_data.items():
                logger.info(f"{portfolio_name.upper()}: {df.shape}")
                print(f"\n{portfolio_name.upper()}: {df.shape}")
        else:
            logger.warning("No company data loaded")
            print("No company data loaded")

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
        if period_config is None:
            raise ValueError(f"Period '{period_type}' not found in config")

        start = datetime.datetime.strptime(period_config["start"], "%Y-%m-%d")
        end = datetime.datetime.strptime(period_config["end"], "%Y-%m-%d")
        interval = period_config["interval"]

        print(f"  Date range: {start.date()} to {end.date()}")
        print(f"  Interval: {interval}")

        # Portfolio data (equities)
        print(f"  Fetching portfolio data ({len(self.config.get('data.universe'))} stocks)...")
        portfolio_df = LS.getHistoryData(
            universe=self.config.get("data.universe"),
            fields=self.config.get("data.fields"),
            start=start,
            end=end,
            interval=interval
        )

        # Index data (DAX, SDAX, VDAX)
        print(f"  Fetching index data ({len(self.config.get('data.indices'))} indices)...")
        index_df = LS.getHistoryData(
            universe=self.config.get("data.indices"),
            fields=["TRDPRC_1"],
            start=start,
            end=end,
            interval=interval
        )

        # Combine portfolio and index data
        print("  Combining data...")
        combined_df = self.combine_data(portfolio_df, index_df)

        # Save as Excel (compatibility with version 1)
        print("  Writing Excel...")
        self.exceltextwriter(combined_df, f"combined_{period_type}")

        print(f"  {period_type.capitalize()} data loaded successfully: {combined_df.shape}")
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
        # Concatenate along columns (indices should already align)
        combined_df = pd.concat([portfolio_df, index_df], axis=1)

        # Remove duplicate column names if present
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
        # Create a date column from the index if needed
        d = df.copy()
        if isinstance(d.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            d = d.reset_index()
            # After reset_index the column is named after the index or 'index'
            if "index" in d.columns:
                d = d.rename(columns={"index": "Date"})
            elif d.columns[0] not in ("Date", "Datetime"):
                d = d.rename(columns={d.columns[0]: "Date"})
        else:
            # Otherwise detect an existing date/time column and normalize
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
                # Sheet name: max 31 characters, clean invalid characters
                sheet_name = str(column).strip()[:31]
                # Remove invalid characters for Excel sheet names
                sheet_name = sheet_name.replace("/", "_").replace("\\", "_").replace("?", "_")
                sheet_name = sheet_name.replace("*", "_").replace("[", "_").replace("]", "_")
                sheet_name = sheet_name.replace(":", "_").replace("'", "_")
                
                if not sheet_name:
                    sheet_name = "sheet"
                
                # Build frame with date and current column, keep original names
                if "Date" in d.columns:
                    frame = d[["Date", column]].copy()
                else:
                    # Fallback: without date column
                    frame = d[[column]].copy()
                
                # Keep original column names and flatten MultiIndex if needed
                if isinstance(frame.columns, pd.MultiIndex):
                    frame.columns = [str(col).replace("/", "_") for col in frame.columns]

                frame.to_excel(writer, index=False, sheet_name=sheet_name)

        logger.info(f"Excel saved: {out_path}")
        print(f"    Excel saved: {out_path}")
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
    # Test
    from logger_config import setup_logging
    setup_logging()
    
    grabber = DataGrabber()
    
    # Fetch historical data (daily + intraday)
    all_data = grabber.fetch_all_data()
    
    for portfolio_name, portfolio_data in all_data.items():
        logger.info(f"{portfolio_name}:")
        logger.info(f"  Daily: {portfolio_data['daily'].shape}")
        logger.info(f"  Intraday: {portfolio_data['intraday'].shape}")
        print(f"\n{portfolio_name}:")
        print(f"  Daily: {portfolio_data['daily'].shape}")
        print(f"  Intraday: {portfolio_data['intraday'].shape}")
    
    # Fetch fundamental company data (for Fama-French models)
    print("\n" + "="*70)
    print("STARTING COMPANY DATA FETCH")
    print("="*70)
    company_data = grabber.fetch_company_data()
    
    for portfolio_name, df in company_data.items():
        logger.info(f"{portfolio_name} Company Data: {df.shape}")
        print(f"\n{portfolio_name.upper()} Company Data: {df.shape}")
