"""
LSEG.py - LSEG/Refinitiv Data API Interface
"""

import lseg.data as ld
import pandas as pd
import datetime
from typing import List, Dict, Any, Optional

from logger_config import get_logger

logger = get_logger(__name__)


def getHistoryData(
    universe: List[str], 
    fields: List[str], 
    start: datetime.datetime, 
    end: datetime.datetime, 
    interval: str
) -> pd.DataFrame:
    """
    Fetch historical data from the LSEG/Refinitiv API.

    Args:
        universe: List of instruments (e.g. ['SAP.DE', '.GDAXI'])
        fields: List of fields (e.g. ['TRDPRC_1', 'ACVOL_1'])
        start: Start date
        end: End date
        interval: Interval (e.g. 'daily', '30min')

    Returns:
        DataFrame with historical data

    Raises:
        ValueError: If parameters are invalid
    """
    if not universe:
        raise ValueError("Universe must not be empty")
    if not fields:
        raise ValueError("Fields must not be empty")
    if start >= end:
        raise ValueError(f"Start date ({start}) must be before end date ({end})")

    logger.debug(
        "Fetching data: %d instruments, %d fields, %s to %s, interval: %s",
        len(universe), len(fields), start.date(), end.date(), interval
    )

    ld.open_session()
    logger.debug("LSEG session opened")

    df = ld.get_history(
        universe=universe,
        fields=fields,
        start=start,
        end=end,
        interval=interval,
    )

    logger.info(
        "Data loaded: %s rows, %s columns",
        len(df), len(df.columns) if df is not None else 0
    )

    if df is not None and not df.empty:
        logger.debug("First row:\n%s", df.head(1))

    ld.close_session()
    logger.debug("LSEG session closed")

    return df

DEFAULT_COMPANY_FIELDS = [
    "TR.CompanyMarketCapitalization.Date",
    "TR.CompanyMarketCapitalization",
    "TR.BookValuePerShare",
    "TR.BVPSActValue(Period=FY0)",
    "TR.NumberofSharesOutstandingActual(Period=FY0)",
    "TR.SharesOutstanding",
    "TR.OperatingIncome",
    "TR.TotalAssetsActual(Period=FY0)",
]

DEFAULT_COMPANY_PARAMS = {
    'Curn': 'USD',
    'SDate': '2024-01-01',
    "EDate": "2025-11-15",
    "Frq": "D"
}


def getCompanyData(
    universe: List[str],
    fields: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Fetch fundamental company data from the LSEG/Refinitiv API.
    
    Used for Fama-French/Carhart models (market cap, book value, etc.).

    Args:
        universe: List of tickers (e.g. ['RHMG.DE', 'ENR1n.DE'])
        fields: List of fields (default: DEFAULT_COMPANY_FIELDS)
        parameters: Parameter dict (default: DEFAULT_COMPANY_PARAMS)

    Returns:
        DataFrame with company fundamentals

    Raises:
        ValueError: If parameters are invalid
    """
    if not universe:
        raise ValueError("Universe must not be empty")
    
    if fields is None:
        fields = DEFAULT_COMPANY_FIELDS
    if parameters is None:
        parameters = DEFAULT_COMPANY_PARAMS.copy()

    logger.debug(
        "Fetching company data: %d instruments, %d fields",
        len(universe), len(fields)
    )

    ld.open_session()
    logger.debug("LSEG session opened (company data)")

    df = ld.get_data(
        universe=universe,
        fields=fields,
        parameters=parameters
    )

    if df is not None and not df.empty:
        df = df.copy()
        
        date_col = None
        for col in df.columns:
            if isinstance(col, tuple):
                col_str = ' '.join(str(c) for c in col)
            else:
                col_str = str(col)
            
            if 'TR.CompanyMarketCapitalization.Date' in col_str or (
                'CompanyMarketCapitalization' in col_str and 'Date' in col_str
            ):
                date_col = col
                break
        
        if date_col is not None:
            df = df.rename(columns={date_col: 'Date'})
            logger.info(f"Date column taken from field: {date_col}")
        else:
            for col in df.columns:
                if isinstance(col, tuple):
                    col_str = ' '.join(str(c) for c in col)
                else:
                    col_str = str(col)
                if 'date' in col_str.lower():
                    df = df.rename(columns={col: 'Date'})
                    logger.info(f"Date column taken from field: {col} (fallback)")
                    break
            else:
                if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
                    df['Date'] = df.index
                    logger.info("Date column created from index")
                else:
                    logger.warning("Could not find a date column")
        
        if 'Date' in df.columns:
            cols = ['Date'] + [col for col in df.columns if col != 'Date']
            df = df[cols]
        
        logger.debug("First company data row:\n%s", df.head(1))

    logger.info(
        "Company data loaded: %s rows, %s columns",
        len(df), len(df.columns) if df is not None else 0
    )

    ld.close_session()
    logger.debug("LSEG session closed (company data)")

    return df
