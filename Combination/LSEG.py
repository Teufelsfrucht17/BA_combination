"""
LSEG.py - LSEG/Refinitiv Data API Interface
"""

import lseg.data as ld
import pandas as pd
from lseg.data.discovery import Chain
import datetime
import time
from functools import wraps
from typing import List, Optional, Dict, Any

import GloablVariableStorage
from logger_config import get_logger

logger = get_logger(__name__)

# Constants
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0
DEFAULT_RETRY_BACKOFF = 2.0


def retry(max_attempts: int = DEFAULT_MAX_RETRIES, delay: float = DEFAULT_RETRY_DELAY, 
          backoff: float = DEFAULT_RETRY_BACKOFF):
    """
    Decorator für Retry-Logik bei API-Calls

    Args:
        max_attempts: Maximale Anzahl Versuche
        delay: Initiale Wartezeit in Sekunden
        backoff: Multiplikator für Wartezeit bei jedem Retry

    Returns:
        Decorated Function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(
                            "Max Retries erreicht für %s: %s", 
                            func.__name__, e, exc_info=True
                        )
                        raise
                    
                    logger.warning(
                        "Versuch %d/%d fehlgeschlagen für %s: %s. Retry in %.1fs...",
                        attempt, max_attempts, func.__name__, e, current_delay
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        return wrapper
    return decorator


@retry(max_attempts=DEFAULT_MAX_RETRIES, delay=DEFAULT_RETRY_DELAY)
def getHistoryData(
    universe: List[str], 
    fields: List[str], 
    start: datetime.datetime, 
    end: datetime.datetime, 
    interval: str
) -> pd.DataFrame:
    """
    Holt historische Daten von LSEG/Refinitiv API mit Retry-Logik

    Args:
        universe: Liste von Aktien/Indizes (z.B. ['SAP.DE', '.GDAXI'])
        fields: Liste von Feldern (z.B. ['TRDPRC_1', 'ACVOL_1'])
        start: Start-Datum
        end: End-Datum
        interval: Intervall (z.B. 'daily', '30min')

    Returns:
        DataFrame mit historischen Daten

    Raises:
        RuntimeError: Wenn API-Call fehlschlägt
        ValueError: Wenn Parameter ungültig sind
    """
    if not universe:
        raise ValueError("Universe darf nicht leer sein")
    if not fields:
        raise ValueError("Fields darf nicht leer sein")
    if start >= end:
        raise ValueError(f"Start-Datum ({start}) muss vor End-Datum ({end}) sein")

    logger.debug(
        "Hole Daten: %d Instrumente, %d Felder, %s bis %s, Intervall: %s",
        len(universe), len(fields), start.date(), end.date(), interval
    )

    try:
        ld.open_session()
        logger.debug("LSEG Session geöffnet")

        df = ld.get_history(
            universe=universe,
            fields=fields,
            start=start,
            end=end,
            interval=interval,
        )

        logger.info(
            "Daten erfolgreich geladen: %s Zeilen, %s Spalten",
            len(df), len(df.columns) if df is not None else 0
        )

        if df is not None and not df.empty:
            logger.debug("Erste Datenzeile:\n%s", df.head(1))

        return df

    except Exception as e:
        logger.error("Fehler beim Abrufen der Daten: %s", e, exc_info=True)
        raise RuntimeError(f"LSEG API-Call fehlgeschlagen: {e}") from e

    finally:
        try:
            ld.close_session()
            logger.debug("LSEG Session geschlossen")
        except Exception as e:
            logger.warning("Fehler beim Schließen der Session: %s", e)


# Keine Tests - wird von Datagrabber aufgerufen


# Constants für Company Data
DEFAULT_COMPANY_FIELDS = [
    "TR.CompanyMarketCapitalization.Date",  # Datumsfeld für Market Cap
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
    "EDate": "2024-12-31",
    "Frq": "D"
}


@retry(max_attempts=DEFAULT_MAX_RETRIES, delay=DEFAULT_RETRY_DELAY)
def getCompanyData(
    universe: List[str],
    fields: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Holt fundamentale Company-Daten von LSEG/Refinitiv API mit Retry-Logik.
    
    Diese Daten werden für Fama-French/Carhart Modelle benötigt (Market Cap, Book Value, etc.)

    Args:
        universe: Liste von Aktien (z.B. ['RHMG.DE', 'ENR1n.DE'])
        fields: Liste von Feldern (default: DEFAULT_COMPANY_FIELDS)
        parameters: Dictionary mit Parametern (default: DEFAULT_COMPANY_PARAMS)

    Returns:
        DataFrame mit fundamentalen Company-Daten

    Raises:
        RuntimeError: Wenn API-Call fehlschlägt
        ValueError: Wenn Parameter ungültig sind
    """
    if not universe:
        raise ValueError("Universe darf nicht leer sein")
    
    # Verwende Defaults falls nicht angegeben
    if fields is None:
        fields = DEFAULT_COMPANY_FIELDS
    if parameters is None:
        parameters = DEFAULT_COMPANY_PARAMS.copy()

    logger.debug(
        "Hole Company-Daten: %d Instrumente, %d Felder",
        len(universe), len(fields)
    )

    try:
        ld.open_session()
        logger.debug("LSEG Session geöffnet (Company Data)")

        df = ld.get_data(
            universe=universe,
            fields=fields,
            parameters=parameters
        )

        if df is not None and not df.empty:
            df = df.copy()
            
            # Finde die "TR.CompanyMarketCapitalization.Date" Spalte und benenne sie in "Date" um
            date_col = None
            for col in df.columns:
                if isinstance(col, tuple):
                    # MultiIndex: Prüfe alle Ebenen
                    col_str = ' '.join(str(c) for c in col)
                else:
                    col_str = str(col)
                
                # Suche nach dem exakten Feld oder ähnlichem
                if 'TR.CompanyMarketCapitalization.Date' in col_str or (
                    'CompanyMarketCapitalization' in col_str and 'Date' in col_str
                ):
                    date_col = col
                    break
            
            if date_col is not None:
                # Benenne die Spalte einfach in "Date" um
                df = df.rename(columns={date_col: 'Date'})
                logger.info(f"Date-Spalte übernommen von Feld: {date_col}")
            else:
                # Fallback: Suche nach beliebiger Date-Spalte
                for col in df.columns:
                    if isinstance(col, tuple):
                        col_str = ' '.join(str(c) for c in col)
                    else:
                        col_str = str(col)
                    if 'date' in col_str.lower():
                        df = df.rename(columns={col: 'Date'})
                        logger.info(f"Date-Spalte übernommen von Feld: {col} (Fallback)")
                        break
                else:
                    # Letzter Fallback: Verwende Index
                    if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
                        df['Date'] = df.index
                        logger.info("Date-Spalte aus Index erstellt")
                    else:
                        logger.warning("Konnte Date-Spalte nicht finden")
            
            # Verschiebe Date-Spalte an den Anfang, falls vorhanden
            if 'Date' in df.columns:
                cols = ['Date'] + [col for col in df.columns if col != 'Date']
                df = df[cols]
            
            logger.debug("Erste Company-Datenzeile:\n%s", df.head(1))

        logger.info(
            "Company-Daten erfolgreich geladen: %s Zeilen, %s Spalten",
            len(df), len(df.columns) if df is not None else 0
        )

        return df

    except Exception as e:
        logger.error("Fehler beim Abrufen der Company-Daten: %s", e, exc_info=True)
        raise RuntimeError(f"LSEG API-Call für Company-Daten fehlgeschlagen: {e}") from e

    finally:
        try:
            ld.close_session()
            logger.debug("LSEG Session geschlossen (Company Data)")
        except Exception as e:
            logger.warning("Fehler beim Schließen der Session: %s", e)