"""
LSEG.py - LSEG/Refinitiv Data API Interface
"""

import lseg.data as ld
import pandas as pd
from lseg.data.discovery import Chain
import datetime
import time
from functools import wraps
from typing import List, Optional
from IPython.display import display, clear_output

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