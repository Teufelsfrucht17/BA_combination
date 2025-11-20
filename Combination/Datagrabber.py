"""
Datagrabber.py - Erweiterte Version für BA_combination
Holt sowohl tägliche als auch 30-Min Daten basierend auf config.yaml
"""

import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import LSEG as LS
from ConfigManager import ConfigManager
from logger_config import get_logger

logger = get_logger(__name__)


class DataGrabber:
    """Erweiterte Version des DataGrabbers mit Config-Unterstützung"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialisiert DataGrabber

        Args:
            config_path: Pfad zur Config-Datei
        """
        self.config = ConfigManager(config_path)

    def fetch_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Holt Daten für alle Portfolios (DAX, SDAX) und beide Perioden (daily, intraday)

        Returns:
            Dictionary: {portfolio_name: {period_type: DataFrame}}
            Beispiel: {"dax": {"daily": df, "intraday": df}, "sdax": {...}}

        Raises:
            ValueError: Wenn Portfolio oder Periode nicht gefunden wird
            RuntimeError: Wenn API-Calls fehlschlagen
        """
        logger.info("="*70)
        logger.info("DATENABRUF GESTARTET - PORTFOLIO-BASIERT")
        logger.info("="*70)
        print("\n" + "="*70)
        print("DATENABRUF GESTARTET - PORTFOLIO-BASIERT")
        print("="*70)

        all_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        portfolios = self.config.get("data.portfolios", {})

        if not portfolios:
            logger.warning("Keine Portfolios in Config gefunden")
            raise ValueError("Keine Portfolios in Config definiert")

        for portfolio_name in portfolios.keys():
            portfolio_display = portfolios[portfolio_name].get('name', portfolio_name.upper())
            logger.info("="*70)
            logger.info(f"PORTFOLIO: {portfolio_display}")
            logger.info("="*70)
            print(f"\n{'='*70}")
            print(f"PORTFOLIO: {portfolio_display}")
            print(f"{'='*70}")

            all_data[portfolio_name] = {}

            # Tägliche Daten
            logger.info(f"[1/2] Hole tägliche Daten für {portfolio_name.upper()}...")
            print(f"\n[1/2] Hole tägliche Daten für {portfolio_name.upper()}...")
            all_data[portfolio_name]["daily"] = self.fetch_portfolio_data(portfolio_name, "daily")

            # 30-Min Daten
            logger.info(f"[2/2] Hole 30-Minuten Daten für {portfolio_name.upper()}...")
            print(f"\n[2/2] Hole 30-Minuten Daten für {portfolio_name.upper()}...")
            all_data[portfolio_name]["intraday"] = self.fetch_portfolio_data(portfolio_name, "intraday")

        logger.info("="*70)
        logger.info("DATENABRUF ABGESCHLOSSEN")
        logger.info("="*70)
        print("\n" + "="*70)
        print("DATENABRUF ABGESCHLOSSEN")
        print("="*70)

        # Zeige Zusammenfassung
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
        Holt Daten für ein bestimmtes Portfolio und eine Periode

        Args:
            portfolio_name: Name des Portfolios (z.B. "dax", "sdax")
            period_type: "daily" oder "intraday"

        Returns:
            DataFrame mit Portfolio-Aktien, Portfolio-Index und gemeinsamen Indizes
        """
        portfolio_config = self.config.get(f"data.portfolios.{portfolio_name}")
        if portfolio_config is None:
            raise ValueError(f"Portfolio '{portfolio_name}' nicht in Config gefunden")

        period_config = self.config.get(f"data.periods.{period_type}")
        if period_config is None:
            raise ValueError(f"Periode '{period_type}' nicht in Config gefunden")

        # Konvertiere Datum-Strings zu datetime
        try:
            start = datetime.datetime.strptime(period_config["start"], "%Y-%m-%d")
            end = datetime.datetime.strptime(period_config["end"], "%Y-%m-%d")
        except (KeyError, ValueError) as e:
            logger.error(f"Ungültiges Datumsformat in Config: {e}", exc_info=True)
            raise ValueError(f"Ungültiges Datumsformat in Config: {e}") from e

        interval = period_config["interval"]

        logger.info(f"Zeitraum: {start.date()} bis {end.date()}, Interval: {interval}")
        print(f"  Zeitraum: {start.date()} bis {end.date()}")
        print(f"  Interval: {interval}")

        # Portfolio Daten (Aktien)
        universe = portfolio_config["universe"]
        if not universe:
            raise ValueError(f"Portfolio '{portfolio_name}' hat leeres Universe")

        logger.info(f"Hole Portfolio-Daten ({len(universe)} Aktien)...")
        print(f"  Hole Portfolio-Daten ({len(universe)} Aktien)...")
        
        try:
            portfolio_df = LS.getHistoryData(
                universe=universe,
                fields=self.config.get("data.fields"),
                start=start,
                end=end,
                interval=interval
            )
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Portfolio-Daten: {e}", exc_info=True)
            raise RuntimeError(f"Fehler beim Abrufen der Portfolio-Daten: {e}") from e

        # Portfolio-spezifischer Index (DAX oder SDAX)
        portfolio_index = portfolio_config.get("index")
        if not portfolio_index:
            logger.warning(f"Kein Index für Portfolio '{portfolio_name}' definiert")
        else:
            logger.info(f"Hole Portfolio-Index ({portfolio_index})...")
            print(f"  Hole Portfolio-Index ({portfolio_index})...")
            
            try:
                index_df = LS.getHistoryData(
                    universe=[portfolio_index],
                    fields=["TRDPRC_1"],
                    start=start,
                    end=end,
                    interval=interval
                )
            except Exception as e:
                logger.error(f"Fehler beim Abrufen des Index: {e}", exc_info=True)
                raise RuntimeError(f"Fehler beim Abrufen des Index: {e}") from e

        # Gemeinsame Indizes (VDAX)
        common_indices = self.config.get("data.common_indices", [])
        if len(common_indices) > 0:
            logger.info(f"Hole gemeinsame Indizes ({len(common_indices)})...")
            print(f"  Hole gemeinsame Indizes ({len(common_indices)})...")
            
            try:
                common_df = LS.getHistoryData(
                    universe=common_indices,
                    fields=["TRDPRC_1"],
                    start=start,
                    end=end,
                    interval=interval
                )
                # Kombiniere alle drei
                combined_df = pd.concat([portfolio_df, index_df, common_df], axis=1)
            except Exception as e:
                logger.warning(f"Fehler beim Abrufen der gemeinsamen Indizes: {e}", exc_info=True)
                # Fallback: nur Portfolio + Index
                combined_df = pd.concat([portfolio_df, index_df], axis=1)
        else:
            # Nur Portfolio + Index
            combined_df = pd.concat([portfolio_df, index_df], axis=1)

        # Entferne Duplikate in Spaltennamen falls vorhanden
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

        # Speichere als Excel
        logger.debug(f"Speichere als Excel: {portfolio_name}_{period_type}")
        print(f"  Speichere als Excel...")
        self.exceltextwriter(combined_df, f"{portfolio_name}_{period_type}")

        logger.info(f"{portfolio_name.upper()} {period_type} Daten geladen: {combined_df.shape}")
        print(f"  ✓ {portfolio_name.upper()} {period_type} Daten geladen: {combined_df.shape}")
        return combined_df

    def fetch_period_data(self, period_type: str) -> pd.DataFrame:
        """
        Holt Daten für eine bestimmte Periode (daily/intraday)

        Args:
            period_type: "daily" oder "intraday"

        Returns:
            DataFrame mit kombinierten Portfolio- und Index-Daten
        """
        period_config = self.config.get(f"data.periods.{period_type}")
        if period_config is None:
            raise ValueError(f"Periode '{period_type}' nicht in Config gefunden")

        # Konvertiere Datum-Strings zu datetime
        start = datetime.datetime.strptime(period_config["start"], "%Y-%m-%d")
        end = datetime.datetime.strptime(period_config["end"], "%Y-%m-%d")
        interval = period_config["interval"]

        print(f"  Zeitraum: {start.date()} bis {end.date()}")
        print(f"  Interval: {interval}")

        # Portfolio Daten (Aktien)
        print(f"  Hole Portfolio-Daten ({len(self.config.get('data.universe'))} Aktien)...")
        portfolio_df = LS.getHistoryData(
            universe=self.config.get("data.universe"),
            fields=self.config.get("data.fields"),
            start=start,
            end=end,
            interval=interval
        )

        # Index Daten (DAX, SDAX, VDAX)
        print(f"  Hole Index-Daten ({len(self.config.get('data.indices'))} Indizes)...")
        index_df = LS.getHistoryData(
            universe=self.config.get("data.indices"),
            fields=["TRDPRC_1"],  # Nur Preis für Indizes
            start=start,
            end=end,
            interval=interval
        )

        # Kombiniere Portfolio und Index Daten
        print(f"  Kombiniere Daten...")
        combined_df = self.combine_data(portfolio_df, index_df)

        # Speichere als Excel (wie in Version 1)
        print(f"  Speichere als Excel...")
        self.exceltextwriter(combined_df, f"combined_{period_type}")

        print(f"  ✓ {period_type.capitalize()} Daten erfolgreich geladen: {combined_df.shape}")
        return combined_df

    def combine_data(self, portfolio_df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
        """
        Kombiniert Portfolio und Index Daten

        Args:
            portfolio_df: DataFrame mit Portfolio-Daten
            index_df: DataFrame mit Index-Daten

        Returns:
            Kombiniertes DataFrame
        """
        # Alle Daten entlang der Spalten zusammenfügen
        # Indizes sollten bereits aligned sein (gleiche Zeitpunkte)
        combined_df = pd.concat([portfolio_df, index_df], axis=1)

        # Entferne Duplikate in Spaltennamen falls vorhanden
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

        return combined_df

    def exceltextwriter(self, df: pd.DataFrame, name: str) -> None:
        """
        Speichert DataFrame als Excel (Original-Funktion aus Version 1)

        Args:
            df: Zu speichernder DataFrame
            name: Name der Excel-Datei (ohne .xlsx)

        Raises:
            IOError: Wenn Excel-Datei nicht geschrieben werden kann
        """
        if df is None or df.empty:
            logger.warning("Keine Daten zurückgegeben – keine Excel erstellt.")
            print("⚠️ Keine Daten zurückgegeben – keine Excel erstellt.")
            return

        # Datumsspalte aus dem Index herstellen, falls nötig
        d = df.copy()
        if isinstance(d.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            d = d.reset_index()
            # Nach reset_index heißt die Spalte entweder nach Index-Name oder 'index'
            if "index" in d.columns:
                d = d.rename(columns={"index": "Date"})
            elif d.columns[0] not in ("Date", "Datetime"):
                d = d.rename(columns={d.columns[0]: "Date"})
        else:
            # Alternativ: vorhandene Datums-/Zeitspalten erkennen und vereinheitlichen
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
                sheet_name = str(column).strip()[:31] or "sheet"
                if "Date" in d.columns:
                    frame = d[["Date", column]].copy()
                else:
                    # Fallback: ohne Datums-Spalte
                    frame = d[[column]].copy()
                # Spaltennamen vereinheitlichen, damit keine MultiIndex-Header entstehen
                if "Date" in frame.columns and frame.shape[1] == 2:
                    frame.columns = ["Date", "Price"]
                elif frame.shape[1] == 1:
                    frame.columns = ["Price"]

                frame.to_excel(writer, index=False, sheet_name=sheet_name)

        logger.info(f"Excel gespeichert: {out_path}")
        print(f"    ✓ Excel gespeichert: {out_path}")
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
    Daten holen und als Excel speichern (Kompatibilitätsfunktion)

    Args:
        universe: Liste von Aktien/Indizes
        fields: Liste von Feldern
        start: Start-Datum
        end: End-Datum
        interval: Intervall (D, 30min, etc.)
        name: Name der Excel-Datei
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
    all_data = grabber.fetch_all_data()
    
    for portfolio_name, portfolio_data in all_data.items():
        logger.info(f"{portfolio_name}:")
        logger.info(f"  Daily: {portfolio_data['daily'].shape}")
        logger.info(f"  Intraday: {portfolio_data['intraday'].shape}")
        print(f"\n{portfolio_name}:")
        print(f"  Daily: {portfolio_data['daily'].shape}")
        print(f"  Intraday: {portfolio_data['intraday'].shape}")
