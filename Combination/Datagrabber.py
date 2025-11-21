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

    def fetch_company_data(self) -> Dict[str, pd.DataFrame]:
        """
        Holt fundamentale Company-Daten für alle Portfolios (DAX, SDAX)
        
        Diese Daten werden für Fama-French/Carhart Modelle benötigt.
        Erstellt separate Excel-Dateien für jedes Portfolio.

        Returns:
            Dictionary: {portfolio_name: DataFrame}
            Beispiel: {"dax": df, "sdax": df}

        Raises:
            ValueError: Wenn Portfolio nicht gefunden wird
            RuntimeError: Wenn API-Calls fehlschlagen
        """
        logger.info("="*70)
        logger.info("COMPANY-DATENABRUF GESTARTET")
        logger.info("="*70)
        print("\n" + "="*70)
        print("COMPANY-DATENABRUF GESTARTET")
        print("="*70)

        all_company_data: Dict[str, pd.DataFrame] = {}
        portfolios = self.config.get("data.portfolios", {})

        if not portfolios:
            logger.warning("Keine Portfolios in Config gefunden")
            raise ValueError("Keine Portfolios in Config definiert")

        for portfolio_name in portfolios.keys():
            portfolio_config = self.config.get(f"data.portfolios.{portfolio_name}")
            if portfolio_config is None:
                logger.error(f"Portfolio '{portfolio_name}' nicht in Config gefunden")
                raise ValueError(f"Portfolio '{portfolio_name}' nicht in Config gefunden")

            portfolio_display = portfolio_config.get('name', portfolio_name.upper())
            logger.info("="*70)
            logger.info(f"PORTFOLIO: {portfolio_display} - COMPANY DATA")
            logger.info("="*70)
            print(f"\n{'='*70}")
            print(f"PORTFOLIO: {portfolio_display} - COMPANY DATA")
            print(f"{'='*70}")

            # Hole Universe (Aktien) aus Portfolio
            universe = portfolio_config["universe"]
            if not universe:
                logger.warning(f"Portfolio '{portfolio_name}' hat leeres Universe, überspringe...")
                continue

            logger.info(f"Hole Company-Daten für {len(universe)} Aktien...")
            print(f"  Hole Company-Daten ({len(universe)} Aktien)...")

            try:
                # Hole Company-Daten via LSEG API
                # Verwende gleichen Zeitraum wie Price-Daten
                period_config = self.config.get("data.periods.daily")  # Verwende daily Zeitraum
                if period_config:
                    # Erstelle Parameter mit gleichem Zeitraum wie Price-Daten
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
                    raise ValueError("Leere API-Antwort")

                # Speichere als Excel
                logger.info(f"Speichere als Excel: {portfolio_name}_company_data")
                print(f"  Speichere als Excel...")
                self.exceltextwriter(company_df, f"{portfolio_name}_company_data")

                all_company_data[portfolio_name] = company_df

                logger.info(f"{portfolio_name.upper()} Company-Daten geladen: {company_df.shape}")
                print(f"  ✓ {portfolio_name.upper()} Company-Daten geladen: {company_df.shape}")

            except Exception as e:
                logger.error(
                    f"Fehler beim Abrufen der Company-Daten für Portfolio '{portfolio_name}': {e}",
                    exc_info=True
                )
                print(f"  ✗ Fehler: {e}")

                # Fallback: versuche lokale Dateien zu laden
                fallback_df = self._load_company_data_from_storage(portfolio_name)
                if fallback_df is not None:
                    all_company_data[portfolio_name] = fallback_df
                    print(f"  ✓ Fallback-Daten aus DataStorage geladen: {fallback_df.shape}")
                    logger.info(
                        f"Fallback Company-Daten aus DataStorage für {portfolio_name} geladen: {fallback_df.shape}"
                    )
                else:
                    logger.warning(
                        f"Keine Company-Daten (API oder Fallback) für Portfolio '{portfolio_name}' verfügbar"
                    )
                    print("  ⚠️ Keine Company-Daten verfügbar (weder API noch DataStorage)")
                    # Weiter mit nächstem Portfolio
                    continue

        logger.info("="*70)
        logger.info("COMPANY-DATENABRUF ABGESCHLOSSEN")
        logger.info("="*70)
        print("\n" + "="*70)
        print("COMPANY-DATENABRUF ABGESCHLOSSEN")
        print("="*70)

        # Zeige Zusammenfassung
        if all_company_data:
            for portfolio_name, df in all_company_data.items():
                logger.info(f"{portfolio_name.upper()}: {df.shape}")
                print(f"\n{portfolio_name.upper()}: {df.shape}")
        else:
            logger.warning("Keine Company-Daten geladen")
            print("⚠️ Keine Company-Daten geladen")

        return all_company_data

    def _load_company_data_from_storage(self, portfolio_name: str) -> Optional[pd.DataFrame]:
        """Versucht Company-Daten aus dem DataStorage Verzeichnis zu laden."""

        storage_dir = Path("DataStorage")
        if not storage_dir.exists():
            logger.debug("DataStorage Verzeichnis nicht gefunden – kein Fallback möglich")
            return None

        # Suche nach Dateien, die dem Namensmuster entsprechen
        possible_files = sorted(storage_dir.glob(f"{portfolio_name}_company_data*.xlsx"), reverse=True)
        if not possible_files:
            logger.debug(f"Keine gespeicherten Company-Daten für {portfolio_name} gefunden")
            return None

        for file_path in possible_files:
            try:
                # Lade alle Sheets und setze sie zu einem DataFrame zusammen
                sheets = pd.read_excel(file_path, sheet_name=None)
                merged_df = None

                for sheet_name, sheet_df in sheets.items():
                    if sheet_df is None or sheet_df.empty:
                        continue

                    # Stelle sicher, dass eine Datums-Spalte existiert
                    if "Date" not in sheet_df.columns:
                        # Versuche erste Spalte als Datum zu interpretieren
                        sheet_df = sheet_df.rename(columns={sheet_df.columns[0]: "Date"})

                    # Konvertiere Date-Spalte und entferne offensichtliche Duplikate
                    sheet_df["Date"] = pd.to_datetime(sheet_df["Date"], errors="coerce")
                    sheet_df = sheet_df.dropna(subset=["Date"]).drop_duplicates(subset=["Date"])

                    if merged_df is None:
                        merged_df = sheet_df
                    else:
                        merged_df = pd.merge(merged_df, sheet_df, on="Date", how="outer")

                if merged_df is not None and not merged_df.empty:
                    merged_df = merged_df.sort_values("Date")
                    logger.info(
                        f"Geladene Fallback-Datei für {portfolio_name}: {file_path} – Spalten: {list(merged_df.columns)}"
                    )
                    return merged_df

                logger.warning(
                    f"Gefundene Fallback-Datei {file_path} enthält keine verwertbaren Daten – versuche nächste Datei"
                )
            except Exception as e:
                logger.error(f"Fehler beim Laden der Fallback-Datei {file_path}: {e}", exc_info=True)

        # Keine gültige Datei gefunden
        return None

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
                # Sheet-Name: Max 31 Zeichen (Excel-Limit), bereinige für ungültige Zeichen
                sheet_name = str(column).strip()[:31]
                # Entferne ungültige Zeichen für Excel Sheet-Namen
                sheet_name = sheet_name.replace("/", "_").replace("\\", "_").replace("?", "_")
                sheet_name = sheet_name.replace("*", "_").replace("[", "_").replace("]", "_")
                sheet_name = sheet_name.replace(":", "_").replace("'", "_")
                
                if not sheet_name:
                    sheet_name = "sheet"
                
                # Erstelle Frame mit Date und aktueller Spalte, behalte Originalnamen
                if "Date" in d.columns:
                    frame = d[["Date", column]].copy()
                else:
                    # Fallback: ohne Datums-Spalte
                    frame = d[[column]].copy()
                
                # Behalte die Original-Spaltennamen (keine Umbenennung in "Date" und "Price")
                # Nur bereinige MultiIndex falls vorhanden
                if isinstance(frame.columns, pd.MultiIndex):
                    frame.columns = [str(col).replace("/", "_") for col in frame.columns]

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
    
    # Hole historische Daten (Daily + Intraday)
    all_data = grabber.fetch_all_data()
    
    for portfolio_name, portfolio_data in all_data.items():
        logger.info(f"{portfolio_name}:")
        logger.info(f"  Daily: {portfolio_data['daily'].shape}")
        logger.info(f"  Intraday: {portfolio_data['intraday'].shape}")
        print(f"\n{portfolio_name}:")
        print(f"  Daily: {portfolio_data['daily'].shape}")
        print(f"  Intraday: {portfolio_data['intraday'].shape}")
    
    # Hole fundamentale Company-Daten (für Fama-French Modelle)
    print("\n" + "="*70)
    print("STARTE COMPANY-DATENABRUF")
    print("="*70)
    company_data = grabber.fetch_company_data()
    
    for portfolio_name, df in company_data.items():
        logger.info(f"{portfolio_name} Company Data: {df.shape}")
        print(f"\n{portfolio_name.upper()} Company Data: {df.shape}")
