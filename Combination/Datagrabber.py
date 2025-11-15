"""
Datagrabber.py - Erweiterte Version für BA_combination
Holt sowohl tägliche als auch 30-Min Daten basierend auf config.yaml
"""

import datetime
from pathlib import Path
import pandas as pd
import LSEG as LS
from ConfigManager import ConfigManager


class DataGrabber:
    """Erweiterte Version des DataGrabbers mit Config-Unterstützung"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialisiert DataGrabber

        Args:
            config_path: Pfad zur Config-Datei
        """
        self.config = ConfigManager(config_path)

    def fetch_all_data(self):
        """
        Holt sowohl tägliche als auch 30-Min Daten

        Returns:
            Tuple von (daily_data, intraday_data) als DataFrames
        """
        print("\n" + "="*70)
        print("DATENABRUF GESTARTET")
        print("="*70)

        # Tägliche Daten
        print("\n[1/2] Hole tägliche Daten...")
        daily_data = self.fetch_period_data("daily")

        # 30-Min Daten
        print("\n[2/2] Hole 30-Minuten Daten...")
        intraday_data = self.fetch_period_data("intraday")

        print("\n" + "="*70)
        print("DATENABRUF ABGESCHLOSSEN")
        print("="*70)
        print(f"Daily Daten: {daily_data.shape}")
        print(f"Intraday Daten: {intraday_data.shape}")

        return daily_data, intraday_data

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

        # Index Daten (DAX, VDAX)
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
        """
        if df is None or df.empty:
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

        print(f"    ✓ Excel gespeichert: {out_path}")
        return


def createExcel(
    universe: list[str],
    fields: list[str],
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
    grabber = DataGrabber()
    daily_data, intraday_data = grabber.fetch_all_data()
    print(f"\nDaily Data Shape: {daily_data.shape}")
    print(f"Intraday Data Shape: {intraday_data.shape}")
