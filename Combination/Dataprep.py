"""
Dataprep.py - Modifizierte Version für BA_combination
Bereitet Daten für Training vor basierend auf config.yaml
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from ConfigManager import ConfigManager
from logger_config import get_logger

logger = get_logger(__name__)

# Constants
DEFAULT_MIN_TRAIN_SIZE = 50
DEFAULT_MIN_TEST_SIZE = 10
DEFAULT_WARNING_DATASET_SIZE = 100
DEFAULT_MISSING_VALUES_WARNING_THRESHOLD = 10.0
DEFAULT_MISSING_VALUES_ERROR_THRESHOLD = 50.0
DEFAULT_EPSILON = 1e-8

# Verfügbare Features
AVAILABLE_FEATURES = {
    'momentum_5', 'momentum_10', 'momentum_20',
    'portfolio_index_change', 'change_dax', 'change_sdax',
    'vdax_absolute', 'volume_ratio',
    'rolling_volatility_10', 'rolling_volatility_20',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'rsi_14',
    # Fama-French/Carhart Faktoren
    'Mkt_Rf', 'SMB', 'HML', 'WML'
}


class DataPrep:
    """Bereitet Daten für Machine Learning vor"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialisiert DataPrep

        Args:
            config_path: Pfad zur Config-Datei
        """
        self.config = ConfigManager(config_path)

    def prepare_data(
        self, 
        df: pd.DataFrame, 
        portfolio_name: Optional[str] = None, 
        period_type: str = "daily",
        ff_factors: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Bereitet Daten für Training vor

        Args:
            df: Rohdaten-DataFrame
            portfolio_name: Name des Portfolios (z.B. "dax", "sdax")
            period_type: "daily" oder "intraday"

        Returns:
            Tuple von (X, y) - Features und Target

        Raises:
            ValueError: Wenn Daten leer oder ungültig sind
        """
        # Validierung
        if df.empty:
            raise ValueError("DataFrame ist leer!")
        
        if len(df) < DEFAULT_WARNING_DATASET_SIZE:
            logger.warning("Sehr kleine Datenmenge: %d Zeilen", len(df))

        # Prüfe auf fehlende Werte
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        if missing_pct > DEFAULT_MISSING_VALUES_ERROR_THRESHOLD:
            raise ValueError(f"Zu viele fehlende Werte: {missing_pct:.1f}%")
        elif missing_pct > DEFAULT_MISSING_VALUES_WARNING_THRESHOLD:
            logger.warning("Viele fehlende Werte: %.1f%%", missing_pct)

        # Prüfe auf Duplikate im Index
        if df.index.duplicated().any():
            logger.warning("Duplikate im Index gefunden, entferne sie...")
            df = df[~df.index.duplicated(keep='first')]

        portfolio_label = portfolio_name.upper() if portfolio_name else "UNKNOWN"
        logger.info("="*60)
        logger.info(f"DATENAUFBEREITUNG - {portfolio_label} {period_type.upper()}")
        logger.info("="*60)
        print(f"\n{'='*60}")
        print(f"DATENAUFBEREITUNG - {portfolio_label} {period_type.upper()}")
        print(f"{'='*60}")

        # Feature Engineering
        logger.info("Erstelle Features...")
        print("Erstelle Features...")
        features_df = self.create_features(df, portfolio_name=portfolio_name, ff_factors=ff_factors, period_type=period_type)
        logger.info(f"Features erstellt: {features_df.shape}")
        print(f"  ✓ Features erstellt: {features_df.shape}")

        # X und Y erstellen basierend auf Config
        logger.info("Erstelle X und Y...")
        print("Erstelle X und Y...")
        X, y = self.create_xy(features_df, portfolio_name=portfolio_name)
        logger.info(f"X Shape: {X.shape}, y Shape: {y.shape}")
        print(f"  ✓ X Shape: {X.shape}")
        print(f"  ✓ y Shape: {y.shape}")

        logger.info("="*60)
        print(f"{'='*60}\n")
        return X, y

    def create_features(
        self, 
        df: pd.DataFrame, 
        portfolio_name: Optional[str] = None,
        ff_factors: Optional[pd.DataFrame] = None,
        period_type: str = "daily"
    ) -> pd.DataFrame:
        """
        Erstellt Features basierend auf Config und Portfolio

        Args:
            df: Rohdaten mit TRDPRC_1, VOLUME, etc.
            portfolio_name: Name des Portfolios (z.B. "dax", "sdax")

        Returns:
            DataFrame mit berechneten Features
        """
        # Defensive Kopie und Datetime-Index erzwingen
        df = df.copy()
        if 'Date' in df.columns and not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.set_index('Date')
        elif not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            # Fallback: versuche Index zu parsen
            df.index = pd.to_datetime(df.index, errors='coerce')

        # Fehlende Werte frühzeitig behandeln (Preise/Volumen)
        price_like = [col for col in df.columns if 'TRDPRC_1' in str(col) or 'Price' in str(col)]
        volume_like = [col for col in df.columns if 'VOLUME' in str(col) or 'ACVOL_1' in str(col)]
        if price_like:
            df[price_like] = df[price_like].ffill().bfill()
        if volume_like:
            df[volume_like] = df[volume_like].ffill().bfill()

        features = pd.DataFrame(index=df.index)

        # Hole Portfolio-Konfiguration für Index-Feature-Namen
        if portfolio_name:
            portfolio_config = self.config.get(f"data.portfolios.{portfolio_name}")
            index_identifier = portfolio_config.get("index", "").replace(".", "")  # .GDAXI -> GDAXI
            index_feature_name = portfolio_config.get("index_feature", "portfolio_index_change")
        else:
            index_identifier = None
            index_feature_name = "portfolio_index_change"

        # Identifiziere Aktien-Spalten (enden mit .DE)
        stock_columns = [col for col in df.columns if '.DE' in str(col) and 'TRDPRC_1' in str(col)]

        # Falls Multi-Level Columns, flatten
        if isinstance(df.columns, pd.MultiIndex):
            # Erstelle flache Spaltennamen
            df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
            stock_columns = [col for col in df.columns if '.DE' in col and 'TRDPRC_1' in col]

        # Berechne Portfolio-Durchschnittspreis (alle Aktien)
        if len(stock_columns) > 0:
            portfolio_prices = df[stock_columns].mean(axis=1)
        else:
            # Fallback: verwende erste numerische Spalte
            portfolio_prices = df.select_dtypes(include=[np.number]).iloc[:, 0]

        # ==========================================
        # Momentum Features (aus Config)
        # ==========================================
        for period in self.config.get("features.momentum_periods", [5, 10, 20]):
            features[f'momentum_{period}'] = portfolio_prices.pct_change(period)

        # ==========================================
        # Index Features (Portfolio-spezifisch)
        # ==========================================

        # Portfolio Index Change (DAX oder SDAX je nach Portfolio)
        if index_identifier:
            index_columns = [col for col in df.columns if index_identifier in col]
            if len(index_columns) > 0:
                index_prices = df[index_columns[0]]
                features[index_feature_name] = index_prices.pct_change()
                # Aliasiere auch als portfolio_index_change für generische Config
                features['portfolio_index_change'] = features[index_feature_name]
            else:
                features[index_feature_name] = 0.0
                features['portfolio_index_change'] = 0.0
        else:
            features['portfolio_index_change'] = 0.0

        # VDAX Absolute (Volatilität) - gemeinsam für alle Portfolios
        vdax_columns = [col for col in df.columns if 'V1XI' in col]
        if len(vdax_columns) > 0:
            features['vdax_absolute'] = df[vdax_columns[0]].abs()
        else:
            features['vdax_absolute'] = 0.0

        # ==========================================
        # Volume Features
        # ==========================================
        volume_columns = [col for col in df.columns if 'VOLUME' in col and '.DE' in col]
        if len(volume_columns) > 0:
            portfolio_volume = df[volume_columns].mean(axis=1)
            rolling_window = self.config.get("features.rolling_window", 20)
            features['volume_ratio'] = portfolio_volume / portfolio_volume.rolling(rolling_window).mean()
        else:
            features['volume_ratio'] = 1.0

        # ==========================================
        # Optional: RSI (falls in Config aktiviert)
        # ==========================================
        if 'rsi_14' in self.config.get("features.input_features", []):
            features['rsi_14'] = self.calculate_rsi(portfolio_prices, period=14)

        # ==========================================
        # Y Variable: Nächste Preisänderung
        # ==========================================
        # Berechne zukünftige Returns (für alle Aktien)
        if len(stock_columns) > 0:
            clipped = df[stock_columns].clip(lower=DEFAULT_EPSILON)
            portfolio_returns = np.log(clipped / clipped.shift(1)).mean(axis=1)
        else:
            clipped = portfolio_prices.clip(lower=DEFAULT_EPSILON)
            portfolio_returns = np.log(clipped / clipped.shift(1))

        features['price_change_next'] = portfolio_returns.shift(-1)

        # ==========================================
        # Volatilitäts-Features
        # ==========================================
        for window in self.config.get("features.volatility_windows", []):
            features[f'rolling_volatility_{window}'] = portfolio_returns.rolling(window, min_periods=window).std()

        # ==========================================
        # Zeit-/Kalender-Features
        # ==========================================
        if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            dt_index = df.index
            if isinstance(dt_index, pd.PeriodIndex):
                dt_index = dt_index.to_timestamp()
    
            dow = dt_index.weekday
            features['day_of_week'] = dow
            # Zyklische Kodierung
            features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
            features['dow_cos'] = np.cos(2 * np.pi * dow / 7)

            hours = dt_index.hour
            features['hour_of_day'] = hours
            features['hour_sin'] = np.sin(2 * np.pi * hours / 24)
            features['hour_cos'] = np.cos(2 * np.pi * hours / 24)
        else:
            # Fallback: keine Zeitfeatures
            features['day_of_week'] = 0
            features['dow_sin'] = 0
            features['dow_cos'] = 0
            features['hour_of_day'] = 0
            features['hour_sin'] = 0
            features['hour_cos'] = 0

        # Entferne NaN Werte
        # Initiale Droppings nur für Fenster-Effekte
        drop_n = 0
        momentum_periods = self.config.get("features.momentum_periods", [])
        if momentum_periods:
            drop_n = max(drop_n, max(momentum_periods))
        rolling_window = self.config.get("features.rolling_window", 0)
        drop_n = max(drop_n, rolling_window)
        volatility_windows = self.config.get("features.volatility_windows", [])
        if volatility_windows:
            drop_n = max(drop_n, max(volatility_windows))

        if drop_n > 0 and len(features) > drop_n:
            features = features.iloc[drop_n:]

        # Preisänderung muss vorhanden sein, sonst später kein Target
        features = features.dropna(subset=['price_change_next'])

        # Auffüllen verbleibender vereinzelter NaNs (falls Imputation nicht alles abgedeckt hat)
        fillable_cols = [c for c in features.columns if c not in ['price_change_next', 'price_direction_next']]
        features[fillable_cols] = features[fillable_cols].ffill()
        features = features.dropna()

        # ==========================================
        # Fama-French/Carhart Faktoren (optional)
        # ==========================================
        if ff_factors is not None and not ff_factors.empty:
            # Merge FFC-Faktoren basierend auf Datum
            # Für Intraday: Company-Daten müssen bereits auf Tagesbasis zugeordnet sein
            if isinstance(features.index, pd.DatetimeIndex) and isinstance(ff_factors.index, pd.DatetimeIndex):
                # Align FFC-Faktoren mit Features-Index
                # Bei Intraday: Use date (ohne Zeit) für Alignment
                if period_type == "intraday" or features.index.hour.nunique() > 1:
                    # Intraday: Extrahiere nur Datum für Alignment
                    features_date = features.index.normalize()
                    ff_factors_date = ff_factors.index.normalize()
                    
                    # Erstelle Mapping: Datum -> FFC-Faktoren
                    ff_dict = {}
                    for date, row in ff_factors.iterrows():
                        date_only = pd.Timestamp(date).normalize()
                        if date_only not in ff_dict:
                            ff_dict[date_only] = row
                    
                    # Ordne FFC-Faktoren zu (gleiche Werte für alle Intervalle eines Tages)
                    for date_idx in features.index:
                        date_only = pd.Timestamp(date_idx).normalize()
                        if date_only in ff_dict:
                            ff_row = ff_dict[date_only]
                            for col in ['Mkt_Rf', 'SMB', 'HML', 'WML']:
                                if col in ff_row:
                                    features.loc[date_idx, col] = ff_row[col]
                else:
                    # Daily: Direktes Alignment
                    for col in ['Mkt_Rf', 'SMB', 'HML', 'WML']:
                        if col in ff_factors.columns:
                            features[col] = ff_factors[col]
                
                logger.info("FFC-Faktoren hinzugefügt: Mkt_Rf, SMB, HML, WML")
            else:
                logger.warning("FFC-Faktoren konnten nicht zugeordnet werden (Index-Problem)")

        # ==========================================
        # Optionale Klassifikations-Target
        # ==========================================
        features['price_direction_next'] = (features['price_change_next'] > 0).astype(int)

        return features

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Berechnet Relative Strength Index (RSI)

        Args:
            prices: Preis-Serie
            period: RSI-Periode (Standard: 14)

        Returns:
            RSI-Werte
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def create_xy(
        self, 
        features_df: pd.DataFrame, 
        portfolio_name: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Erstellt X und y basierend auf Config

        Args:
            features_df: DataFrame mit allen Features
            portfolio_name: Name des Portfolios (optional, für spezifische Features)

        Returns:
            Tuple von (X, y)

        Raises:
            ValueError: Wenn keine Features gefunden werden
        """
        # Hole gewünschte Features aus Config
        input_features = self.config.get("features.input_features", [])
        target = self.config.get("features.target", "price_change_next")

        # Validiere Features
        invalid_features = set(input_features) - AVAILABLE_FEATURES
        if invalid_features:
            logger.warning(
                "Unbekannte Features in Config: %s. Verfügbare: %s",
                invalid_features, sorted(AVAILABLE_FEATURES)
            )

        # Filtere nur die Features die in Config aktiviert sind und verfügbar
        available_features = [f for f in input_features if f in features_df.columns]
        missing_features = set(input_features) - set(available_features)

        if missing_features:
            logger.warning("Features in Config aber nicht verfügbar: %s", missing_features)

        if len(available_features) == 0:
            error_msg = (
                f"Keine Features gefunden!\n"
                f"Angefordert: {input_features}\n"
                f"Verfügbar: {list(features_df.columns)}\n"
                f"Verfügbare Features: {sorted(AVAILABLE_FEATURES)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Erstelle X und y
        X = features_df[available_features].copy()
        y = features_df[target].copy()

        # Align X und y (entferne Zeilen mit NaN)
        common_index = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[common_index]
        y = y.loc[common_index]

        return X, y


def time_series_split(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2, 
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE, 
    min_test_size: int = DEFAULT_MIN_TEST_SIZE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Chronologischer Split: frühe Daten -> Training, späte -> Test

    Args:
        X: Features
        y: Target
        test_size: Anteil für Test-Set (Standard: 0.2)
        min_train_size: Minimale Anzahl Samples im Trainingsset
        min_test_size: Minimale Anzahl Samples im Testset

    Returns:
        Tuple von (X_train, X_test, y_train, y_test)

    Raises:
        ValueError: Wenn nicht genügend Samples für sinnvollen Split vorhanden
    """
    n_samples = len(X)

    # Prüfe ob überhaupt Daten vorhanden
    if n_samples == 0:
        raise ValueError("Keine Datenpunkte vorhanden.")

    # Berechne Split-Index
    split_idx = int(n_samples * (1 - test_size))
    n_train = split_idx
    n_test = n_samples - split_idx

    # Validiere Split-Größen
    if n_train < min_train_size:
        raise ValueError(
            f"Trainingsset zu klein: {n_train} Samples (Minimum: {min_train_size}). "
            f"Bitte reduziere test_size oder erhöhe die Datenmenge."
        )

    if n_test < min_test_size:
        raise ValueError(
            f"Testset zu klein: {n_test} Samples (Minimum: {min_test_size}). "
            f"Bitte erhöhe test_size oder erhöhe die Datenmenge."
        )

    # Durchführe Split
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # Warne bei sehr kleinen Datasets
    if n_samples < DEFAULT_WARNING_DATASET_SIZE:
        logger.warning(
            "Sehr kleiner Datensatz (%d Samples). Ergebnisse könnten nicht aussagekräftig sein.",
            n_samples
        )
        print(f"⚠️  Warnung: Sehr kleiner Datensatz ({n_samples} Samples). "
              f"Ergebnisse könnten nicht aussagekräftig sein.")

    logger.debug(
        "Time Series Split: Train=%d, Test=%d (test_size=%.2f)",
        n_train, n_test, test_size
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test
    print("DataPrep Test")

    # Erstelle Test-Daten
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    test_df = pd.DataFrame({
        'SAP.DE_TRDPRC_1': np.random.randn(1000).cumsum() + 100,
        'SIE.DE_TRDPRC_1': np.random.randn(1000).cumsum() + 80,
        'SAP.DE_VOLUME': np.random.randint(1000, 10000, 1000),
        'SIE.DE_VOLUME': np.random.randint(1000, 10000, 1000),
        '.GDAXI_TRDPRC_1': np.random.randn(1000).cumsum() + 15000,
        '.V1XI_TRDPRC_1': np.abs(np.random.randn(1000)) * 20,
    }, index=dates)

    prep = DataPrep()
    X, y = prep.prepare_data(test_df, "daily")

    print(f"\nX Shape: {X.shape}")
    print(f"y Shape: {y.shape}")
    print(f"\nX columns: {list(X.columns)}")
    print(f"\nErste 5 Zeilen von X:\n{X.head()}")
