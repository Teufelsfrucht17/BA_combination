"""
Dataprep.py - Modifizierte Version für BA_combination
Bereitet Daten für Training vor basierend auf config.yaml
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from ConfigManager import ConfigManager


class DataPrep:
    """Bereitet Daten für Machine Learning vor"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialisiert DataPrep

        Args:
            config_path: Pfad zur Config-Datei
        """
        self.config = ConfigManager(config_path)

    def prepare_data(self, df: pd.DataFrame, portfolio_name: str = None, period_type: str = "daily"):
        """
        Bereitet Daten für Training vor

        Args:
            df: Rohdaten-DataFrame
            portfolio_name: Name des Portfolios (z.B. "dax", "sdax")
            period_type: "daily" oder "intraday"

        Returns:
            Tuple von (X, y) - Features und Target
        """
        portfolio_label = portfolio_name.upper() if portfolio_name else "UNKNOWN"
        print(f"\n{'='*60}")
        print(f"DATENAUFBEREITUNG - {portfolio_label} {period_type.upper()}")
        print(f"{'='*60}")

        # Feature Engineering
        print("Erstelle Features...")
        features_df = self.create_features(df, portfolio_name=portfolio_name)
        print(f"  ✓ Features erstellt: {features_df.shape}")

        # X und Y erstellen basierend auf Config
        print("Erstelle X und Y...")
        X, y = self.create_xy(features_df, portfolio_name=portfolio_name)
        print(f"  ✓ X Shape: {X.shape}")
        print(f"  ✓ y Shape: {y.shape}")

        print(f"{'='*60}\n")
        return X, y

    def create_features(self, df: pd.DataFrame, portfolio_name: str = None) -> pd.DataFrame:
        """
        Erstellt Features basierend auf Config und Portfolio

        Args:
            df: Rohdaten mit TRDPRC_1, VOLUME, etc.
            portfolio_name: Name des Portfolios (z.B. "dax", "sdax")

        Returns:
            DataFrame mit berechneten Features
        """
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
            portfolio_returns = df[stock_columns].pct_change().mean(axis=1)
        else:
            portfolio_returns = portfolio_prices.pct_change()

        features['price_change_next'] = portfolio_returns.shift(-1)

        # Entferne NaN Werte
        features = features.dropna()

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

    def create_xy(self, features_df: pd.DataFrame, portfolio_name: str = None) -> tuple[pd.DataFrame, pd.Series]:
        """
        Erstellt X und y basierend auf Config

        Args:
            features_df: DataFrame mit allen Features
            portfolio_name: Name des Portfolios (optional, für spezifische Features)

        Returns:
            Tuple von (X, y)
        """
        # Hole gewünschte Features aus Config
        input_features = self.config.get("features.input_features")
        target = self.config.get("features.target")

        # Filtere nur die Features die in Config aktiviert sind und verfügbar
        available_features = [f for f in input_features if f in features_df.columns]

        if len(available_features) == 0:
            raise ValueError(
                f"Keine Features gefunden!\n"
                f"Angefordert: {input_features}\n"
                f"Verfügbar: {list(features_df.columns)}"
            )

        # Erstelle X und y
        X = features_df[available_features].copy()
        y = features_df[target].copy()

        # Align X und y (entferne Zeilen mit NaN)
        common_index = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[common_index]
        y = y.loc[common_index]

        return X, y


def time_series_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, min_train_size: int = 50, min_test_size: int = 10):
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
    if n_samples < 100:
        print(f"⚠️  Warnung: Sehr kleiner Datensatz ({n_samples} Samples). "
              f"Ergebnisse könnten nicht aussagekräftig sein.")

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
