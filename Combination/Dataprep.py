"""
Dataprep.py - Modified version for BA_combination
Prepares data for training based on config.yaml
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from ConfigManager import ConfigManager

DEFAULT_MIN_TRAIN_SIZE = 50
DEFAULT_MIN_TEST_SIZE = 10
DEFAULT_WARNING_DATASET_SIZE = 100
DEFAULT_MISSING_VALUES_WARNING_THRESHOLD = 10.0
DEFAULT_MISSING_VALUES_ERROR_THRESHOLD = 50.0
DEFAULT_EPSILON = 1e-8

AVAILABLE_FEATURES = {
    'momentum_5', 'momentum_10', 'momentum_20',
    'portfolio_index_change', 'change_dax', 'change_sdax',
    'vdax_absolute', 'volume_ratio',
    'rolling_volatility_10', 'rolling_volatility_20',
    'rsi_14',
    'Mkt_Rf', 'SMB', 'HML', 'WML'
}


class DataPrep:
    """Prepare data for machine learning"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize DataPrep

        Args:
            config_path: Path to the config file
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
        Prepare data for training

        Args:
            df: Raw data DataFrame
            portfolio_name: Portfolio name (e.g. "dax", "sdax")
            period_type: "daily" or "intraday"

        Returns:
            Tuple of (X, y) - features and target

        """
        features_df = self.create_features(df, portfolio_name=portfolio_name, ff_factors=ff_factors, period_type=period_type)

        X, y = self.create_xy(features_df, portfolio_name=portfolio_name)
        return X, y

    def create_features(
        self, 
        df: pd.DataFrame, 
        portfolio_name: Optional[str] = None,
        ff_factors: Optional[pd.DataFrame] = None,
        period_type: str = "daily"
    ) -> pd.DataFrame:
        """
        Create features based on config and portfolio settings

        Args:
            df: Raw data with TRDPRC_1, volume, etc.
            portfolio_name: Portfolio name (e.g. "dax", "sdax")

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        if 'Date' in df.columns and not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.set_index('Date')
        elif not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            df.index = pd.to_datetime(df.index, errors='coerce')

        price_like = [col for col in df.columns if 'TRDPRC_1' in str(col) or 'Price' in str(col) or 'HIGH_1' in str(col) or 'LOW_1' in str(col)]
        volume_like = [col for col in df.columns if 'VOLUME' in str(col) or 'ACVOL_1' in str(col)]
        if price_like:
            df[price_like] = df[price_like].ffill().bfill()
        if volume_like:
            df[volume_like] = df[volume_like].ffill().bfill()

        features = pd.DataFrame(index=df.index)

        if portfolio_name:
            portfolio_config = self.config.get(f"data.portfolios.{portfolio_name}")
            index_identifier = portfolio_config.get("index", "").replace(".", "")
            index_feature_name = portfolio_config.get("index_feature", "portfolio_index_change")
        else:
            index_identifier = None
            index_feature_name = "portfolio_index_change"

        stock_price_cols = [col for col in df.columns if '.DE' in str(col) and 'TRDPRC_1' in str(col)]

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
            stock_price_cols = [col for col in df.columns if '.DE' in col and 'TRDPRC_1' in col]

        if len(stock_price_cols) > 0:
            # Fall 1: Es gibt echte TRDPRC_1-Preise für die Aktien -> wie bisher Mittelwert
            portfolio_prices = df[stock_price_cols].mean(axis=1)
        else:
            # Fall 2: Keine TRDPRC_1 für Aktien (z.B. sdax_daily) -> Mid-Price aus HIGH_1 und LOW_1
            high_cols = [c for c in df.columns if '.DE' in str(c) and 'HIGH_1' in str(c)]
            low_cols = [c for c in df.columns if '.DE' in str(c) and 'LOW_1' in str(c)]

            mid_series = []
            for h_col in high_cols:
                stem = str(h_col).replace('HIGH_1', '').rstrip('_')
                # Versuche ein passendes LOW_1 zu finden (MultiIndex- oder String-Namen tolerant)
                candidates = [c for c in low_cols if str(c).startswith(stem)]
                if not candidates:
                    continue
                l_col = candidates[0]
                mid = (df[h_col] + df[l_col]) / 2.0
                mid_series.append(mid)

            if mid_series:
                portfolio_prices = pd.concat(mid_series, axis=1).mean(axis=1)
            else:
                # letzter Fallback: erste numerische Spalte
                portfolio_prices = df.select_dtypes(include=[np.number]).iloc[:, 0]

        for period in self.config.get("features.momentum_periods", [5, 10, 20]):
            features[f'momentum_{period}'] = portfolio_prices.pct_change(period)

        if index_identifier:
            index_columns = [col for col in df.columns if index_identifier in col]
            if len(index_columns) > 0:
                index_prices = df[index_columns[0]]
                features[index_feature_name] = index_prices.pct_change()
                features['portfolio_index_change'] = features[index_feature_name]
            else:
                features[index_feature_name] = 0.0
                features['portfolio_index_change'] = 0.0
        else:
            features['portfolio_index_change'] = 0.0

        vdax_columns = [col for col in df.columns if 'V1XI' in col]
        if len(vdax_columns) > 0:
            features['vdax_absolute'] = df[vdax_columns[0]].abs()
        else:
            features['vdax_absolute'] = 0.0

        volume_columns = [col for col in df.columns if 'VOLUME' in col and '.DE' in col]
        if len(volume_columns) > 0:
            portfolio_volume = df[volume_columns].mean(axis=1)
            rolling_window = self.config.get("features.rolling_window", 20)
            features['volume_ratio'] = portfolio_volume / portfolio_volume.rolling(rolling_window).mean()
        else:
            features['volume_ratio'] = 1.0

        if 'rsi_14' in self.config.get("features.input_features", []):
            features['rsi_14'] = self.calculate_rsi(portfolio_prices, period=14)

        clipped = portfolio_prices.clip(lower=DEFAULT_EPSILON)
        portfolio_returns = np.log(clipped / clipped.shift(1))

        features['price_change_next'] = portfolio_returns.shift(-1)

        for window in self.config.get("features.volatility_windows", []):
            features[f'rolling_volatility_{window}'] = portfolio_returns.rolling(window, min_periods=window).std()

        if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            dt_index = df.index
            if isinstance(dt_index, pd.PeriodIndex):
                dt_index = dt_index.to_timestamp()
    
            features['day_of_week'] = dt_index.weekday
            features['hour_of_day'] = dt_index.hour
        else:
            features['day_of_week'] = 0
            features['hour_of_day'] = 0

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

        features = features.dropna(subset=['price_change_next'])

        # Add FFC factors BEFORE dropna() to ensure they are included
        if ff_factors is not None and not ff_factors.empty:
            # Use a robust merge to join Fama-French factors.
            # This handles both daily and intraday data by merging on the normalized date.
            ff_factors_daily = ff_factors.copy()
            ff_factors_daily.index = pd.to_datetime(ff_factors_daily.index).normalize()

            # Create a temporary key on the features DataFrame for the merge
            features['merge_key'] = pd.to_datetime(features.index).normalize()

            ff_cols_to_merge = [col for col in ['Mkt_Rf', 'SMB', 'HML', 'WML'] if col in ff_factors_daily.columns]

            if ff_cols_to_merge:
                # Store original feature count
                n_features_before = len(features.columns)
                
                features = pd.merge(
                    features,
                    ff_factors_daily[ff_cols_to_merge],
                    left_on='merge_key',
                    right_index=True,
                    how='left',
                    sort=False  # Preserve the original time-series order
                )
                
                # Check if FFC factors were actually added
                n_features_after = len(features.columns)
                ffc_added = [col for col in ff_cols_to_merge if col in features.columns]
                
                if len(ffc_added) > 0:
                    # Check for NaN values in FFC columns
                    ffc_nan_counts = features[ffc_added].isna().sum()
                    total_rows = len(features)
                    ffc_valid_ratio = (total_rows - ffc_nan_counts.max()) / total_rows if total_rows > 0 else 0
                    
                    # Only warn if most values are NaN
                    if ffc_valid_ratio < 0.5:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"FFC-Faktoren wurden hinzugefügt, aber {ffc_nan_counts.max()} von {total_rows} "
                            f"Werten sind NaN. Validitätsrate: {ffc_valid_ratio:.2%}"
                        )
            else:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"FFC-Faktoren DataFrame vorhanden, aber keine erwarteten Spalten gefunden. "
                    f"Verfügbare Spalten: {list(ff_factors_daily.columns)}"
                )

            # Clean up the temporary key
            if 'merge_key' in features.columns:
                features = features.drop(columns=['merge_key'])

        # Identify FFC factor columns before filling
        ff_feature_names = ['Mkt_Rf', 'SMB', 'HML', 'WML']
        ffc_cols = [col for col in ff_feature_names if col in features.columns]
        
        # Fill missing values for all columns except target and FFC factors
        # FFC factors should be filled separately to preserve their structure
        fillable_cols = [c for c in features.columns if c not in ['price_change_next', 'price_direction_next'] + ffc_cols]
        if fillable_cols:
            features[fillable_cols] = features[fillable_cols].ffill()
        
        # For FFC factors, forward fill and backward fill to minimize NaN values
        if ffc_cols:
            # Forward fill and backward fill FFC factors to preserve as much data as possible
            features[ffc_cols] = features[ffc_cols].ffill().bfill()
            
            # Log FFC factor statistics
            import logging
            logger = logging.getLogger(__name__)
            ffc_nan_after_fill = features[ffc_cols].isna().sum()
            if ffc_nan_after_fill.sum() > 0:
                logger.warning(f"FFC-Faktoren haben nach ffill/bfill noch NaN: {ffc_nan_after_fill.to_dict()}")
            
            # Only drop rows where non-FFC features are NaN
            non_ffc_cols = [c for c in features.columns if c not in ffc_cols and c != 'price_direction_next']
            features = features.dropna(subset=non_ffc_cols)
        else:
            # No FFC factors, drop all NaN rows as before (except target)
            features = features.dropna(subset=[c for c in features.columns if c not in ['price_change_next', 'price_direction_next']])

        features['price_direction_next'] = (features['price_change_next'] > 0).astype(int)

        return features

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)

        Args:
            prices: Price series
            period: RSI period (default: 14)

        Returns:
            RSI values
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
        Build X and y based on config

        Args:
            features_df: DataFrame with all features
            portfolio_name: Portfolio name (optional, for specific features)

        Returns:
            Tuple of (X, y)

        Raises:
            ValueError: If no features are found
        """
        input_features = self.config.get("features.input_features", [])
        target = self.config.get("features.target", "price_change_next")

        available_features = [f for f in input_features if f in features_df.columns]
        missing_features = set(input_features) - set(available_features)

        # Wenn Fama-French-Faktoren vorhanden sind, immer mit aufnehmen (nur im FFC-Run vorhanden)
        ff_feature_names = ['Mkt_Rf', 'SMB', 'HML', 'WML']
        ffc_features_found = []
        for col in ff_feature_names:
            if col in features_df.columns and col not in available_features:
                available_features.append(col)
                ffc_features_found.append(col)
        
        # Debug: Log if FFC features are present
        if ffc_features_found:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"FFC-Faktoren in Features aufgenommen: {ffc_features_found}")
            # Check for NaN values
            ffc_nan_info = features_df[ffc_features_found].isna().sum()
            if ffc_nan_info.sum() > 0:
                logger.warning(f"FFC-Faktoren enthalten NaN-Werte: {ffc_nan_info.to_dict()}")

        if len(available_features) == 0:
            available_features = list(features_df.columns)

        X = features_df[available_features].copy()
        y = features_df[target].copy()

        # Don't drop rows based on FFC factors - they may have NaN values
        # Only drop rows where non-FFC features or target are NaN
        ff_feature_names = ['Mkt_Rf', 'SMB', 'HML', 'WML']
        ffc_cols_in_X = [col for col in ff_feature_names if col in X.columns]
        non_ffc_cols = [col for col in X.columns if col not in ffc_cols_in_X]
        
        if non_ffc_cols:
            # Drop rows where non-FFC features are NaN
            X_valid = X[non_ffc_cols].dropna()
            y_valid = y.dropna()
            common_index = X_valid.index.intersection(y_valid.index)
        else:
            # All columns are FFC factors, only check target
            y_valid = y.dropna()
            common_index = y_valid.index
        
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
    Chronological split: earlier data -> train, later -> test

    Args:
        X: Features
        y: Target
        test_size: Share for test set (default: 0.2)
        min_train_size: Minimum samples in the train set
        min_test_size: Minimum samples in the test set

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)

    """
    n_samples = len(X)

    split_idx = int(n_samples * (1 - test_size))
    n_train = split_idx
    n_test = n_samples - split_idx

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    prep = DataPrep()
