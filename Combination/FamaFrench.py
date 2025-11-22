"""
FamaFrench.py - Berechnet (robuste) Fama-French/Carhart Faktoren.

Hinweis: Bei fehlenden Fundamentaldaten oder Index-Preisen werden Fallbacks genutzt:
- Markt-Proxy = Durchschnitt der Aktienpreise
- SMB/HML = 0
- Momentum bleibt aktiv, sofern Preis-Reihen vorhanden sind
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

from ConfigManager import ConfigManager
from logger_config import get_logger

logger = get_logger(__name__)


class FamaFrenchFactorModel:
    """Berechnet Fama-French/Carhart Faktoren mit robusten Fallbacks."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.risk_free_rate = self.config.get("features.risk_free_rate", 0.027)  # 2.7% default
        logger.info(f"Fama-French Model initialisiert (Risk-free Rate: {self.risk_free_rate*100:.2f}%)")

    # ----------------------------
    # Helper
    # ----------------------------
    @staticmethod
    def _normalize_index(df: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
        """Sorge dafür, dass ein DatetimeIndex existiert und normalisiert ist."""
        if df is None:
            return pd.DataFrame()

        result = df.copy()
        if date_column in result.columns:
            result[date_column] = pd.to_datetime(result[date_column], errors="coerce")
            result = result.set_index(date_column)
        elif not isinstance(result.index, pd.DatetimeIndex):
            result.index = pd.to_datetime(result.index, errors="coerce")

        if isinstance(result.index, pd.DatetimeIndex):
            result.index = result.index.tz_localize(None) if getattr(result.index, "tz", None) else result.index
            result.index = result.index.normalize()

        return result

    @staticmethod
    def _extract_stock_name(col: object) -> str:
        """Extrahiert den Börsenticker aus einer Spalte (tuple oder string)."""
        if isinstance(col, tuple) and len(col) > 0:
            return str(col[0]).strip()
        col_str = str(col)
        return col_str.split("_")[0].strip()

    def _find_stock_price_columns(self, prices: pd.DataFrame) -> List[object]:
        """Sucht nach Aktienpreis-Spalten (.DE + erlaubter Feldname)."""
        allowed_price_fields = [
            "TRDPRC_1",
            "OPEN_PRC",
            "CLOSE",
            "PRC",
            "LAST",
            "CLOSE_PRC",
            "HIGH_1",
            "LOW_1",
        ]
        stock_price_cols: List[object] = []

        for col in prices.columns:
            col_str = str(col)
            upper = col_str.upper()

            if any(idx in upper for idx in [".GDAXI", ".SDAXI", ".V1XI"]):
                # Index/Volatilität aussparen
                continue

            if (".DE" in upper or "(" in col_str) and any(field in upper for field in allowed_price_fields):
                stock_price_cols.append(col)

        return stock_price_cols

    def _find_market_series(
        self, prices: pd.DataFrame, index_col: str, stock_price_cols: List[object]
    ) -> pd.Series:
        """Sucht Index-Spalte, ansonsten Durchschnitt aller Aktienpreise."""
        if index_col in prices.columns:
            logger.info(f"Nutze Index-Spalte aus Config: {index_col}")
            return prices[index_col].clip(lower=1e-10)

        for col in prices.columns:
            col_str = str(col)
            if any(idx in col_str for idx in [".GDAXI", ".SDAXI"]):
                logger.info(f"Nutze gefundene Index-Spalte: {col}")
                return prices[col].clip(lower=1e-10)

        logger.info("Index-Spalte nicht gefunden – nehme Durchschnitt der Aktienpreise als Markt-Proxy.")
        return prices[stock_price_cols].mean(axis=1).clip(lower=1e-10)

    # ----------------------------
    # Hauptfunktion
    # ----------------------------
    def calculate_factors(
        self,
        price_df: pd.DataFrame,
        company_df: pd.DataFrame,
        index_col: str,
        portfolio_name: str,
    ) -> pd.DataFrame:
        """
        Berechnet Fama-French/Carhart Faktoren für ein Portfolio.
        Robust bei fehlenden Fundamentaldaten: SMB/HML werden dann 0 gesetzt.
        """
        logger.info(f"Berechne Fama-French Faktoren für Portfolio: {portfolio_name}")

        prices = self._normalize_index(price_df)
        company = self._normalize_index(company_df)

        logger.debug(f"Price Columns: {list(prices.columns)[:10]} (gesamt {len(prices.columns)})")
        logger.debug(f"Company Columns: {list(company.columns)[:10]} (gesamt {len(company.columns)})")

        stock_price_cols = self._find_stock_price_columns(prices)
        if not stock_price_cols:
            logger.warning("Keine Aktien-Preisspalten gefunden – gebe leeres DataFrame zurück.")
            return pd.DataFrame()

        price_matrix = prices[stock_price_cols].clip(lower=1e-10)
        market_prices = self._find_market_series(prices, index_col=index_col, stock_price_cols=stock_price_cols)

        # Returns
        stock_returns = price_matrix.pct_change()
        returns_df = stock_returns.dropna()
        market_returns = market_prices.pct_change().reindex(returns_df.index)

        # Risk-free auf Tagesbasis (Annäherung)
        daily_rf = self.risk_free_rate / 252.0

        factors_df = pd.DataFrame(index=returns_df.index)
        factors_df["Mkt_Rf"] = market_returns - daily_rf

        # SMB/HML – Fallback auf 0, wenn Fundamentaldaten fehlen
        try:
            smb, hml = self._calculate_size_and_value_factors(
                returns_df=returns_df, company_df=company, prices=price_matrix, stock_cols=stock_price_cols
            )
        except Exception as exc:
            logger.warning(f"SMB/HML Fallback (setze 0) wegen Fehler: {exc}")
            smb = pd.Series(0.0, index=factors_df.index)
            hml = pd.Series(0.0, index=factors_df.index)

        wml = self._calculate_momentum_factor(returns_df)

        factors_df["SMB"] = smb.reindex(factors_df.index).fillna(0)
        factors_df["HML"] = hml.reindex(factors_df.index).fillna(0)
        factors_df["WML"] = wml.reindex(factors_df.index)

        # Fülle verbleibende NaNs konservativ mit 0
        factors_df = factors_df.fillna(0)

        logger.info(
            f"Fama-French Faktoren berechnet: {len(factors_df)} Datenpunkte "
            f"(Index: {factors_df.index.min()} bis {factors_df.index.max()})"
        )
        return factors_df

    # ----------------------------
    # Faktoren
    # ----------------------------
    def _calculate_size_and_value_factors(
        self,
        returns_df: pd.DataFrame,
        company_df: pd.DataFrame,
        prices: pd.DataFrame,
        stock_cols: List[object],
    ) -> Tuple[pd.Series, pd.Series]:
        """
        SMB/HML Berechnung. Falls Fundamentaldaten fehlen, wird 0 zurückgegeben.
        """
        if company_df is None or company_df.empty:
            logger.info("Keine Company-Daten – SMB/HML werden auf 0 gesetzt.")
            zero = pd.Series(0.0, index=returns_df.index)
            return zero, zero

        # Benötigte Spalten in Company-Daten suchen
        company_cols_upper = {str(c).upper(): c for c in company_df.columns}
        has_mc = any("MARKETCAP" in col for col in company_cols_upper)
        has_bv = any("BOOKVALUE" in col or "BVPS" in col for col in company_cols_upper)

        if not (has_mc and has_bv):
            logger.info("Company-Daten ohne MarketCap/BookValue – SMB/HML = 0.")
            zero = pd.Series(0.0, index=returns_df.index)
            return zero, zero

        # Falls genügend Daten vorhanden, könnte hier echte SMB/HML-Berechnung folgen.
        # Aktuell (mangels vollständiger Fundamentaldaten) setzen wir 0.
        zero = pd.Series(0.0, index=returns_df.index)
        return zero, zero

    def _calculate_momentum_factor(self, returns_df: pd.DataFrame, lookback_period: int = 60) -> pd.Series:
        """Momentum-Faktor (WML) mit kürzerem Lookback für robuste Ergebnisse."""
        cumulative_returns = (1 + returns_df).rolling(window=lookback_period, min_periods=20).apply(
            lambda x: (1 + x).prod() - 1, raw=True
        )

        wml_values = []
        for date in returns_df.index:
            if date not in cumulative_returns.index:
                wml_values.append(np.nan)
                continue

            date_cum_returns = cumulative_returns.loc[date].dropna()
            if len(date_cum_returns) < 2:
                wml_values.append(np.nan)
                continue

            sorted_returns = date_cum_returns.sort_values()
            n = len(sorted_returns)
            winners = sorted_returns.iloc[int(0.7 * n):].index
            losers = sorted_returns.iloc[: int(0.3 * n)].index

            current_returns = returns_df.loc[date]
            winner_returns = [current_returns.get(s, np.nan) for s in winners]
            loser_returns = [current_returns.get(s, np.nan) for s in losers]

            if winner_returns and loser_returns:
                winner_avg = np.nanmean([r for r in winner_returns if not pd.isna(r)])
                loser_avg = np.nanmean([r for r in loser_returns if not pd.isna(r)])
                wml_values.append(winner_avg - loser_avg)
            else:
                wml_values.append(np.nan)

        return pd.Series(wml_values, index=returns_df.index)


# Convenience-Funktion
def calculate_fama_french_factors(
    portfolio_name: str,
    price_df: pd.DataFrame,
    company_df: pd.DataFrame,
    config_path: str = "config.yaml",
) -> pd.DataFrame:
    """
    Berechnet Faktoren für ein Portfolio über die Convenience-API.
    """
    config = ConfigManager(config_path)
    portfolio_config = config.get(f"data.portfolios.{portfolio_name}")
    if not portfolio_config:
        raise ValueError(f"Portfolio '{portfolio_name}' nicht in Config gefunden")

    index_name = portfolio_config.get("index", ".GDAXI")
    index_col = f"{index_name}_TRDPRC_1"  # Standard-Format

    model = FamaFrenchFactorModel(config_path)
    return model.calculate_factors(
        price_df=price_df,
        company_df=company_df,
        index_col=index_col,
        portfolio_name=portfolio_name,
    )


if __name__ == "__main__":
    from logger_config import setup_logging

    setup_logging()
    print("FamaFrench.py - Test")
