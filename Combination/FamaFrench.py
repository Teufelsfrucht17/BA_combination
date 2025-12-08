"""
FamaFrench.py - Compute robust Fama-French/Carhart factors.

If fundamental data or index prices are missing, fallbacks are used:
- Market proxy = average of stock prices
- SMB/HML = 0
- Momentum stays active as long as price series exist
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

from ConfigManager import ConfigManager
from logger_config import get_logger


class FamaFrenchFactorModel:
    """Calculate Fama-French/Carhart factors with robust fallbacks."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.risk_free_rate = self.config.get("features.risk_free_rate", 0.027)
        self.logger = get_logger(__name__)

    @staticmethod
    def _normalize_index(df: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
        """Ensure a normalized DatetimeIndex exists."""
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
        """Extract ticker symbol from a column (tuple or string)."""
        if isinstance(col, tuple) and len(col) > 0:
            return str(col[0]).strip()
        col_str = str(col)
        if ".DE" in col_str:
            end = col_str.find(".DE") + len(".DE")
            start = end - 1
            while start > 0 and col_str[start - 1] not in " ,('\"":
                start -= 1
            return col_str[start:end].strip()
        return col_str.split("_")[0].strip()

    def _find_stock_price_columns(self, prices: pd.DataFrame) -> List[object]:
        """Find stock price columns (.DE plus allowed field name)."""
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
                continue

            if (".DE" in upper or "(" in col_str) and any(field in upper for field in allowed_price_fields):
                stock_price_cols.append(col)

        return stock_price_cols

    def _find_market_series(
        self, prices: pd.DataFrame, index_col: str, stock_price_cols: List[object]
    ) -> pd.Series:
        """Sucht Index-Spalte, ansonsten Durchschnitt aller Aktienpreise."""
        if index_col in prices.columns:
            return prices[index_col].clip(lower=1e-10)

        for col in prices.columns:
            col_str = str(col)
            if any(idx in col_str for idx in [".GDAXI", ".SDAXI"]):
                return prices[col].clip(lower=1e-10)

        return prices[stock_price_cols].mean(axis=1).clip(lower=1e-10)

    def calculate_factors(
        self,
        price_df: pd.DataFrame,
        company_df: pd.DataFrame,
        index_col: str,
        portfolio_name: str,
    ) -> pd.DataFrame:
        """
        Calculate Fama-French/Carhart factors for a portfolio.
        Robust when fundamentals are missing: SMB/HML are set to 0.
        """
        prices = self._normalize_index(price_df)
        company = self._normalize_index(company_df)

        stock_price_cols = self._find_stock_price_columns(prices)
        if not stock_price_cols:
            return pd.DataFrame()

        price_matrix = prices[stock_price_cols].clip(lower=1e-10)
        market_prices = self._find_market_series(prices, index_col=index_col, stock_price_cols=stock_price_cols)

        stock_returns = price_matrix.pct_change()
        returns_df = stock_returns.dropna()
        market_returns = market_prices.pct_change().reindex(returns_df.index)

        daily_rf = self.risk_free_rate / 252.0

        factors_df = pd.DataFrame(index=returns_df.index)
        factors_df["Mkt_Rf"] = market_returns - daily_rf

        smb, hml = self._calculate_size_and_value_factors(
            returns_df=returns_df, company_df=company, prices=price_matrix, stock_cols=stock_price_cols
        )

        wml = self._calculate_momentum_factor(returns_df)

        factors_df["SMB"] = smb.reindex(factors_df.index).fillna(0)
        factors_df["HML"] = hml.reindex(factors_df.index).fillna(0)
        factors_df["WML"] = wml.reindex(factors_df.index)

        factors_df = factors_df.fillna(0)

        return factors_df

    def _calculate_size_and_value_factors(
        self,
        returns_df: pd.DataFrame,
        company_df: pd.DataFrame,
        prices: pd.DataFrame,
        stock_cols: List[object],
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute SMB (Small Minus Big) and HML (High Minus Low) factors.

        Uses fundamental data per instrument and date:
        - Size proxy: company market capitalization
        - Value proxy: book-to-market ratio (book value per share * shares / market cap)

        Falls back to 0 series if required fields are missing.
        """
        if company_df is None or company_df.empty:
            zero = pd.Series(0.0, index=returns_df.index)
            return zero, zero

        cols_lower = {str(c).lower(): c for c in company_df.columns}

        instrument_col = None
        for key, col in cols_lower.items():
            if "instrument" in key:
                instrument_col = col
                break

        mc_col = None
        for key, col in cols_lower.items():
            if "market" in key and "capital" in key:
                mc_col = col
                break

        bvps_col = None
        # Prefer „Book Value Per Share - Actual“
        for key, col in cols_lower.items():
            if "book value per share - actual" in key or ("bvps" in key and "actual" in key):
                bvps_col = col
                break
        if bvps_col is None:
            for key, col in cols_lower.items():
                if "book value per share" in key or "bvps" in key:
                    bvps_col = col
                    break

        shares_col = None
        for key, col in cols_lower.items():
            if "number of shares outstanding" in key:
                shares_col = col
                break
        if shares_col is None:
            for key, col in cols_lower.items():
                if "outstanding shares" in key:
                    shares_col = col
                    break

        if instrument_col is None or mc_col is None:
            self.logger.warning(
                "Cannot compute SMB/HML – missing instrument or market cap column "
                "(instrument_col=%s, mc_col=%s)",
                instrument_col,
                mc_col,
            )
            zero = pd.Series(0.0, index=returns_df.index)
            return zero, zero

        # Map ticker -> price/return column
        ticker_to_col = {}
        for col in stock_cols:
            ticker = self._extract_stock_name(col)
            if ticker and ticker not in ticker_to_col:
                ticker_to_col[ticker] = col

        smb_values = []
        hml_values = []
        index_vals = []

        for date in returns_df.index:
            if date not in company_df.index:
                index_vals.append(date)
                smb_values.append(np.nan)
                hml_values.append(np.nan)
                continue

            sub = company_df.loc[date]
            if isinstance(sub, pd.Series):
                sub = sub.to_frame().T

            rows = []
            for _, row in sub.iterrows():
                inst = row.get(instrument_col)
                if pd.isna(inst):
                    continue

                ticker = str(inst).strip()
                price_col = ticker_to_col.get(ticker)
                if price_col is None or price_col not in returns_df.columns:
                    continue

                ret = returns_df.at[date, price_col]
                if pd.isna(ret):
                    continue

                mc = row.get(mc_col)
                if pd.isna(mc) or mc <= 0:
                    continue

                entry = {"ticker": ticker, "ret": float(ret), "mc": float(mc)}

                if bvps_col is not None and shares_col is not None:
                    bvps = row.get(bvps_col)
                    shares = row.get(shares_col)
                    if pd.notna(bvps) and pd.notna(shares) and shares > 0:
                        try:
                            bm = float(bvps) * float(shares) / float(mc)
                        except ZeroDivisionError:
                            bm = np.nan
                    else:
                        bm = np.nan
                    entry["bm"] = bm

                rows.append(entry)

            index_vals.append(date)

            if len(rows) < 2:
                smb_values.append(np.nan)
                hml_values.append(np.nan)
                continue

            cross = pd.DataFrame(rows)

            # SMB: Small minus Big by median market cap
            median_mc = cross["mc"].median()
            small = cross[cross["mc"] <= median_mc]
            big = cross[cross["mc"] > median_mc]
            if len(small) == 0 or len(big) == 0:
                smb_t = np.nan
            else:
                smb_t = small["ret"].mean() - big["ret"].mean()

            # HML: High minus Low by book-to-market
            if "bm" in cross.columns:
                valid_bm = cross.dropna(subset=["bm"])
                if len(valid_bm) >= 2:
                    q30 = valid_bm["bm"].quantile(0.3)
                    q70 = valid_bm["bm"].quantile(0.7)
                    value = valid_bm[valid_bm["bm"] >= q70]
                    growth = valid_bm[valid_bm["bm"] <= q30]
                    if len(value) > 0 and len(growth) > 0:
                        hml_t = value["ret"].mean() - growth["ret"].mean()
                    else:
                        hml_t = np.nan
                else:
                    hml_t = np.nan
            else:
                hml_t = np.nan

            smb_values.append(smb_t)
            hml_values.append(hml_t)

        smb_series = pd.Series(smb_values, index=index_vals)
        hml_series = pd.Series(hml_values, index=index_vals)

        return smb_series, hml_series

    def _calculate_momentum_factor(self, returns_df: pd.DataFrame, lookback_period: int = 60) -> pd.Series:
        """Momentum factor (WML) with shorter lookback for robustness."""
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


def calculate_fama_french_factors(
    portfolio_name: str,
    price_df: pd.DataFrame,
    company_df: pd.DataFrame,
    config_path: str = "config.yaml",
) -> pd.DataFrame:
    """
    Calculate factors for a portfolio via the convenience API.
    """
    config = ConfigManager(config_path)
    portfolio_config = config.get(f"data.portfolios.{portfolio_name}")

    index_name = (portfolio_config.get("index") if portfolio_config else None) or ".GDAXI"
    index_col = f"{index_name}_TRDPRC_1"

    model = FamaFrenchFactorModel(config_path)
    return model.calculate_factors(
        price_df=price_df,
        company_df=company_df,
        index_col=index_col,
        portfolio_name=portfolio_name,
    )


if __name__ == "__main__":
    model = FamaFrenchFactorModel()
