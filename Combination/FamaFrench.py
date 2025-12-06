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


class FamaFrenchFactorModel:
    """Calculate Fama-French/Carhart factors with robust fallbacks."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.risk_free_rate = self.config.get("features.risk_free_rate", 0.027)

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
        Compute simplified SMB (Small minus Big) and HML (High minus Low)
        based on market capitalization and book-to-market ratios.

        Falls Fundamentaldaten nicht vorhanden oder unbrauchbar sind,
        wird für beide Faktoren ein Nullvektor zurückgegeben.
        """
        # Fallback: keine Fundamentaldaten
        if company_df is None or company_df.empty:
            zero = pd.Series(0.0, index=returns_df.index)
            return zero, zero

        df = company_df.copy()

        # Datumsspalte herstellen
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            date_col = df.columns[0]
        elif "Date" in df.columns:
            date_col = "Date"
        else:
            zero = pd.Series(0.0, index=returns_df.index)
            return zero, zero

        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])

        # Instrumentspalte finden
        instrument_col = None
        for c in df.columns:
            cname = str(c).strip().upper()
            if cname in ("INSTRUMENT", "RIC", "TICKER"):
                instrument_col = c
                break

        if instrument_col is None:
            zero = pd.Series(0.0, index=returns_df.index)
            return zero, zero

        # Markt-Kapitalisierung und Buchwertspalten robust suchen
        company_cols_upper = {str(c).upper(): c for c in df.columns}

        mc_col = None
        for up, orig in company_cols_upper.items():
            if ("MARKET" in up and "CAP" in up) or "MARKETCAP" in up:
                mc_col = orig
                break

        bv_col = None
        for up, orig in company_cols_upper.items():
            if ("BOOK" in up and "VALUE" in up) or "BVPS" in up:
                bv_col = orig
                break

        if mc_col is None or bv_col is None:
            zero = pd.Series(0.0, index=returns_df.index)
            return zero, zero

        df = df[["Date", instrument_col, mc_col, bv_col]].copy()
        df = df.rename(columns={instrument_col: "Instrument", mc_col: "MC", bv_col: "BV"})

        df["Instrument"] = df["Instrument"].astype(str).str.strip()
        df["MC"] = pd.to_numeric(df["MC"], errors="coerce")
        df["BV"] = pd.to_numeric(df["BV"], errors="coerce")
        df = df.dropna(subset=["Instrument", "MC", "BV"])

        if df.empty:
            zero = pd.Series(0.0, index=returns_df.index)
            return zero, zero

        # Book-to-Market (BTM) berechnen
        df["BTM"] = df["BV"] / df["MC"]
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["BTM"])

        if df.empty:
            zero = pd.Series(0.0, index=returns_df.index)
            return zero, zero

        # MultiIndex (Date, Instrument) für schnellen Zugriff
        df = df.set_index(["Date", "Instrument"]).sort_index()

        # Mapping von Preisspalten zu Instrumenten
        col_to_inst = {}
        for col in stock_cols:
            inst = self._extract_stock_name(col)
            if inst:
                col_to_inst[col] = inst

        smb_values = []
        hml_values = []

        for date in returns_df.index:
            date_norm = pd.Timestamp(date).normalize()
            cross_ret = returns_df.loc[date]

            rows = []
            for col, inst in col_to_inst.items():
                r = cross_ret.get(col, np.nan)
                if pd.isna(r):
                    continue

                key = (date_norm, inst)
                try:
                    if key in df.index:
                        row = df.loc[key]
                        if isinstance(row, pd.DataFrame):
                            row = row.iloc[-1]
                    else:
                        # Fallback: letzter bekannter Wert <= Datum
                        inst_data = df.xs(inst, level="Instrument", drop_level=False)
                        inst_dates = inst_data.index.get_level_values("Date")
                        mask = inst_dates <= date_norm
                        if not mask.any():
                            continue
                        row = inst_data[mask].iloc[-1]
                except Exception:
                    continue

                mc = row["MC"]
                btm = row["BTM"]
                if pd.isna(mc) or pd.isna(btm):
                    continue

                rows.append({"Instrument": inst, "MC": mc, "BTM": btm, "Ret": r})

            if len(rows) < 2:
                smb_values.append(np.nan)
                hml_values.append(np.nan)
                continue

            cs = pd.DataFrame(rows)

            mc_median = cs["MC"].median()
            btm_median = cs["BTM"].median()

            small = cs[cs["MC"] <= mc_median]
            big = cs[cs["MC"] > mc_median]

            high = cs[cs["BTM"] >= btm_median]
            low = cs[cs["BTM"] < btm_median]

            if len(small) == 0 or len(big) == 0:
                smb_values.append(np.nan)
            else:
                smb_values.append(small["Ret"].mean() - big["Ret"].mean())

            if len(high) == 0 or len(low) == 0:
                hml_values.append(np.nan)
            else:
                hml_values.append(high["Ret"].mean() - low["Ret"].mean())

        smb_series = pd.Series(smb_values, index=returns_df.index)
        hml_series = pd.Series(hml_values, index=returns_df.index)
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
