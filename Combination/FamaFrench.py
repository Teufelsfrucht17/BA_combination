"""
FamaFrench.py - Berechnet Fama-French/Carhart Faktoren

Implementiert:
- 3-Factor Model (Fama & French, 1993): Mkt-Rf, SMB, HML
- 4-Factor Model (Carhart, 1997): + Momentum (WML)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import datetime

from ConfigManager import ConfigManager
from logger_config import get_logger

logger = get_logger(__name__)


class FamaFrenchFactorModel:
    """
    Berechnet Fama-French/Carhart Faktoren aus fundamentalen und Preisdaten
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialisiert Fama-French Factor Model

        Args:
            config_path: Pfad zur Config-Datei
        """
        self.config = ConfigManager(config_path)
        self.risk_free_rate = self.config.get("features.risk_free_rate", 0.027)  # 2.7% default
        
        logger.info(f"Fama-French Model initialisiert (Risk-free Rate: {self.risk_free_rate*100:.2f}%)")

    def calculate_factors(
        self,
        price_df: pd.DataFrame,
        company_df: pd.DataFrame,
        index_col: str,
        portfolio_name: str
    ) -> pd.DataFrame:
        """
        Berechnet Fama-French/Carhart Faktoren für ein Portfolio

        Args:
            price_df: DataFrame mit Preisdaten (Date im Index, Spalten: STOCK.DE_TRDPRC_1)
            company_df: DataFrame mit fundamentalen Daten (Date, TR.CompanyMarketCapitalization, etc.)
            index_col: Name der Index-Spalte (z.B. ".GDAXI_TRDPRC_1")
            portfolio_name: Name des Portfolios (z.B. "dax", "sdax")

        Returns:
            DataFrame mit Faktoren: Date, Mkt_Rf, SMB, HML, WML (Momentum)
        """
        logger.info(f"Berechne Fama-French Faktoren für Portfolio: {portfolio_name}")

        # Kopiere und bereite Daten vor
        prices = price_df.copy()
        company = company_df.copy()

        # Setze Date als Index falls vorhanden
        if "Date" in prices.columns:
            prices["Date"] = pd.to_datetime(prices["Date"], errors='coerce')
            prices = prices.set_index("Date")
        # Validiere und normalisiere Company-Daten
        logger.debug(f"Company-Daten vor Konvertierung: Index-Type={type(company.index)}, Columns={list(company.columns)[:5]}")
        
        if "Date" in company.columns:
            # Konvertiere Date-Spalte zu DatetimeIndex
            company["Date"] = pd.to_datetime(company["Date"], errors='coerce')
            # Entferne Zeitzone-Information falls vorhanden
            if hasattr(company["Date"].dtype, 'tz') or (hasattr(company["Date"], 'dt') and company["Date"].dt.tz is not None):
                company["Date"] = company["Date"].dt.tz_localize(None)
            # Normalisiere auf Tagesbasis (entferne Zeit-Information)
            if hasattr(company["Date"], 'dt'):
                company["Date"] = company["Date"].dt.normalize()
            company = company.set_index("Date")
            logger.debug(f"Company-Daten: Date-Spalte zu Index konvertiert, Index-Type={type(company.index)}")
        elif not isinstance(company.index, pd.DatetimeIndex):
            # Versuche Index zu konvertieren
            company.index = pd.to_datetime(company.index, errors='coerce')
            # Entferne Zeitzone-Information falls vorhanden
            if hasattr(company.index, 'tz') and company.index.tz is not None:
                company.index = company.index.tz_localize(None)
            # Normalisiere auf Tagesbasis
            if isinstance(company.index, pd.DatetimeIndex):
                company.index = company.index.normalize()

        # Validiere und normalisiere Price-Daten Index
        logger.debug(f"Price-Daten vor Konvertierung: Index-Type={type(prices.index)}, Index-Range={prices.index.min() if len(prices) > 0 else 'N/A'} bis {prices.index.max() if len(prices) > 0 else 'N/A'}")
        
        if not isinstance(prices.index, pd.DatetimeIndex):
            # Versuche Index zu konvertieren
            prices.index = pd.to_datetime(prices.index, errors='coerce')
        
        # Normalisiere Index auf Tagesbasis (für Daily-Daten)
        if isinstance(prices.index, pd.DatetimeIndex):
            # Entferne Zeitzone-Information falls vorhanden
            if prices.index.tz is not None:
                prices.index = prices.index.tz_localize(None)
            # Normalisiere auf Tagesbasis (entferne Zeit-Information)
            prices.index = prices.index.normalize()
        
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError(f"Price DataFrame muss DatetimeIndex haben, aber hat {type(prices.index)}")
        if not isinstance(company.index, pd.DatetimeIndex):
            raise ValueError(f"Company DataFrame muss DatetimeIndex haben, aber hat {type(company.index)}")
        
        # Normalisiere Company-Index auf Tagesbasis
        if isinstance(company.index, pd.DatetimeIndex):
            # Entferne Zeitzone-Information falls vorhanden
            if company.index.tz is not None:
                company.index = company.index.tz_localize(None)
            # Normalisiere auf Tagesbasis
            company.index = company.index.normalize()
        
        logger.debug(f"Price-Daten nach Konvertierung: Index-Type={type(prices.index)}, Index-Range={prices.index.min()} bis {prices.index.max()}, Anzahl={len(prices.index)}")
        logger.debug(f"Company-Daten nach Konvertierung: Index-Type={type(company.index)}, Index-Range={company.index.min()} bis {company.index.max()}, Anzahl={len(company.index)}")

        # Debug: Zeige verfügbare Spalten
        logger.debug(f"Verfügbare Spalten im Price DataFrame: {list(prices.columns)[:20]}...")
        logger.debug(f"Anzahl Spalten: {len(prices.columns)}")
        
        # Hole Stock-Spalten (Preise) - flexibler: unterstütze auch MultiIndex
        # WICHTIG: Durchsuche ALLE Spalten, nicht nur die ersten!
        stock_price_cols = []
        all_trdprc_cols = []  # Debug: Sammle alle TRDPRC_1 Spalten
        
        for col in prices.columns:
            col_str = str(col)
            is_tuple = isinstance(col, tuple)
            
            # Prüfe ob es ein String ist, der wie ein MultiIndex aussieht: "('RHMG.DE', 'TRDPRC_1')"
            is_string_tuple = False
            parsed = None
            if not is_tuple and col_str.startswith("('") and "', '" in col_str:
                # Versuche String-Tuple zu parsen
                try:
                    import ast
                    parsed = ast.literal_eval(col_str)
                    if isinstance(parsed, tuple) and len(parsed) >= 2:
                        is_string_tuple = True
                except:
                    pass
            
            if is_tuple or is_string_tuple:
                # MultiIndex: z.B. ('RHMG.DE', 'TRDPRC_1') oder ('RHMG.DE', 'OPEN_PRC')
                if is_string_tuple:
                    first_level = str(parsed[0])
                    second_level = str(parsed[1])
                else:
                    if len(col) >= 2:
                        first_level = str(col[0])
                        second_level = str(col[1])
                    else:
                        continue
                
                # Debug: Sammle alle TRDPRC_1 Spalten
                if 'TRDPRC_1' in second_level:
                    all_trdprc_cols.append(col)
                # Prüfe ob es eine Stock-Preis-Spalte ist
                # Erste Ebene sollte .DE enthalten (Aktie), zweite Ebene sollte TRDPRC_1 sein
                if '.DE' in first_level and 'TRDPRC_1' in second_level:
                    # Prüfe dass es nicht ein Index ist
                    if not any(idx in first_level for idx in ['.GDAXI', '.SDAXI', '.V1XI']):
                        stock_price_cols.append(col)
            else:
                # Einfacher String: z.B. "RHMG.DE_TRDPRC_1" oder "TRDPRC_1"
                # Debug: Sammle alle TRDPRC_1 Spalten
                if 'TRDPRC_1' in col_str:
                    all_trdprc_cols.append(col)
                # Suche nach .DE (Aktien) und TRDPRC_1 (Preis)
                if '.DE' in col_str and 'TRDPRC_1' in col_str:
                    # Prüfe dass es nicht ein Index ist
                    if not any(idx in col_str for idx in ['.GDAXI', '.SDAXI', '.V1XI']):
                        stock_price_cols.append(col)
        
        logger.debug(f"Gefundene TRDPRC_1 Spalten (alle): {all_trdprc_cols[:10]}... (insgesamt {len(all_trdprc_cols)})")
        logger.debug(f"Gefundene Stock-Preis-Spalten (nach erster Suche): {stock_price_cols[:5]}... (insgesamt {len(stock_price_cols)})")
        
        # Fallback: Bei Daily-Daten kann TRDPRC_1 als einfache Spalte vorliegen
        # In diesem Fall müssen wir die Stock-Namen aus den MultiIndex-Spalten extrahieren
        if not stock_price_cols:
            # Suche nach einfacher 'TRDPRC_1' Spalte
            simple_trdprc = None
            for col in prices.columns:
                if not isinstance(col, tuple) and str(col) == 'TRDPRC_1':
                    simple_trdprc = col
                    break
            
            if simple_trdprc is not None:
                # Finde alle Stock-Namen aus MultiIndex-Spalten
                stock_names = set()
                for col in prices.columns:
                    if isinstance(col, tuple) and len(col) >= 1:
                        first_level = str(col[0])
                        if '.DE' in first_level and not any(idx in first_level for idx in ['.GDAXI', '.SDAXI', '.V1XI']):
                            stock_names.add(first_level)
                
                # Erstelle Stock-Preis-Spalten: Verwende die einfache TRDPRC_1 Spalte für alle Stocks
                # Aber das funktioniert nicht, da wir eine Spalte pro Stock brauchen
                # Stattdessen: Suche nach MultiIndex-Spalten mit Stock-Namen und erstelle künstliche Spalten
                logger.warning("TRDPRC_1 als einfache Spalte gefunden, aber keine Stock-spezifischen Spalten. Versuche Alternative...")
                # Versuche andere Felder zu verwenden (z.B. OPEN_PRC als Proxy)
                # ABER: Prüfe zuerst, ob TRDPRC_1 als MultiIndex vorhanden ist (auch String-Tuples)
                for col in prices.columns:
                    col_str = str(col)
                    is_tuple = isinstance(col, tuple)
                    
                    # Prüfe String-Tuple
                    is_string_tuple = False
                    parsed = None
                    if not is_tuple and col_str.startswith("('") and "', '" in col_str:
                        try:
                            import ast
                            parsed = ast.literal_eval(col_str)
                            if isinstance(parsed, tuple) and len(parsed) >= 2:
                                is_string_tuple = True
                        except:
                            pass
                    
                    if is_tuple or is_string_tuple:
                        if is_string_tuple:
                            first_level = str(parsed[0])
                            second_level = str(parsed[1])
                        else:
                            if len(col) >= 2:
                                first_level = str(col[0])
                                second_level = str(col[1])
                            else:
                                continue
                        
                        # Prüfe zuerst auf TRDPRC_1
                        if '.DE' in first_level and 'TRDPRC_1' in second_level:
                            if not any(idx in first_level for idx in ['.GDAXI', '.SDAXI', '.V1XI']):
                                stock_price_cols.append(col)
                                logger.debug(f"Gefunden: {col} (TRDPRC_1)")
                        # Fallback: OPEN_PRC als Proxy
                        elif '.DE' in first_level and 'OPEN_PRC' in second_level:
                            if not any(idx in first_level for idx in ['.GDAXI', '.SDAXI', '.V1XI']):
                                stock_price_cols.append(col)
                                logger.debug(f"Verwende {col} als Proxy für Stock-Preis (OPEN_PRC)")
        
        # Fallback 2: Suche nach TRDPRC_1 in verschiedenen Formaten
        if not stock_price_cols:
            for col in prices.columns:
                col_str = str(col)
                if isinstance(col, tuple):
                    col_str = ' '.join(str(c) for c in col)
                # Prüfe ob TRDPRC_1 enthalten ist, aber nicht Index
                if 'TRDPRC_1' in col_str and not any(idx in col_str for idx in ['.GDAXI', '.SDAXI', '.V1XI']):
                    # Prüfe ob es eine Aktie ist (enthält bekannte Aktien-Namen oder .DE)
                    if any(x in col_str for x in ['RHMG', 'ENR1n', 'TKAG', 'FTKn', 'ACT1', 'DEZG', 'AOFG', 'CWCG', 'ADNGk', '1U1', 'DMPG', 'COKG', '.DE']):
                        stock_price_cols.append(col)
        
        if not stock_price_cols:
            logger.error(f"Verfügbare Spalten (alle): {list(prices.columns)}")
            raise ValueError(f"Keine Stock-Preis-Spalten gefunden. Verfügbare Spalten: {list(prices.columns)[:20]}...")

        logger.debug(f"Gefundene Stock-Preis-Spalten: {stock_price_cols[:5]}... (insgesamt {len(stock_price_cols)})")

        # Hole Index-Return - flexibler: suche auch nach Varianten
        # Erwartetes Format: index_col = ".GDAXI_TRDPRC_1" oder ".SDAXI_TRDPRC_1"
        index_name = index_col.split('_')[0]  # z.B. ".GDAXI"
        index_found = False
        actual_index_col = None
        
        # Prüfe ob index_col direkt vorhanden ist
        if index_col in prices.columns:
            actual_index_col = index_col
            index_found = True
            logger.debug(f"Index-Spalte direkt gefunden: {index_col}")
        else:
            # Suche nach MultiIndex-Format: ('.GDAXI', 'TRDPRC_1') oder String-Tuple
            for col in prices.columns:
                col_str = str(col)
                is_tuple = isinstance(col, tuple)
                
                # Prüfe ob es ein String ist, der wie ein MultiIndex aussieht
                is_string_tuple = False
                parsed = None
                if not is_tuple and col_str.startswith("('") and "', '" in col_str:
                    try:
                        import ast
                        parsed = ast.literal_eval(col_str)
                        if isinstance(parsed, tuple) and len(parsed) >= 2:
                            is_string_tuple = True
                    except:
                        pass
                
                if is_tuple or is_string_tuple:
                    # MultiIndex: Prüfe ob erster Teil der Index-Name ist
                    if is_string_tuple:
                        first_level = str(parsed[0])
                        second_level = str(parsed[1])
                    else:
                        if len(col) >= 2:
                            first_level = str(col[0])
                            second_level = str(col[1])
                        else:
                            continue
                    
                    if first_level == index_name and 'TRDPRC_1' in second_level:
                        actual_index_col = col
                        index_found = True
                        logger.debug(f"Index-Spalte als MultiIndex gefunden: {col}")
                        break
                else:
                    # Einfacher String: Prüfe ob Index-Name und TRDPRC_1 enthalten sind
                    if index_name in col_str and 'TRDPRC_1' in col_str:
                        actual_index_col = col
                        index_found = True
                        logger.debug(f"Index-Spalte als String-Variante gefunden: {col}")
                        break
            
            # Fallback: Falls nur eine einfache 'TRDPRC_1' Spalte existiert (wenn nur ein Index vorhanden)
            if not index_found:
                # Suche nach einfacher TRDPRC_1 Spalte (auch String-Tuples)
                simple_trdprc_cols = []
                for col in prices.columns:
                    col_str = str(col)
                    if col_str == 'TRDPRC_1':
                        simple_trdprc_cols.append(col)
                    elif isinstance(col, tuple) and len(col) == 1 and str(col[0]) == 'TRDPRC_1':
                        simple_trdprc_cols.append(col)
                    elif col_str.startswith("('") and "', '" in col_str:
                        try:
                            import ast
                            parsed = ast.literal_eval(col_str)
                            if isinstance(parsed, tuple) and len(parsed) == 1 and str(parsed[0]) == 'TRDPRC_1':
                                simple_trdprc_cols.append(col)
                        except:
                            pass
                
                if len(simple_trdprc_cols) == 1:
                    # Nur eine TRDPRC_1 Spalte - wahrscheinlich der Index
                    actual_index_col = simple_trdprc_cols[0]
                    index_found = True
                    logger.debug(f"Index-Spalte als einzige TRDPRC_1 Spalte gefunden: {actual_index_col}")
        
        if not index_found:
            logger.error(f"Verfügbare Spalten: {list(prices.columns)}")
            raise ValueError(f"Index-Spalte '{index_col}' (Index-Name: '{index_name}') nicht in Price DataFrame gefunden. Verfügbare Spalten: {list(prices.columns)[:10]}...")
        
        # Verwende die gefundene Spalte
        index_col = actual_index_col

        # Berechne Returns für alle Aktien
        stock_returns = {}
        for col in stock_price_cols:
            # Extrahiere Stock-Name aus Spaltenname (unterstütze verschiedene Formate)
            col_str = str(col)
            is_tuple = isinstance(col, tuple)
            
            # Prüfe ob es ein String-Tuple ist
            is_string_tuple = False
            parsed = None
            if not is_tuple and col_str.startswith("('") and "', '" in col_str:
                try:
                    import ast
                    parsed = ast.literal_eval(col_str)
                    if isinstance(parsed, tuple) and len(parsed) >= 2:
                        is_string_tuple = True
                        stock_name = str(parsed[0])
                except:
                    pass
            
            if is_tuple:
                # MultiIndex: Nimm ersten Teil
                stock_name = str(col[0])
            elif is_string_tuple:
                # String-Tuple: bereits oben extrahiert
                pass
            else:
                # Einfacher Spaltenname: Split bei '_' und nimm ersten Teil
                stock_name = col_str.split('_')[0]
            
            # Bereinige Stock-Name (entferne eventuelle Leerzeichen)
            stock_name = stock_name.strip()
            
            # Verwende die Spalte (kann Tuple oder String sein)
            try:
                stock_prices = prices[col].clip(lower=1e-10)  # Vermeide Division durch 0
            except (KeyError, TypeError):
                # Falls Spalte nicht gefunden, versuche als String
                logger.debug(f"Spalte {col} nicht direkt gefunden, versuche als String...")
                try:
                    stock_prices = prices[col_str].clip(lower=1e-10)
                except KeyError:
                    logger.error(f"Spalte {col} (als {col_str}) nicht gefunden!")
                    continue
            
            stock_returns[stock_name] = np.log(stock_prices / stock_prices.shift(1))
        
        logger.debug(f"Berechne Returns für {len(stock_returns)} Aktien: {list(stock_returns.keys())[:5]}...")

        returns_df = pd.DataFrame(stock_returns, index=prices.index)

        # Berechne Market Return (Index Return)
        try:
            market_prices = prices[index_col].clip(lower=1e-10)
        except (KeyError, TypeError):
            # Falls Spalte nicht gefunden, versuche als String
            logger.debug(f"Index-Spalte {index_col} nicht direkt gefunden, versuche als String...")
            market_prices = prices[str(index_col)].clip(lower=1e-10)
        market_returns = np.log(market_prices / market_prices.shift(1))

        # Berechne Market Return minus Risk-free Rate
        # Risk-free Rate auf täglicher Basis (annualisiert / 252 Trading Days)
        daily_rf = self.risk_free_rate / 252.0
        mkt_rf = market_returns - daily_rf

        # Berechne SMB (Small Minus Big) und HML (High Minus Low)
        smb, hml = self._calculate_size_and_value_factors(
            returns_df=returns_df,
            company_df=company,
            prices=prices,
            stock_cols=stock_price_cols
        )

        # Berechne Momentum (WML - Winners Minus Losers)
        wml = self._calculate_momentum_factor(returns_df)

        # Kombiniere alle Faktoren
        factors_df = pd.DataFrame({
            'Mkt_Rf': mkt_rf,
            'SMB': smb,
            'HML': hml,
            'WML': wml
        }, index=prices.index)

        # Debug: Zeige Anzahl NaN-Werte vor dropna
        nan_counts = factors_df.isna().sum()
        logger.debug(f"NaN-Werte vor dropna: Mkt_Rf={nan_counts['Mkt_Rf']}, SMB={nan_counts['SMB']}, HML={nan_counts['HML']}, WML={nan_counts['WML']}")
        logger.debug(f"Gesamt NaN: {factors_df.isna().sum().sum()} von {len(factors_df) * 4} Werten")
        
        # Entferne NaN-Werte
        factors_df = factors_df.dropna()
        
        if len(factors_df) == 0:
            logger.warning(f"Faktoren-DataFrame ist nach dropna() leer! Ursache: Zu viele NaN-Werte.")
            logger.warning(f"Ursprüngliche Länge: {len(prices)}, Mkt_Rf NaN: {mkt_rf.isna().sum()}, SMB NaN: {smb.isna().sum()}, HML NaN: {hml.isna().sum()}, WML NaN: {wml.isna().sum()}")

        logger.info(f"Fama-French Faktoren berechnet: {len(factors_df)} Datenpunkte")
        if len(factors_df) > 0:
            logger.debug(f"Durchschnittliche Faktoren:\n{factors_df.mean()}")

        return factors_df

    def _calculate_size_and_value_factors(
        self,
        returns_df: pd.DataFrame,
        company_df: pd.DataFrame,
        prices: pd.DataFrame,
        stock_cols: list
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Berechnet SMB (Small Minus Big) und HML (High Minus Low) Faktoren

        Portfolio Formation:
        - Size: Small (Bottom 50%) vs Big (Top 50%) nach Market Cap
        - Value: High (Top 30%), Medium (Middle 40%), Low (Bottom 30%) nach Book-to-Market

        Args:
            returns_df: DataFrame mit Stock Returns
            company_df: DataFrame mit fundamentalen Daten
            prices: DataFrame mit Preisen
            stock_cols: Liste der Stock-Preis-Spalten

        Returns:
            Tuple von (SMB, HML) Series
        """
        logger.debug("Berechne SMB und HML Faktoren...")

        # Erstelle gemeinsamen Index (täglich)
        # Normalisiere beide Indizes auf Tagesbasis
        returns_index = returns_df.index
        company_index = company_df.index
        
        if isinstance(returns_index, pd.DatetimeIndex):
            returns_index = returns_index.normalize()
        if isinstance(company_index, pd.DatetimeIndex):
            company_index = company_index.normalize()
        
        logger.debug(f"Returns Index: {returns_index.min()} bis {returns_index.max()}, Anzahl={len(returns_index)}")
        logger.debug(f"Company Index: {company_index.min()} bis {company_index.max()}, Anzahl={len(company_index)}")
        
        common_dates = returns_index.intersection(company_index)
        logger.debug(f"Gemeinsame Datenpunkte: {len(common_dates)} von {len(returns_index)} Returns und {len(company_index)} Company-Daten")
        
        if len(common_dates) == 0:
            logger.error(f"Keine gemeinsamen Datenpunkte! Returns: {returns_index[:5].tolist()}, Company: {company_index[:5].tolist()}")
            raise ValueError(f"Keine gemeinsamen Datenpunkte zwischen Returns ({len(returns_index)} Datenpunkte) und Company-Daten ({len(company_index)} Datenpunkte)")

        # Extrahiere Stock-Namen (unterstütze verschiedene Formate)
        stock_names = []
        for col in stock_cols:
            col_str = str(col)
            if isinstance(col, tuple):
                stock_name = str(col[0])
            else:
                stock_name = col_str.split('_')[0]
            stock_names.append(stock_name.strip())

        smb_values = []
        hml_values = []

        for date in common_dates:
            try:
                # Hole fundamentale Daten für aktuelles Datum
                # Forward fill falls keine Daten für exaktes Datum
                company_date_data = company_df.loc[company_df.index <= date].iloc[-1]

                # Berechne Book-to-Market für jede Aktie
                btms = {}
                market_caps = {}

                for stock_name in stock_names:
                    # Finde entsprechende Spalten
                    mc_col = None
                    bv_col = None
                    price_col = None

                    # Suche in Company-Daten (unterstütze verschiedene Formate)
                    for col in company_df.columns:
                        col_str = str(col).upper()
                        if isinstance(col, tuple):
                            col_str = ' '.join(str(c).upper() for c in col)
                        
                        # Prüfe ob diese Spalte zu stock_name gehört
                        stock_name_upper = stock_name.upper().replace('.DE', '')
                        if stock_name_upper in col_str or stock_name.upper() in col_str:
                            if 'COMPANYMARKETCAPITALIZATION' in col_str and 'DATE' not in col_str:
                                mc_col = col
                            if 'BOOKVALUEPERSHARE' in col_str or 'BVPS' in col_str:
                                bv_col = col

                    # Finde Preis-Spalte (flexibler)
                    for col in stock_cols:
                        col_str = str(col)
                        if isinstance(col, tuple):
                            col_str = ' '.join(str(c) for c in col)
                        if stock_name in col_str or stock_name.replace('.DE', '') in col_str:
                            price_col = col
                            break

                    if mc_col and bv_col and price_col:
                        # Extrahiere Werte
                        mc = company_date_data.get(mc_col, np.nan)
                        bv = company_date_data.get(bv_col, np.nan)
                        price = prices.loc[date, price_col] if date in prices.index else np.nan

                        if not pd.isna(mc) and not pd.isna(bv) and not pd.isna(price) and price > 0:
                            market_caps[stock_name] = mc
                            # Book-to-Market = Book Value per Share / Price
                            btms[stock_name] = bv / price

                if len(market_caps) < 3 or len(btms) < 3:
                    # Zu wenige Datenpunkte
                    smb_values.append(np.nan)
                    hml_values.append(np.nan)
                    continue

                # Sortiere nach Market Cap (für SMB)
                sorted_by_cap = sorted(market_caps.items(), key=lambda x: x[1])
                mid_idx = len(sorted_by_cap) // 2
                small_stocks = [s[0] for s in sorted_by_cap[:mid_idx]]
                big_stocks = [s[0] for s in sorted_by_cap[mid_idx:]]

                # Sortiere nach Book-to-Market (für HML)
                sorted_by_btm = sorted(btms.items(), key=lambda x: x[1])
                n = len(sorted_by_btm)
                high_btm_stocks = [s[0] for s in sorted_by_btm[int(0.7 * n):]]  # Top 30%
                medium_btm_stocks = [s[0] for s in sorted_by_btm[int(0.3 * n):int(0.7 * n)]]  # Middle 40%
                low_btm_stocks = [s[0] for s in sorted_by_btm[:int(0.3 * n)]]  # Bottom 30%

                # Berechne Portfolio Returns
                if date in returns_df.index:
                    date_returns = returns_df.loc[date]

                    # SMB = (Small + SmallHigh + SmallLow) / 3 - (Big + BigHigh + BigLow) / 3
                    # Vereinfacht: (Small - Big)
                    small_returns = [date_returns.get(s, np.nan) for s in small_stocks if s in date_returns.index]
                    big_returns = [date_returns.get(s, np.nan) for s in big_stocks if s in date_returns.index]

                    if small_returns and big_returns:
                        small_return = np.nanmean([r for r in small_returns if not pd.isna(r)])
                        big_return = np.nanmean([r for r in big_returns if not pd.isna(r)])
                        smb_values.append(small_return - big_return)
                    else:
                        smb_values.append(np.nan)

                    # HML = (SmallHigh + BigHigh) / 2 - (SmallLow + BigLow) / 2
                    high_btm_returns = [date_returns.get(s, np.nan) for s in high_btm_stocks if s in date_returns.index]
                    low_btm_returns = [date_returns.get(s, np.nan) for s in low_btm_stocks if s in date_returns.index]

                    if high_btm_returns and low_btm_returns:
                        high_return = np.nanmean([r for r in high_btm_returns if not pd.isna(r)])
                        low_return = np.nanmean([r for r in low_btm_returns if not pd.isna(r)])
                        hml_values.append(high_return - low_return)
                    else:
                        hml_values.append(np.nan)
                else:
                    smb_values.append(np.nan)
                    hml_values.append(np.nan)

            except Exception as e:
                logger.warning(f"Fehler bei Berechnung für {date}: {e}")
                smb_values.append(np.nan)
                hml_values.append(np.nan)

        # Erstelle Series
        smb_series = pd.Series(smb_values, index=common_dates)
        hml_series = pd.Series(hml_values, index=common_dates)

        return smb_series, hml_series

    def _calculate_momentum_factor(self, returns_df: pd.DataFrame, lookback_period: int = 252) -> pd.Series:
        """
        Berechnet Momentum-Faktor (WML - Winners Minus Losers)

        Args:
            returns_df: DataFrame mit Stock Returns
            lookback_period: Lookback-Periode für Momentum (Standard: 252 Tage = 1 Jahr)

        Returns:
            WML Series
        """
        logger.debug(f"Berechne Momentum-Faktor (Lookback: {lookback_period} Tage)...")

        # Berechne kumulative Returns über Lookback-Periode
        cumulative_returns = (1 + returns_df).rolling(window=lookback_period, min_periods=60).apply(
            lambda x: (1 + x).prod() - 1, raw=True
        )

        wml_values = []
        for date in returns_df.index:
            if date in cumulative_returns.index:
                date_cum_returns = cumulative_returns.loc[date].dropna()

                if len(date_cum_returns) >= 2:
                    # Sortiere nach kumulativen Returns
                    sorted_returns = date_cum_returns.sort_values()
                    n = len(sorted_returns)
                    
                    # Top 30% = Winners, Bottom 30% = Losers
                    winners = sorted_returns.iloc[int(0.7 * n):].index
                    losers = sorted_returns.iloc[:int(0.3 * n)].index

                    # Hole aktuelle Returns
                    if date in returns_df.index:
                        current_returns = returns_df.loc[date]
                        winner_returns = [current_returns.get(s, np.nan) for s in winners if s in current_returns.index]
                        loser_returns = [current_returns.get(s, np.nan) for s in losers if s in current_returns.index]

                        if winner_returns and loser_returns:
                            winner_avg = np.nanmean([r for r in winner_returns if not pd.isna(r)])
                            loser_avg = np.nanmean([r for r in loser_returns if not pd.isna(r)])
                            wml_values.append(winner_avg - loser_avg)
                        else:
                            wml_values.append(np.nan)
                    else:
                        wml_values.append(np.nan)
                else:
                    wml_values.append(np.nan)
            else:
                wml_values.append(np.nan)

        wml_series = pd.Series(wml_values, index=returns_df.index)
        return wml_series


def calculate_fama_french_factors(
    portfolio_name: str,
    price_df: pd.DataFrame,
    company_df: pd.DataFrame,
    config_path: str = "config.yaml"
) -> pd.DataFrame:
    """
    Convenience-Funktion zum Berechnen der Fama-French Faktoren

    Args:
        portfolio_name: Name des Portfolios (z.B. "dax", "sdax")
        price_df: DataFrame mit Preisdaten
        company_df: DataFrame mit fundamentalen Daten
        config_path: Pfad zur Config-Datei

    Returns:
        DataFrame mit Faktoren (Date, Mkt_Rf, SMB, HML, WML)
    """
    config = ConfigManager(config_path)
    
    # Hole Index-Spalte aus Config
    portfolio_config = config.get(f"data.portfolios.{portfolio_name}")
    if not portfolio_config:
        raise ValueError(f"Portfolio '{portfolio_name}' nicht in Config gefunden")
    
    index_name = portfolio_config.get("index", ".GDAXI")
    index_col = f"{index_name}_TRDPRC_1"  # Standard-Format

    # Erstelle Model und berechne Faktoren
    model = FamaFrenchFactorModel(config_path)
    factors_df = model.calculate_factors(
        price_df=price_df,
        company_df=company_df,
        index_col=index_col,
        portfolio_name=portfolio_name
    )

    return factors_df


if __name__ == "__main__":
    # Test
    from logger_config import setup_logging
    setup_logging()

    print("FamaFrench.py - Test")
    print("="*70)

