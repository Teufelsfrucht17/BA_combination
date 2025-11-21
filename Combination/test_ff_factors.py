#!/usr/bin/env python3
"""Test-Script für FFC-Faktoren-Berechnung ohne API-Calls"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*70)
print("TEST: FFC-FAKTOREN-BERECHNUNG")
print("="*70)

# 1. Erstelle Test-Daten mit gleichem Zeitraum
print("\n1. ERSTELLE TEST-DATEN:")
print("-"*70)

# Price-Daten: 2024-01-01 bis 2025-11-15 (täglich)
start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 11, 15)
dates = pd.date_range(start=start_date, end=end_date, freq='D')
# Nur Werktage
dates = dates[dates.weekday < 5]  # Montag=0, Freitag=4

print(f"  Price-Daten: {len(dates)} Datenpunkte")
print(f"    Von: {dates.min()}")
print(f"    Bis: {dates.max()}")

# Company-Daten: Gleicher Zeitraum
company_dates = dates.copy()
print(f"  Company-Daten: {len(company_dates)} Datenpunkte")
print(f"    Von: {company_dates.min()}")
print(f"    Bis: {company_dates.max()}")

# 2. Erstelle Test-DataFrames
print("\n2. ERSTELLE TEST-DATAFRAMES:")
print("-"*70)

# Price DataFrame mit MultiIndex-Spalten
price_data = {}
for stock in ['RHMG.DE', 'ENR1n.DE', 'TKAG.DE']:
    price_data[(stock, 'TRDPRC_1')] = np.random.randn(len(dates)).cumsum() + 100
    price_data[('.GDAXI', 'TRDPRC_1')] = np.random.randn(len(dates)).cumsum() + 15000

price_df = pd.DataFrame(price_data, index=dates)
print(f"  Price-DF Shape: {price_df.shape}")
print(f"  Price-DF Columns: {list(price_df.columns)[:5]}")
print(f"  Price-DF Index-Type: {type(price_df.index)}")

# Company DataFrame
company_data = {
    'Date': company_dates,
    'RHMG.DE_TR.CompanyMarketCapitalization': np.random.randn(len(company_dates)).cumsum() + 1000000,
    'RHMG.DE_TR.BookValuePerShare': np.random.randn(len(company_dates)).cumsum() + 50,
    'ENR1n.DE_TR.CompanyMarketCapitalization': np.random.randn(len(company_dates)).cumsum() + 2000000,
    'ENR1n.DE_TR.BookValuePerShare': np.random.randn(len(company_dates)).cumsum() + 60,
    'TKAG.DE_TR.CompanyMarketCapitalization': np.random.randn(len(company_dates)).cumsum() + 1500000,
    'TKAG.DE_TR.BookValuePerShare': np.random.randn(len(company_dates)).cumsum() + 55,
}
company_df = pd.DataFrame(company_data)
company_df = company_df.set_index('Date')
print(f"  Company-DF Shape: {company_df.shape}")
print(f"  Company-DF Columns: {list(company_df.columns)[:5]}")
print(f"  Company-DF Index-Type: {type(company_df.index)}")

# 3. Teste Datumskonvertierung (wie in FamaFrench.py)
print("\n3. TESTE DATUMSKONVERTIERUNG:")
print("-"*70)

# Normalisiere beide Indizes
price_index = price_df.index
company_index = company_df.index

if isinstance(price_index, pd.DatetimeIndex):
    if price_index.tz is not None:
        price_index = price_index.tz_localize(None)
    price_index = price_index.normalize()

if isinstance(company_index, pd.DatetimeIndex):
    if company_index.tz is not None:
        company_index = company_index.tz_localize(None)
    company_index = company_index.normalize()

print(f"  Price Index normalisiert: {price_index.min()} bis {price_index.max()}")
print(f"  Company Index normalisiert: {company_index.min()} bis {company_index.max()}")

# 4. Teste gemeinsame Datenpunkte
print("\n4. TESTE GEMEINSAME DATENPUNKTE:")
print("-"*70)

common_dates = price_index.intersection(company_index)
print(f"  Gemeinsame Datenpunkte: {len(common_dates)}")
print(f"  Erwartet: ~{len(dates)} (alle sollten übereinstimmen)")

if len(common_dates) > 0:
    print(f"  ✓ ERFOLG: {len(common_dates)} gemeinsame Datenpunkte gefunden!")
    print(f"    Erste 5: {common_dates[:5].tolist()}")
    print(f"    Letzte 5: {common_dates[-5:].tolist()}")
else:
    print(f"  ✗ FEHLER: Keine gemeinsamen Datenpunkte!")

# 5. Teste Spaltenerkennung
print("\n5. TESTE SPALTENERKENNUNG:")
print("-"*70)

stock_price_cols = []
for col in price_df.columns:
    col_str = str(col)
    is_tuple = isinstance(col, tuple)
    
    if is_tuple:
        if len(col) >= 2:
            first_level = str(col[0])
            second_level = str(col[1])
            if '.DE' in first_level and 'TRDPRC_1' in second_level:
                if not any(idx in first_level for idx in ['.GDAXI', '.SDAXI', '.V1XI']):
                    stock_price_cols.append(col)

print(f"  Gefundene Stock-Preis-Spalten: {len(stock_price_cols)}")
for col in stock_price_cols:
    print(f"    - {col}")

index_col = None
for col in price_df.columns:
    col_str = str(col)
    is_tuple = isinstance(col, tuple)
    if is_tuple and len(col) >= 2:
        if str(col[0]) == '.GDAXI' and 'TRDPRC_1' in str(col[1]):
            index_col = col
            break

if index_col:
    print(f"  ✓ Index-Spalte gefunden: {index_col}")
else:
    print(f"  ✗ Index-Spalte nicht gefunden!")

print("\n" + "="*70)
print("TEST ABGESCHLOSSEN")
print("="*70)

if len(common_dates) > 0 and len(stock_price_cols) > 0 and index_col:
    print("\n✓ ALLE TESTS BESTANDEN - FFC-Faktoren sollten berechnet werden können!")
else:
    print("\n✗ EINIGE TESTS FEHLGESCHLAGEN - Bitte prüfen!")

