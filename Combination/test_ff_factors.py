#!/usr/bin/env python3
"""Test script for FFC factor calculation without API calls"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*70)
print("TEST: FFC FACTOR CALCULATION")
print("="*70)

# 1. Create test data with matching range
print("\n1. CREATE TEST DATA:")
print("-"*70)

# Price data: 2024-01-01 to 2025-11-15 (daily)
start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 11, 15)
dates = pd.date_range(start=start_date, end=end_date, freq='D')
# Only weekdays
dates = dates[dates.weekday < 5]  # Monday=0, Friday=4

print(f"  Price data points: {len(dates)}")
print(f"    From: {dates.min()}")
print(f"    To: {dates.max()}")

# Company data: same range
company_dates = dates.copy()
print(f"  Company data points: {len(company_dates)}")
print(f"    From: {company_dates.min()}")
print(f"    To: {company_dates.max()}")

# 2. Create test DataFrames
print("\n2. CREATE TEST DATAFRAMES:")
print("-"*70)

# Price DataFrame mit MultiIndex-Spalten
price_data = {}
for stock in ['RHMG.DE', 'ENR1n.DE', 'TKAG.DE']:
    price_data[(stock, 'TRDPRC_1')] = np.random.randn(len(dates)).cumsum() + 100
    price_data[('.GDAXI', 'TRDPRC_1')] = np.random.randn(len(dates)).cumsum() + 15000

price_df = pd.DataFrame(price_data, index=dates)
print(f"  Price DF shape: {price_df.shape}")
print(f"  Price DF columns: {list(price_df.columns)[:5]}")
print(f"  Price DF index type: {type(price_df.index)}")

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
print(f"  Company DF shape: {company_df.shape}")
print(f"  Company DF columns: {list(company_df.columns)[:5]}")
print(f"  Company DF index type: {type(company_df.index)}")

# 3. Test date normalization (as in FamaFrench.py)
print("\n3. TEST DATE NORMALIZATION:")
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

print(f"  Price index normalized: {price_index.min()} to {price_index.max()}")
print(f"  Company index normalized: {company_index.min()} to {company_index.max()}")

# 4. Test overlapping dates
print("\n4. TEST OVERLAPPING DATES:")
print("-"*70)

common_dates = price_index.intersection(company_index)
print(f"  Overlapping dates: {len(common_dates)}")
print(f"  Expected: ~{len(dates)} (all should match)")

if len(common_dates) > 0:
    print(f"  SUCCESS: {len(common_dates)} overlapping dates found!")
    print(f"    First 5: {common_dates[:5].tolist()}")
    print(f"    Last 5: {common_dates[-5:].tolist()}")
else:
    print("  ERROR: No overlapping dates!")

# 5. Test column detection
print("\n5. TEST COLUMN DETECTION:")
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

print(f"  Stock price columns found: {len(stock_price_cols)}")
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
    print(f"  Index column found: {index_col}")
else:
    print("  Index column not found!")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)

if len(common_dates) > 0 and len(stock_price_cols) > 0 and index_col:
    print("\nALL TESTS PASSED - FFC factors should be computable!")
else:
    print("\nSOME TESTS FAILED - Please investigate!")
