"""Check structure of company and price data"""

import pandas as pd

print("="*70)
print("DATA STRUCTURE CHECK")
print("="*70)

print("\n1. COMPANY DATA (dax_company_data.xlsx):")
print("-"*70)
company_df = pd.read_excel('DataStorage/dax_company_data.xlsx', sheet_name=0)
print(f"  Shape: {company_df.shape}")
print(f"  Columns: {list(company_df.columns)[:10]}")
print(f"  Index-Type: {type(company_df.index)}")
print(f"  Has Date column: {'Date' in company_df.columns}")
if 'Date' in company_df.columns:
    print(f"  Date column type: {company_df['Date'].dtype}")
    print("  First 5 dates:")
    print(f"    {company_df['Date'].head().tolist()}")
    print("  Last 5 dates:")
    print(f"    {company_df['Date'].tail().tolist()}")
print(f"  Index first 5: {company_df.index[:5].tolist()}")

print("\n2. PRICE DATA (dax_daily.xlsx):")
print("-"*70)
price_df = pd.read_excel('DataStorage/dax_daily.xlsx', sheet_name=0)
print(f"  Shape: {price_df.shape}")
print(f"  Columns (first 10): {list(price_df.columns)[:10]}")
print(f"  Index-Type: {type(price_df.index)}")
print(f"  Has Date column: {'Date' in price_df.columns}")
if 'Date' in price_df.columns:
    print(f"  Date column type: {price_df['Date'].dtype}")
    print("  First 5 dates:")
    print(f"    {price_df['Date'].head().tolist()}")
    print("  Last 5 dates:")
    print(f"    {price_df['Date'].tail().tolist()}")
print(f"  Index first 5: {price_df.index[:5].tolist()}")

print("\n3. DATE RANGE COMPARISON:")
print("-"*70)
company_df = pd.read_excel('DataStorage/dax_company_data.xlsx', sheet_name=0)
price_df = pd.read_excel('DataStorage/dax_daily.xlsx', sheet_name=0)

if 'Date' in company_df.columns and 'Date' in price_df.columns:
    company_dates = pd.to_datetime(company_df['Date'], errors='coerce').dt.normalize()
    price_dates = pd.to_datetime(price_df['Date'], errors='coerce').dt.normalize()
    
    print(f"  Company data: {company_dates.min()} to {company_dates.max()} ({len(company_dates)} points)")
    print(f"  Price data: {price_dates.min()} to {price_dates.max()} ({len(price_dates)} points)")
    
    company_index = pd.DatetimeIndex(company_dates.dropna())
    price_index = pd.DatetimeIndex(price_dates.dropna())
    common = company_index.intersection(price_index)
    print(f"  Common data points: {len(common)}")
    
    if len(common) == 0:
        print("  PROBLEM: No overlapping data points!")
        print(f"    Company first 10: {company_dates.dropna().head(10).tolist()}")
        print(f"    Price first 10: {price_dates.dropna().head(10).tolist()}")

print("\n" + "="*70)
