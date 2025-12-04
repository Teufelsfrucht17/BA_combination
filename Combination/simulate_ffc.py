"""Simulate FFC factor calculation with existing data"""

import pandas as pd
import numpy as np
import os

print("="*70)
print("SIMULATION: FFC FACTOR CALCULATION")
print("="*70)

print("\n1. LOAD PRICE DATA:")
print("-"*70)
price_df = pd.read_excel('DataStorage/dax_daily.xlsx', sheet_name=0)
print(f"  Price data loaded: {price_df.shape}")
print(f"    Columns: {list(price_df.columns)[:5]}")
print(f"    Hat Date: {'Date' in price_df.columns}")

if 'Date' in price_df.columns:
    price_df['Date'] = pd.to_datetime(price_df['Date'], errors='coerce')
    price_df = price_df.set_index('Date')
    print(f"    Date range: {price_df.index.min()} to {price_df.index.max()}")

print("\n2. CHECK COMPANY DATA:")
print("-"*70)
company_files = [f for f in os.listdir('DataStorage') 
                 if 'company' in f.lower() 
                 and f.endswith('.xlsx') 
                 and not f.startswith('~$')
                 and 'dax' in f.lower()]

if len(company_files) == 0:
    print("  NO company data files found!")
    print("  Creating synthetic company data...")
    
    dates = price_df.index.normalize().unique()
    company_data = {
        'Date': dates,
        'RHMG.DE_TR.CompanyMarketCapitalization': np.random.randn(len(dates)).cumsum() + 1000000,
        'RHMG.DE_TR.BookValuePerShare': np.random.randn(len(dates)).cumsum() + 50,
    }
    company_df = pd.DataFrame(company_data)
    company_df['Date'] = pd.to_datetime(company_df['Date'], errors='coerce')
    company_df = company_df.set_index('Date')
    print(f"  Test company data created: {company_df.shape}")
else:
    print(f"  Company data file found: {company_files[0]}")
    company_df = pd.read_excel(f'DataStorage/{company_files[0]}', sheet_name=0)
    print(f"    Shape: {company_df.shape}")
    print(f"    Columns: {list(company_df.columns)[:5]}")
    
    if 'Date' in company_df.columns:
        company_df['Date'] = pd.to_datetime(company_df['Date'], errors='coerce')
        company_df = company_df.set_index('Date')
        print(f"    Date range: {company_df.index.min()} to {company_df.index.max()}")

print("\n3. CHECK OVERLAPPING DATES:")
print("-"*70)
price_index = price_df.index.normalize()
company_index = company_df.index.normalize()

common = price_index.intersection(company_index)
print(f"  Price data points: {len(price_index)}")
print(f"  Company data points: {len(company_index)}")
print(f"  Overlap: {len(common)}")

if len(common) == 0:
    print("  PROBLEM: No overlapping data points!")
    print(f"    Price: {price_index[:5].tolist()}")
    print(f"    Company: {company_index[:5].tolist()}")
else:
    print("  Overlap found!")

print("\n" + "="*70)
