"""Check company data files"""

import pandas as pd
import os
from pathlib import Path

print("="*70)
print("CHECK COMPANY DATA FILES")
print("="*70)

data_dir = Path("DataStorage")
if not data_dir.exists():
    print("\nDataStorage directory does not exist!")
    exit(1)

company_files = [f for f in os.listdir(data_dir) 
                 if 'company' in f.lower() 
                 and f.endswith('.xlsx') 
                 and not f.startswith('~$')]

print(f"\nCompany data files found: {len(company_files)}")
for f in company_files:
    print(f"  - {f}")

if len(company_files) == 0:
    print("\nNO company data files found!")
    print("Files must be retrieved.")
    exit(1)

for filename in company_files:
    filepath = data_dir / filename
    print(f"\n{'='*70}")
    print(f"DATEI: {filename}")
    print(f"{'='*70}")
    
    df = pd.read_excel(filepath, sheet_name=0)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)[:10]}")
    print(f"  Index-Type: {type(df.index)}")
    print(f"  Has Date column: {'Date' in df.columns}")
    
    if 'Date' in df.columns:
        dates = pd.to_datetime(df['Date'], errors='coerce')
        dates_clean = dates.dropna()
        print(f"  Date range: {dates_clean.min()} to {dates_clean.max()}")
        print(f"  Valid dates: {len(dates_clean)} of {len(df)}")
        print(f"  First 5 dates: {dates_clean.head().tolist()}")
    else:
        print("  NO Date column found!")
        print(f"  Available columns: {list(df.columns)}")
    
    if df.empty:
        print("  DataFrame is EMPTY!")
    else:
        print("  DataFrame contains data")

print("\n" + "="*70)
