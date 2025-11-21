#!/usr/bin/env python3
"""Simuliert FFC-Faktoren-Berechnung mit vorhandenen Daten"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("SIMULATION: FFC-FAKTOREN-BERECHNUNG")
print("="*70)

# 1. Lade Price-Daten
print("\n1. LADE PRICE-DATEN:")
print("-"*70)
try:
    price_df = pd.read_excel('DataStorage/dax_daily.xlsx', sheet_name=0)
    print(f"  ✓ Price-Daten geladen: {price_df.shape}")
    print(f"    Columns: {list(price_df.columns)[:5]}")
    print(f"    Hat Date: {'Date' in price_df.columns}")
    
    if 'Date' in price_df.columns:
        price_df['Date'] = pd.to_datetime(price_df['Date'], errors='coerce')
        price_df = price_df.set_index('Date')
        print(f"    Date-Bereich: {price_df.index.min()} bis {price_df.index.max()}")
except Exception as e:
    print(f"  ✗ Fehler: {e}")
    exit(1)

# 2. Prüfe ob Company-Daten existieren
print("\n2. PRÜFE COMPANY-DATEN:")
print("-"*70)
company_files = [f for f in os.listdir('DataStorage') 
                 if 'company' in f.lower() 
                 and f.endswith('.xlsx') 
                 and not f.startswith('~$')
                 and 'dax' in f.lower()]

if len(company_files) == 0:
    print("  ⚠️ KEINE Company-Daten-Dateien gefunden!")
    print("  Erstelle Test-Company-Daten...")
    
    # Erstelle Test-Company-Daten mit gleichem Zeitraum
    dates = price_df.index.normalize().unique()
    company_data = {
        'Date': dates,
        'RHMG.DE_TR.CompanyMarketCapitalization': np.random.randn(len(dates)).cumsum() + 1000000,
        'RHMG.DE_TR.BookValuePerShare': np.random.randn(len(dates)).cumsum() + 50,
    }
    company_df = pd.DataFrame(company_data)
    company_df['Date'] = pd.to_datetime(company_df['Date'], errors='coerce')
    company_df = company_df.set_index('Date')
    print(f"  ✓ Test-Company-Daten erstellt: {company_df.shape}")
else:
    print(f"  ✓ Company-Daten-Datei gefunden: {company_files[0]}")
    company_df = pd.read_excel(f'DataStorage/{company_files[0]}', sheet_name=0)
    print(f"    Shape: {company_df.shape}")
    print(f"    Columns: {list(company_df.columns)[:5]}")
    
    if 'Date' in company_df.columns:
        company_df['Date'] = pd.to_datetime(company_df['Date'], errors='coerce')
        company_df = company_df.set_index('Date')
        print(f"    Date-Bereich: {company_df.index.min()} bis {company_df.index.max()}")

# 3. Prüfe gemeinsame Datenpunkte
print("\n3. PRÜFE GEMEINSAME DATENPUNKTE:")
print("-"*70)
price_index = price_df.index.normalize()
company_index = company_df.index.normalize()

common = price_index.intersection(company_index)
print(f"  Price-Daten: {len(price_index)} Datenpunkte")
print(f"  Company-Daten: {len(company_index)} Datenpunkte")
print(f"  Gemeinsame: {len(common)} Datenpunkte")

if len(common) == 0:
    print("  ⚠️ PROBLEM: Keine gemeinsamen Datenpunkte!")
    print(f"    Price: {price_index[:5].tolist()}")
    print(f"    Company: {company_index[:5].tolist()}")
else:
    print(f"  ✓ Gemeinsame Datenpunkte gefunden!")

print("\n" + "="*70)

