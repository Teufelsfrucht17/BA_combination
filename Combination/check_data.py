#!/usr/bin/env python3
"""Prüft die Datenstruktur von Company- und Price-Daten"""

import pandas as pd
import sys

print("="*70)
print("DATENSTRUKTUR-PRÜFUNG")
print("="*70)

# 1. Company-Daten prüfen
print("\n1. COMPANY-DATEN (dax_company_data.xlsx):")
print("-"*70)
try:
    company_df = pd.read_excel('DataStorage/dax_company_data.xlsx', sheet_name=0)
    print(f"  Shape: {company_df.shape}")
    print(f"  Columns: {list(company_df.columns)[:10]}")
    print(f"  Index-Type: {type(company_df.index)}")
    print(f"  Hat Date-Spalte: {'Date' in company_df.columns}")
    if 'Date' in company_df.columns:
        print(f"  Date-Spalte Type: {company_df['Date'].dtype}")
        print(f"  Erste 5 Dates:")
        print(f"    {company_df['Date'].head().tolist()}")
        print(f"  Letzte 5 Dates:")
        print(f"    {company_df['Date'].tail().tolist()}")
    print(f"  Index erste 5: {company_df.index[:5].tolist()}")
except Exception as e:
    print(f"  FEHLER: {e}")

# 2. Price-Daten (Daily) prüfen
print("\n2. PRICE-DATEN (dax_daily.xlsx):")
print("-"*70)
try:
    price_df = pd.read_excel('DataStorage/dax_daily.xlsx', sheet_name=0)
    print(f"  Shape: {price_df.shape}")
    print(f"  Columns (erste 10): {list(price_df.columns)[:10]}")
    print(f"  Index-Type: {type(price_df.index)}")
    print(f"  Hat Date-Spalte: {'Date' in price_df.columns}")
    if 'Date' in price_df.columns:
        print(f"  Date-Spalte Type: {price_df['Date'].dtype}")
        print(f"  Erste 5 Dates:")
        print(f"    {price_df['Date'].head().tolist()}")
        print(f"  Letzte 5 Dates:")
        print(f"    {price_df['Date'].tail().tolist()}")
    print(f"  Index erste 5: {price_df.index[:5].tolist()}")
except Exception as e:
    print(f"  FEHLER: {e}")

# 3. Vergleich der Datumsbereiche
print("\n3. DATUMS-VERGLEICH:")
print("-"*70)
try:
    company_df = pd.read_excel('DataStorage/dax_company_data.xlsx', sheet_name=0)
    price_df = pd.read_excel('DataStorage/dax_daily.xlsx', sheet_name=0)
    
    if 'Date' in company_df.columns and 'Date' in price_df.columns:
        company_dates = pd.to_datetime(company_df['Date'], errors='coerce').dt.normalize()
        price_dates = pd.to_datetime(price_df['Date'], errors='coerce').dt.normalize()
        
        print(f"  Company-Daten: {company_dates.min()} bis {company_dates.max()} ({len(company_dates)} Datenpunkte)")
        print(f"  Price-Daten: {price_dates.min()} bis {price_dates.max()} ({len(price_dates)} Datenpunkte)")
        
        # Konvertiere zu Index für intersection
        company_index = pd.DatetimeIndex(company_dates.dropna())
        price_index = pd.DatetimeIndex(price_dates.dropna())
        common = company_index.intersection(price_index)
        print(f"  Gemeinsame Datenpunkte: {len(common)}")
        
        if len(common) == 0:
            print(f"  ⚠️ PROBLEM: Keine gemeinsamen Datenpunkte!")
            print(f"    Company erste 10: {company_dates.dropna().head(10).tolist()}")
            print(f"    Price erste 10: {price_dates.dropna().head(10).tolist()}")
except Exception as e:
    print(f"  FEHLER: {e}")

print("\n" + "="*70)

