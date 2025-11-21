#!/usr/bin/env python3
"""Prüft Company-Daten-Dateien"""

import pandas as pd
import os
from pathlib import Path

print("="*70)
print("PRÜFE COMPANY-DATEN-DATEIEN")
print("="*70)

data_dir = Path("DataStorage")
if not data_dir.exists():
    print(f"\n⚠️ DataStorage Verzeichnis existiert nicht!")
    exit(1)

# Finde alle Company-Daten-Dateien
company_files = [f for f in os.listdir(data_dir) 
                 if 'company' in f.lower() 
                 and f.endswith('.xlsx') 
                 and not f.startswith('~$')]

print(f"\nGefundene Company-Daten-Dateien: {len(company_files)}")
for f in company_files:
    print(f"  - {f}")

if len(company_files) == 0:
    print("\n⚠️ KEINE Company-Daten-Dateien gefunden!")
    print("   Die Dateien müssen neu geholt werden.")
    exit(1)

# Prüfe jede Datei
for filename in company_files:
    filepath = data_dir / filename
    print(f"\n{'='*70}")
    print(f"DATEI: {filename}")
    print(f"{'='*70}")
    
    try:
        # Lese erste Sheet
        df = pd.read_excel(filepath, sheet_name=0)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)[:10]}")
        print(f"  Index-Type: {type(df.index)}")
        print(f"  Hat Date-Spalte: {'Date' in df.columns}")
        
        if 'Date' in df.columns:
            dates = pd.to_datetime(df['Date'], errors='coerce')
            dates_clean = dates.dropna()
            print(f"  Date-Bereich: {dates_clean.min()} bis {dates_clean.max()}")
            print(f"  Anzahl gültige Dates: {len(dates_clean)} von {len(df)}")
            print(f"  Erste 5 Dates: {dates_clean.head().tolist()}")
        else:
            print(f"  ⚠️ KEINE Date-Spalte gefunden!")
            print(f"  Verfügbare Spalten: {list(df.columns)}")
        
        # Prüfe ob DataFrame leer ist
        if df.empty:
            print(f"  ⚠️ DataFrame ist LEER!")
        else:
            print(f"  ✓ DataFrame hat Daten")
            
    except Exception as e:
        print(f"  ✗ Fehler beim Lesen: {e}")

print("\n" + "="*70)

