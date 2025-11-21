#!/usr/bin/env python3
"""Test-Script um zu prüfen, ob Company-Daten geholt werden können"""

import sys
import traceback

print("="*70)
print("TEST: COMPANY-DATEN ABRUF")
print("="*70)

try:
    from ConfigManager import ConfigManager
    from Datagrabber import DataGrabber
    
    print("\n1. INITIALISIERE CONFIG UND GRABBER:")
    print("-"*70)
    config = ConfigManager("config.yaml")
    grabber = DataGrabber("config.yaml")
    
    print("  ✓ Config geladen")
    print("  ✓ DataGrabber initialisiert")
    
    print("\n2. PRÜFE PORTFOLIO-KONFIGURATION:")
    print("-"*70)
    portfolios = config.get("data.portfolios", {})
    print(f"  Gefundene Portfolios: {list(portfolios.keys())}")
    
    for portfolio_name in portfolios.keys():
        portfolio_config = config.get(f"data.portfolios.{portfolio_name}")
        universe = portfolio_config.get("universe", [])
        print(f"  {portfolio_name}: {len(universe)} Aktien")
        print(f"    Universe: {universe[:3]}...")
    
    print("\n3. PRÜFE PERIOD-KONFIGURATION:")
    print("-"*70)
    period_config = config.get("data.periods.daily")
    if period_config:
        print(f"  Daily Period: {period_config.get('start')} bis {period_config.get('end')}")
    else:
        print("  ⚠️ Keine Daily Period Config gefunden!")
    
    print("\n4. TESTE COMPANY-DATEN ABRUF (nur für DAX):")
    print("-"*70)
    print("  Hinweis: Dies erfordert eine aktive LSEG API-Verbindung")
    print("  Falls die API nicht verfügbar ist, wird ein Fehler auftreten")
    
    # Test nur für DAX
    dax_config = config.get("data.portfolios.dax")
    if dax_config:
        universe = dax_config.get("universe", [])
        period_config = config.get("data.periods.daily")
        
        if period_config:
            company_params = {
                'Curn': 'USD',
                'SDate': period_config.get('start', '2024-01-01'),
                'EDate': period_config.get('end', '2025-11-15'),
                'Frq': 'D'
            }
            print(f"  Parameter: {company_params}")
            print(f"  Universe: {universe}")
            print(f"  Versuche Company-Daten zu holen...")
            
            try:
                import LSEG as LS
                company_df = LS.getCompanyData(universe=universe, parameters=company_params)
                
                if company_df is not None and not company_df.empty:
                    print(f"  ✓ Company-Daten erfolgreich geholt: {company_df.shape}")
                    print(f"    Spalten: {list(company_df.columns)[:5]}")
                    print(f"    Hat Date-Spalte: {'Date' in company_df.columns}")
                    if 'Date' in company_df.columns:
                        print(f"    Date-Bereich: {company_df['Date'].min()} bis {company_df['Date'].max()}")
                else:
                    print("  ⚠️ Company-Daten sind leer!")
            except Exception as e:
                print(f"  ✗ Fehler beim Abrufen der Company-Daten: {e}")
                print(f"    Traceback:")
                traceback.print_exc()
        else:
            print("  ⚠️ Keine Period Config gefunden!")
    else:
        print("  ⚠️ Keine DAX Config gefunden!")
    
    print("\n" + "="*70)
    print("TEST ABGESCHLOSSEN")
    print("="*70)
    
except Exception as e:
    print(f"\n✗ FEHLER: {e}")
    traceback.print_exc()
    sys.exit(1)

