#!/usr/bin/env python3
"""Test script to verify company data retrieval"""

from ConfigManager import ConfigManager
import LSEG as LS

print("="*70)
print("TEST: COMPANY DATA FETCH")
print("="*70)

print("\n1. INITIALIZE CONFIG:")
print("-"*70)
config = ConfigManager("config.yaml")

print("  Config loaded")

print("\n2. CHECK PORTFOLIO CONFIGURATION:")
print("-"*70)
portfolios = config.get("data.portfolios", {})
print(f"  Portfolios: {list(portfolios.keys())}")

for portfolio_name in portfolios.keys():
    portfolio_config = config.get(f"data.portfolios.{portfolio_name}")
    universe = portfolio_config.get("universe", [])
    print(f"  {portfolio_name}: {len(universe)} stocks")
    print(f"    Universe: {universe[:3]}...")

print("\n3. CHECK PERIOD CONFIGURATION:")
print("-"*70)
period_config = config.get("data.periods.daily")
if period_config:
    print(f"  Daily Period: {period_config.get('start')} to {period_config.get('end')}")
else:
    print("  No daily period config found!")

print("\n4. TEST COMPANY DATA FETCH (DAX only):")
print("-"*70)
print("  Note: Requires an active LSEG API connection")
print("  If the API is unavailable, an error will occur")

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
        print("  Attempting to fetch company data...")
        
        company_df = LS.getCompanyData(universe=universe, parameters=company_params)
        
        if company_df is not None and not company_df.empty:
            print(f"  Company data fetched: {company_df.shape}")
            print(f"    Columns: {list(company_df.columns)[:5]}")
            print(f"    Has Date column: {'Date' in company_df.columns}")
            if 'Date' in company_df.columns:
                print(f"    Date range: {company_df['Date'].min()} to {company_df['Date'].max()}")
        else:
            print("  Company data is empty!")
    else:
        print("  No period config found!")
else:
    print("  No DAX config found!")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
