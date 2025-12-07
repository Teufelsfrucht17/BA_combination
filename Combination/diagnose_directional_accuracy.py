#!/usr/bin/env python3
"""Diagnose warum Directional Accuracy identisch ist für FFC=Yes und FFC=No"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("DIAGNOSE: Directional Accuracy Problem")
print("="*70)

# Lade Excel-Datei
file_path = Path("Results/model_comparison.xlsx")
df = pd.read_excel(file_path, sheet_name='Full_Comparison')

# Fokus auf SDAX daily OLS und PyTorch NN
print("\n1. SDAX Portfolio | daily - Vergleich:")
print("-"*70)

sdax_daily = df[(df['Portfolio'] == 'SDAX Portfolio') & (df['Period'] == 'daily')]
print(sdax_daily[['Model', 'FFC_Factors', 'Directional_Accuracy', 'R2_Test', 'MSE', 'MAE']].to_string())

# Prüfe alle Modelle mit identischer Directional Accuracy
print("\n2. Alle Modelle mit identischer Directional Accuracy (FFC=Yes vs FFC=No):")
print("-"*70)

problematic = []
for (portfolio, period, model), group in df.groupby(['Portfolio', 'Period', 'Model']):
    if len(group) != 2:
        continue
    
    no_ffc = group[group['FFC_Factors'] == 'No'].iloc[0]
    yes_ffc = group[group['FFC_Factors'] == 'Yes'].iloc[0]
    
    da_no = no_ffc['Directional_Accuracy']
    da_yes = yes_ffc['Directional_Accuracy']
    
    # Prüfe ob identisch (mit Toleranz)
    if pd.isna(da_no) or pd.isna(da_yes):
        continue
    
    if abs(da_no - da_yes) < 1e-10:
        # Prüfe ob andere Metriken unterschiedlich sind
        r2_diff = abs(no_ffc['R2_Test'] - yes_ffc['R2_Test'])
        mse_diff = abs(no_ffc['MSE'] - yes_ffc['MSE'])
        
        if r2_diff > 1e-6 or mse_diff > 1e-10:
            problematic.append({
                'Portfolio': portfolio,
                'Period': period,
                'Model': model,
                'DA_No': da_no,
                'DA_Yes': da_yes,
                'R2_No': no_ffc['R2_Test'],
                'R2_Yes': yes_ffc['R2_Test'],
                'MSE_No': no_ffc['MSE'],
                'MSE_Yes': yes_ffc['MSE']
            })

if problematic:
    print(f"\n❌ {len(problematic)} Modelle haben identische Directional Accuracy, aber unterschiedliche R²/MSE:")
    for p in problematic:
        print(f"\n  {p['Portfolio']} | {p['Period']} | {p['Model']}:")
        print(f"    Directional Accuracy: {p['DA_No']:.6f} (beide gleich)")
        print(f"    R²: {p['R2_No']:.6f} vs {p['R2_Yes']:.6f} (Diff: {abs(p['R2_No'] - p['R2_Yes']):.6f})")
        print(f"    MSE: {p['MSE_No']:.6f} vs {p['MSE_Yes']:.6f} (Diff: {abs(p['MSE_No'] - p['MSE_Yes']):.6f})")
        print(f"    → Vorhersagen sind unterschiedlich, aber Vorzeichen gleich!")
else:
    print("\n✓ Keine Probleme gefunden")

print("\n" + "="*70)
print("3. Mögliche Ursachen:")
print("-"*70)
print("  - FFC-Faktoren ändern die Vorhersagewerte, aber nicht die Vorzeichen")
print("  - Train-Test-Split könnte unterschiedlich sein (unterschiedliche Zeilenzahlen)")
print("  - FFC-Faktoren könnten nicht korrekt zu Features hinzugefügt werden")
print("  - Scaling könnte FFC-Faktoren zu sehr reduzieren")

print("\n" + "="*70)

