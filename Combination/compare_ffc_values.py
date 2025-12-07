#!/usr/bin/env python3
"""Vergleicht FFC=Yes vs FFC=No Werte in model_comparison.xlsx"""

import pandas as pd
from pathlib import Path

print("="*70)
print("VERGLEICH: FFC=Yes vs FFC=No")
print("="*70)

file_path = Path("Results/model_comparison.xlsx")
df = pd.read_excel(file_path, sheet_name='Full_Comparison')

# Vergleiche für jedes Portfolio/Period/Model Paar
print("\nDETAILLIERTER VERGLEICH:")
print("-"*70)

problematic = []
for (portfolio, period, model), group in df.groupby(['Portfolio', 'Period', 'Model']):
    if len(group) != 2:
        print(f"\n⚠️ {portfolio} | {period} | {model}: {len(group)} Einträge (erwartet: 2)")
        continue
    
    no_ffc = group[group['FFC_Factors'] == 'No'].iloc[0]
    yes_ffc = group[group['FFC_Factors'] == 'Yes'].iloc[0]
    
    print(f"\n{portfolio} | {period} | {model}:")
    print(f"  {'Metrik':<25} {'FFC=No':<15} {'FFC=Yes':<15} {'Gleich?':<10}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*10}")
    
    metrics_to_compare = ['R2_Test', 'R2_Train', 'MSE', 'MAE', 'Directional_Accuracy', 
                          'Directional_Accuracy_Train', 'Training_Time_s']
    
    identical_count = 0
    for metric in metrics_to_compare:
        val_no = no_ffc[metric]
        val_yes = yes_ffc[metric]
        
        # Prüfe ob identisch (mit Toleranz für Floats)
        if pd.isna(val_no) and pd.isna(val_yes):
            identical = True
        elif pd.isna(val_no) or pd.isna(val_yes):
            identical = False
        else:
            # Für Floats: prüfe mit kleiner Toleranz
            if abs(val_no - val_yes) < 1e-10:
                identical = True
            else:
                identical = False
        
        if identical:
            identical_count += 1
            status = "⚠️ IDENTISCH"
        else:
            status = "✓"
        
        print(f"  {metric:<25} {str(val_no):<15} {str(val_yes):<15} {status}")
    
    # Prüfe kritische Metriken
    critical_metrics = ['R2_Test', 'MSE', 'MAE']
    all_identical = True
    
    for metric in critical_metrics:
        val_no = no_ffc[metric]
        val_yes = yes_ffc[metric]
        
        if pd.isna(val_no) or pd.isna(val_yes):
            all_identical = False
            break
        elif abs(val_no - val_yes) >= 1e-10:
            all_identical = False
            break
    
    if all_identical:
        problematic.append({
            'Portfolio': portfolio,
            'Period': period,
            'Model': model,
            'R2_Test_No': no_ffc['R2_Test'],
            'R2_Test_Yes': yes_ffc['R2_Test'],
            'MSE_No': no_ffc['MSE'],
            'MSE_Yes': yes_ffc['MSE']
        })
        print(f"\n  ❌ PROBLEM: ALLE kritischen Werte sind identisch!")
    elif identical_count > 0:
        print(f"\n  ⚠️ WARNUNG: {identical_count} von {len(metrics_to_compare)} Metriken sind identisch")

print("\n" + "="*70)
print("ZUSAMMENFASSUNG: Identische Werte")
print("="*70)

if problematic:
    print(f"\n❌ PROBLEM: {len(problematic)} Modelle haben identische Werte für FFC=Yes und FFC=No:")
    print("-"*70)
    for p in problematic:
        print(f"  {p['Portfolio']} | {p['Period']} | {p['Model']}")
        print(f"    R2_Test: {p['R2_Test_No']} (beide gleich)")
        print(f"    MSE: {p['MSE_No']} (beide gleich)")
else:
    print("\n✓ Alle Modelle haben unterschiedliche Werte für FFC=Yes vs FFC=No")

print("\n" + "="*70)

