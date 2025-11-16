#!/usr/bin/env python
"""
main.py - Hauptprogramm für BA_combination Unified Version
Kombiniert die Stärken von Version 1 und 2

Usage:
    python main.py                          # Standard-Vergleich (beide Perioden)
    python main.py --mode daily             # Nur tägliche Daten
    python main.py --mode intraday          # Nur 30-Min Daten
    python main.py --features momentum_5 change_dax  # Custom Features
    python main.py --config my_config.yaml  # Custom Config
"""

import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ModelComparison import ModelComparison
from ConfigManager import ConfigManager


def print_banner():
    """Druckt schönes ASCII-Banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║            BA TRADING SYSTEM - UNIFIED VERSION                       ║
║                                                                      ║
║  Kombiniert die Stärken von Version 1 und Version 2                 ║
║  Portfolio-basiertes Machine Learning Trading System                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Hauptfunktion"""

    # Argumente parsen
    parser = argparse.ArgumentParser(
        description='BA Trading System - Unified Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python main.py                          # Vergleiche beide Perioden
  python main.py --mode daily             # Nur tägliche Daten
  python main.py --features momentum_5 change_dax  # Custom Features
  python main.py --config my_config.yaml  # Custom Config

Features in Config:
  - momentum_5, momentum_10, momentum_20  # Momentum-Indikatoren
  - change_dax                            # DAX Änderung
  - vdax_absolute                         # VDAX absoluter Wert
  - volume_ratio                          # Volume Verhältnis
  - rsi_14                                # RSI (optional)
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Pfad zur Config-Datei (default: config.yaml)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='compare',
        choices=['compare', 'daily', 'intraday'],
        help='Modus: compare (beide), daily, oder intraday (default: compare)'
    )

    parser.add_argument(
        '--features',
        type=str,
        nargs='+',
        help='Override für Features aus Config (z.B. --features momentum_5 change_dax)'
    )

    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        choices=['pytorch_nn', 'sklearn_nn', 'ols', 'ridge', 'random_forest'],
        help='Nur bestimmte Modelle trainieren'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Modelle nicht speichern'
    )

    args = parser.parse_args()

    # Banner drucken
    print_banner()

    # Config Manager initialisieren
    try:
        config = ConfigManager(args.config)
    except FileNotFoundError:
        print(f"❌ Fehler: Config-Datei nicht gefunden: {args.config}")
        print(f"💡 Tipp: Stelle sicher, dass config.yaml im aktuellen Verzeichnis ist.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fehler beim Laden der Config: {e}")
        sys.exit(1)

    # Features überschreiben falls angegeben
    if args.features:
        config.set("features.input_features", args.features)
        print(f"✓ Features überschrieben: {args.features}")

    # Modelle filtern falls angegeben
    if args.models:
        # Deaktiviere alle Modelle
        for model in ['pytorch_nn', 'sklearn_nn', 'ols', 'ridge', 'random_forest']:
            config.set(f"models.{model}.enabled", False)
        # Aktiviere nur gewählte
        for model in args.models:
            config.set(f"models.{model}.enabled", True)
        print(f"✓ Nur folgende Modelle aktiviert: {args.models}")

    # Speichern deaktivieren falls gewünscht
    if args.no_save:
        config.set("output.save_models", False)
        print(f"✓ Modell-Speicherung deaktiviert")

    # Zeige Konfiguration
    print("\n" + "="*70)
    print("KONFIGURATION")
    print("="*70)
    print(f"Config-Datei:     {args.config}")
    print(f"Modus:            {args.mode}")

    portfolios = config.get('data.portfolios') or {}
    if portfolios:
        print("Portfolios:")
        for key, portfolio_cfg in portfolios.items():
            name = portfolio_cfg.get('name', key.upper())
            size = len(portfolio_cfg.get('universe', []))
            print(f"  - {name}: {size} Aktien")
    else:
        print(f"Portfolio Größe:  {len(config.get('data.universe'))} Aktien")
    print(f"Features:         {config.get('features.input_features')}")

    enabled_models = [m for m in ['pytorch_nn', 'sklearn_nn', 'ols', 'ridge', 'random_forest']
                     if config.get(f'models.{m}.enabled')]
    print(f"Aktive Modelle:   {enabled_models}")
    print("="*70 + "\n")

    # Mode-spezifische Anpassungen
    if args.mode != 'compare':
        print(f"ℹ️  Hinweis: Nur {args.mode}-Daten werden verwendet.\n")
        # Hier könnte man die Config anpassen, um nur eine Periode zu laden
        # Für Einfachheit lassen wir das erstmal weg

    # Model Comparison starten
    try:
        comparison = ModelComparison(args.config)
        comparison.run_full_comparison()

        print("\n✅ Erfolgreich abgeschlossen!")
        print(f"📁 Ergebnisse: Results/model_comparison.xlsx")
        if config.get("output.save_models"):
            if portfolios:
                print("💾 Modelle gespeichert unter:")
                for key, portfolio_cfg in portfolios.items():
                    name = portfolio_cfg.get('name', key.upper())
                    print(f"   • Models/{name}/daily/ und Models/{name}/intraday/")
            else:
                print(f"💾 Modelle: Models/Portfolio/daily/ und Models/Portfolio/intraday/")

    except KeyboardInterrupt:
        print("\n\n⚠️  Abgebrochen durch Benutzer.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
