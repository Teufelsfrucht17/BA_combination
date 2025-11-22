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
import logging
from pathlib import Path
import warnings
from typing import NoReturn

warnings.filterwarnings('ignore')

from ModelComparison import ModelComparison
from ConfigManager import ConfigManager
from logger_config import setup_logging, get_logger

logger = get_logger(__name__)


def print_banner() -> None:
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
    print(banner)  # Banner wird immer auf Console ausgegeben


def main() -> None:
    """Hauptfunktion"""
    # Logging Setup
    log_file = Path("Logs") / "main.log" if Path("Logs").exists() else None
    setup_logging(log_level=logging.INFO, log_file=log_file)
    
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
    except FileNotFoundError as e:
        logger.error(f"Config-Datei nicht gefunden: {args.config}", exc_info=True)
        print(f"❌ Fehler: Config-Datei nicht gefunden: {args.config}")
        print(f"💡 Tipp: Stelle sicher, dass config.yaml im aktuellen Verzeichnis ist.")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Fehler beim Laden der Config: {e}", exc_info=True)
        print(f"❌ Fehler beim Laden der Config: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unerwarteter Fehler beim Laden der Config: {e}", exc_info=True)
        print(f"❌ Unerwarteter Fehler beim Laden der Config: {e}")
        sys.exit(1)

    # Features überschreiben falls angegeben
    if args.features:
        config.set("features.input_features", args.features)
        logger.info(f"Features überschrieben: {args.features}")
        print(f"✓ Features überschrieben: {args.features}")

    # Modelle filtern falls angegeben
    if args.models:
        # Deaktiviere alle Modelle
        for model in ['pytorch_nn', 'sklearn_nn', 'ols', 'ridge', 'random_forest']:
            config.set(f"models.{model}.enabled", False)
        # Aktiviere nur gewählte
        for model in args.models:
            config.set(f"models.{model}.enabled", True)
        logger.info(f"Nur folgende Modelle aktiviert: {args.models}")
        print(f"✓ Nur folgende Modelle aktiviert: {args.models}")

    # Speichern deaktivieren falls gewünscht
    if args.no_save:
        config.set("output.save_models", False)
        logger.info("Modell-Speicherung deaktiviert")
        print(f"✓ Modell-Speicherung deaktiviert")

    # Zeige Konfiguration
    print("\n" + "="*70)
    print("KONFIGURATION")
    print("="*70)
    print(f"Config-Datei:     {args.config}")
    print(f"Modus:            {args.mode}")

    # Berechne Portfolio-Größe (alle Aktien aus allen Portfolios)
    portfolios = config.get('data.portfolios', {})
    total_stocks = sum(len(p.get('universe', [])) for p in portfolios.values())
    print(f"Portfolios:       {', '.join(portfolios.keys())}")
    print(f"Portfolio Größe:  {total_stocks} Aktien")
    print(f"Features:         {config.get('features.input_features')}")

    configured_active_models = config.get("models.active_models", []) or []
    if configured_active_models:
        active_models = [m for m in configured_active_models if m in ['pytorch_nn', 'sklearn_nn', 'ols', 'ridge', 'random_forest']]
    else:
        active_models = [m for m in ['pytorch_nn', 'sklearn_nn', 'ols', 'ridge', 'random_forest']
                         if config.get(f'models.{m}.enabled')]

    print(f"Aktive Modelle:   {active_models}")
    print("="*70 + "\n")

    # Mode-spezifische Anpassungen
    if args.mode != 'compare':
        print(f"ℹ️  Hinweis: Nur {args.mode}-Daten werden verwendet.\n")
        # Hier könnte man die Config anpassen, um nur eine Periode zu laden
        # Für Einfachheit lassen wir das erstmal weg

    # Model Comparison starten
    try:
        logger.info("Starte Modellvergleich...")
        comparison = ModelComparison(args.config)
        comparison.run_full_comparison()

        logger.info("Erfolgreich abgeschlossen!")
        print("\n✅ Erfolgreich abgeschlossen!")
        print(f"📁 Ergebnisse: Results/model_comparison.xlsx")
        if config.get("output.save_models"):
            print(f"💾 Modelle: Models/")

    except KeyboardInterrupt:
        logger.warning("Abgebrochen durch Benutzer")
        print("\n\n⚠️  Abgebrochen durch Benutzer.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unerwarteter Fehler: {e}", exc_info=True)
        print(f"\n\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
