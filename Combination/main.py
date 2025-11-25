#!/usr/bin/env python
"""
main.py - Main program for BA_combination Unified Version
Combines the strengths of version 1 and 2

Usage:
    python main.py                          # Standard comparison (both periods)
    python main.py --mode daily             # Only daily data
    python main.py --mode intraday          # Only 30-minute data
    python main.py --features momentum_5 change_dax  # Custom features
    python main.py --config my_config.yaml  # Custom config
"""

import argparse
import logging
from pathlib import Path

from ModelComparison import ModelComparison
from ConfigManager import ConfigManager
from logger_config import setup_logging, get_logger

logger = get_logger(__name__)


def print_banner() -> None:
    """Print an ASCII banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║            BA TRADING SYSTEM - UNIFIED VERSION                       ║
║                                                                      ║
║  Combines the strengths of version 1 and version 2                  ║
║  Portfolio-based machine learning trading system                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)  # Banner is always printed to the console


def main() -> None:
    """Entry point"""
    log_file = Path("Logs") / "main.log" if Path("Logs").exists() else None
    setup_logging(log_level=logging.INFO, log_file=log_file)
    
    parser = argparse.ArgumentParser(
        description='BA Trading System - Unified Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Compare both periods
  python main.py --mode daily             # Only daily data
  python main.py --features momentum_5 change_dax  # Custom features
  python main.py --config my_config.yaml  # Custom config

Features in config:
  - momentum_5, momentum_10, momentum_20  # Momentum indicators
  - change_dax                            # DAX change
  - vdax_absolute                         # VDAX absolute value
  - volume_ratio                          # Volume ratio
  - rsi_14                                # RSI (optional)
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='compare',
        choices=['compare', 'daily', 'intraday'],
        help='Mode: compare (both), daily, or intraday (default: compare)'
    )

    parser.add_argument(
        '--features',
        type=str,
        nargs='+',
        help='Override features from config (e.g. --features momentum_5 change_dax)'
    )

    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        choices=['pytorch_nn', 'sklearn_nn', 'ols', 'ridge', 'random_forest'],
        help='Train only selected models'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save models'
    )

    args = parser.parse_args()

    print_banner()

    config = ConfigManager(args.config)

    if args.features:
        config.set("features.input_features", args.features)
        logger.info(f"Features overridden: {args.features}")
        print(f"Features overridden: {args.features}")

    if args.models:
        for model in ['pytorch_nn', 'sklearn_nn', 'ols', 'ridge', 'random_forest']:
            config.set(f"models.{model}.enabled", False)
        for model in args.models:
            config.set(f"models.{model}.enabled", True)
        logger.info(f"Only these models enabled: {args.models}")
        print(f"Only these models enabled: {args.models}")

    if args.no_save:
        config.set("output.save_models", False)
        logger.info("Model saving disabled")
        print("Model saving disabled")

    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    print(f"Config file:      {args.config}")
    print(f"Mode:             {args.mode}")

    # Calculate portfolio size (all stocks across all portfolios)
    portfolios = config.get('data.portfolios', {})
    total_stocks = sum(len(p.get('universe', [])) for p in portfolios.values())
    print(f"Portfolios:       {', '.join(portfolios.keys())}")
    print(f"Portfolio size:   {total_stocks} stocks")
    print(f"Features:         {config.get('features.input_features')}")

    configured_active_models = config.get("models.active_models", []) or []
    if configured_active_models:
        active_models = [m for m in configured_active_models if m in ['pytorch_nn', 'sklearn_nn', 'ols', 'ridge', 'random_forest']]
    else:
        active_models = [m for m in ['pytorch_nn', 'sklearn_nn', 'ols', 'ridge', 'random_forest']
                         if config.get(f'models.{m}.enabled')]

    print(f"Active models:    {active_models}")
    print("="*70 + "\n")

    if args.mode != 'compare':
        print(f"Note: Only {args.mode} data will be used.\n")

    logger.info("Starting model comparison...")
    comparison = ModelComparison(args.config)
    comparison.run_full_comparison()

    logger.info("Completed successfully!")
    print("\nCompleted successfully!")
    print("Results: Results/model_comparison.xlsx")
    if config.get("output.save_models"):
        print("Models: Models/")


if __name__ == "__main__":
    main()
