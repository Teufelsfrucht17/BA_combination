"""
main.py - Main program for BA_combination Unified Version
Combines the strengths of version 1 and 2

Usage:
    python main.py                          Standard comparison (both periods)
    python main.py --mode daily             Only daily data
    python main.py --mode intraday          Only 30-minute data
    python main.py --features momentum_5 change_dax  Custom features
    python main.py --config my_config.yaml  Custom config
"""

import argparse
import logging
from pathlib import Path

from ModelComparison import ModelComparison
from ConfigManager import ConfigManager
from logger_config import setup_logging


def main() -> None:
    """Entry point"""
    log_file = Path("Logs") / "main.log" if Path("Logs").exists() else None
    setup_logging(log_level=logging.INFO, log_file=log_file)
    
    parser = argparse.ArgumentParser(
        description='BA Trading System - Unified Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          Compare both periods
  python main.py --mode daily             Only daily data
  python main.py --features momentum_5 change_dax  Custom features
  python main.py --config my_config.yaml  Custom config

Features in config:
  - momentum_5, momentum_10, momentum_20  Momentum indicators
  - change_dax                            DAX change
  - vdax_absolute                         VDAX absolute value
  - volume_ratio                          Volume ratio
  - rsi_14                                RSI (optional)
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
        choices=['pytorch_nn', 'ols', 'ridge', 'random_forest'],
        help='Train only selected models'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save models'
    )

    args = parser.parse_args()

    config = ConfigManager(args.config)

    if args.features:
        config.set("features.input_features", args.features)

    if args.models:
        for model in ['pytorch_nn', 'ols', 'ridge', 'random_forest']:
            config.set(f"models.{model}.enabled", False)
        for model in args.models:
            config.set(f"models.{model}.enabled", True)

    if args.no_save:
        config.set("output.save_models", False)

    comparison = ModelComparison(args.config)
    comparison.run_full_comparison()

    return


if __name__ == "__main__":
    main()
