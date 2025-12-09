"""
ConfigManager.py - Config system for BA_combination
Loads and manages the central config.yaml configuration
"""

import yaml
import datetime
from pathlib import Path
from typing import Any, Dict
from copy import deepcopy

DEFAULT_CONFIG = {
    "data": {
        "portfolios": {},
        "common_indices": [],
        "fields": ["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1", "ACVOL_1"],
        "periods": {
            "daily": {
                "interval": "daily",
                "start": "2024-01-01",
                "end": "2025-11-15"
            },
            "intraday": {
                "interval": "30min",
                "start": "2024-01-01",
                "end": "2025-11-15"
            }
        }
    },
    "features": {
        "input_features": ["momentum_5", "momentum_10", "momentum_20"],
        "target": "price_change_next",
        "momentum_periods": [5, 10, 20],
        "rolling_window": 20,
        "volatility_windows": [10, 20]
    },
    "models": {
        "active_models": [],
        "pytorch_nn": {
            "enabled": True,
            "hidden1": 128,
            "hidden2": 64,
            "epochs": 400,
            "batch_size": 64,
            "learning_rate": 0.0005,
            "validation_split": 0.2,
            "weight_decay": 0.0005,
            "early_stopping_patience": 40,
            "scheduler_patience": 15,
            "visualize_model": False,
            "optuna": {
                "enabled": False,
                "n_trials": None,
                "param_grid": {
                    "hidden1": [64, 128],
                    "hidden2": [32, 64],
                    "batch_size": [32, 64],
                    "lr": [0.001, 0.0005],
                    "weight_decay": [0.0, 0.0005]
                }
            }
        },
        "ols": {
            "enabled": True
        },
        "ridge": {
            "enabled": True,
            "alpha_values": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        },
        "random_forest": {
            "enabled": False,
            "n_estimators": 300,
            "max_depth": 10,
            "min_samples_split": 5
        }
    },
    "training": {
        "ffc_runs": False,
        "test_split": 0.2,
        "cross_validation": {
            "enabled": True,
            "n_splits": 5,
            "type": "TimeSeriesSplit"
        },
        "scaling": {
            "method": "StandardScaler"
        }
    },
    "output": {
        "save_models": True,
        "save_predictions": True,
        "save_comparison": True,
        "format": "excel"
    }
}


class ConfigManager:
    """Manage configuration stored in config.yaml"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the ConfigManager

        Args:
            config_path: Path to the config file (relative or absolute)
        """
        self.path = Path(config_path)
        if not self.path.is_absolute():
            self.path = Path(__file__).parent / self.path

        self.config = self._load_and_validate_config()

    def load_config(self) -> Dict[str, Any]:
        """
        Load config (legacy helper for compatibility)

        Returns:
            Dictionary with configuration
        """
        return self._load_and_validate_config()

    def _load_and_validate_config(self) -> Dict[str, Any]:
        """
        Load the config file and validate it

        Returns:
            Dictionary with configuration (merged with defaults)
        """
        with open(self.path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f) or {}

        config = self._deep_merge(deepcopy(DEFAULT_CONFIG), user_config)

        today = datetime.date.today()
        one_year_ago = today - datetime.timedelta(days=365)
        start_str = one_year_ago.strftime("%Y-%m-%d")
        end_str = today.strftime("%Y-%m-%d")

        periods = config.get("data", {}).get("periods", {})
        for key in ("daily", "intraday"):
            if key in periods:
                periods[key]["start"] = start_str
                periods[key]["end"] = end_str

        self._validate_config(config)
        return config

    @staticmethod
    def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge of two dictionaries

        Args:
            base: Base dictionary
            update: Update dictionary

        Returns:
            Merged Dictionary
        """
        result = deepcopy(base)
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Placeholder validation (no-op in best-case mode)."""
        return
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Retrieve a value from config using dot notation

        Args:
            key_path: Dotted path to the value (e.g. "models.pytorch_nn.epochs")
            default: Value to return if the key does not exist

        Returns:
            Value from config or default

        Examples:
            >>> config.get("models.pytorch_nn.epochs")
            200
            >>> config.get("data.universe")
            ['SAP.DE', 'SIE.DE', ...]
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any) -> None:
        """
        Set a value in config

        Args:
            key_path: Dotted path to the value
            value: Value to set

        Examples:
            >>> config.set("models.pytorch_nn.epochs", 300)
            >>> config.set("features.input_features", ["momentum_5", "change_dax"])
        """
        keys = key_path.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value
        self.save_config()

    def save_config(self) -> None:
        """
        Persist config to disk
        """
        with open(self.path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

    def reload(self) -> None:
        """Reload config file"""
        self.config = self._load_and_validate_config()
        return

    def __repr__(self) -> str:
        return f"ConfigManager(path='{self.path}')"


if __name__ == "__main__":
    config = ConfigManager()
    print("Config loaded.")
