"""
ConfigManager.py - Config-System für BA_combination
Lädt und verwaltet die zentrale config.yaml Konfiguration
"""

import yaml
from pathlib import Path
from typing import Any, Dict
from copy import deepcopy

from logger_config import get_logger

logger = get_logger(__name__)

# Constants
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
            "scheduler_patience": 15
        },
        "sklearn_nn": {
            "enabled": True,
            "hidden_layer_sizes": [64, 32],
            "max_iter": 1500
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
    """Verwaltet die Konfiguration aus config.yaml"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialisiert den ConfigManager

        Args:
            config_path: Pfad zur Config-Datei (relativ oder absolut)
        """
        self.path = Path(config_path)
        if not self.path.is_absolute():
            # Wenn relativer Pfad, dann relativ zum Skript-Verzeichnis
            self.path = Path(__file__).parent / self.path

        self.config = self._load_and_validate_config()

    def load_config(self) -> Dict[str, Any]:
        """
        Lädt Config-Datei (Legacy-Methode für Kompatibilität)

        Returns:
            Dictionary mit Konfiguration

        Raises:
            FileNotFoundError: Wenn config.yaml nicht gefunden wird
            yaml.YAMLError: Wenn YAML-Syntax fehlerhaft ist
        """
        return self._load_and_validate_config()

    def _load_and_validate_config(self) -> Dict[str, Any]:
        """
        Lädt Config-Datei und validiert sie

        Returns:
            Dictionary mit Konfiguration (merged mit Defaults)

        Raises:
            FileNotFoundError: Wenn config.yaml nicht gefunden wird
            yaml.YAMLError: Wenn YAML-Syntax fehlerhaft ist
            ValueError: Wenn Config-Validierung fehlschlägt
        """
        if not self.path.exists():
            logger.warning(f"Config-Datei nicht gefunden: {self.path}, verwende Defaults")
            return deepcopy(DEFAULT_CONFIG)

        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Fehler beim Laden der Config-Datei: {e}")

        # Merge mit Defaults
        config = self._deep_merge(deepcopy(DEFAULT_CONFIG), user_config)

        # Validiere Config
        self._validate_config(config)

        logger.info(f"Config erfolgreich geladen: {self.path}")
        return config

    @staticmethod
    def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge von zwei Dictionaries

        Args:
            base: Basis-Dictionary
            update: Update-Dictionary

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
        """
        Validiert Config-Werte

        Args:
            config: Config-Dictionary

        Raises:
            ValueError: Wenn Validierung fehlschlägt
        """
        # Validiere test_split
        test_split = config.get("training", {}).get("test_split", 0.2)
        if not 0 < test_split < 1:
            raise ValueError(f"test_split muss zwischen 0 und 1 sein, ist aber {test_split}")

        # Validiere ffc_runs
        ffc_runs = config.get("training", {}).get("ffc_runs", False)
        if not isinstance(ffc_runs, bool):
            raise ValueError("training.ffc_runs muss boolesch sein (true/false)")

        # Validiere Portfolio-Struktur
        portfolios = config.get("data", {}).get("portfolios", {})
        if not portfolios:
            logger.warning("Keine Portfolios in Config definiert")

        for name, portfolio in portfolios.items():
            if not isinstance(portfolio, dict):
                raise ValueError(f"Portfolio '{name}' muss ein Dictionary sein")
            if "universe" not in portfolio:
                raise ValueError(f"Portfolio '{name}' hat kein 'universe' Feld")
            if not isinstance(portfolio["universe"], list):
                raise ValueError(f"Portfolio '{name}' universe muss eine Liste sein")
            if len(portfolio["universe"]) == 0:
                raise ValueError(f"Portfolio '{name}' universe ist leer")

        # Validiere Features
        input_features = config.get("features", {}).get("input_features", [])
        if not input_features:
            logger.warning("Keine Features in Config definiert")

        # Validiere Models
        models = config.get("models", {})
        active_models = models.get("active_models", [])

        if active_models and not isinstance(active_models, list):
            raise ValueError("models.active_models muss eine Liste sein")

        valid_model_names = {name for name in models.keys() if name != "active_models"}
        valid_model_names.add("naive_baseline")  # Baseline kann optional explizit gesetzt werden

        if active_models:
            invalid_models = [m for m in active_models if m not in valid_model_names]
            if invalid_models:
                raise ValueError(f"Ungültige Modelle in models.active_models: {invalid_models}")

        enabled_models = [
            name for name, cfg in models.items()
            if isinstance(cfg, dict) and cfg.get("enabled", False)
        ]
        if not enabled_models and not active_models:
            logger.warning("Keine Modelle in Config aktiviert")

        selected_models_count = len(active_models) if active_models else len(enabled_models)
        logger.debug(f"Config validiert: {len(portfolios)} Portfolios, {selected_models_count} Modelle aktiviert/selektiert")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Holt Wert aus Config mit Punkt-Notation

        Args:
            key_path: Pfad zum Wert mit Punkten getrennt
                     Beispiel: "models.pytorch_nn.epochs"
            default: Rückgabewert wenn Key nicht existiert

        Returns:
            Wert aus Config oder default

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
        Setzt Wert in Config

        Args:
            key_path: Pfad zum Wert mit Punkten getrennt
            value: Zu setzender Wert

        Examples:
            >>> config.set("models.pytorch_nn.epochs", 300)
            >>> config.set("features.input_features", ["momentum_5", "change_dax"])
        """
        keys = key_path.split('.')
        config = self.config

        # Navigiere bis zum vorletzten Key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Setze den finalen Wert
        config[keys[-1]] = value
        self.save_config()

    def save_config(self) -> None:
        """
        Speichert Config zurück in Datei

        Raises:
            IOError: Wenn Datei nicht geschrieben werden kann
        """
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        except IOError as e:
            raise IOError(f"Fehler beim Speichern der Config-Datei: {e}")

    def reload(self) -> None:
        """Lädt Config-Datei neu"""
        self.config = self._load_and_validate_config()
        logger.info("Config neu geladen")

    def __repr__(self) -> str:
        return f"ConfigManager(path='{self.path}')"


if __name__ == "__main__":
    # Test
    from logger_config import setup_logging
    setup_logging()
    
    config = ConfigManager()
    logger.info("Config erfolgreich geladen!")
    logger.info(f"Universe: {config.get('data.universe')}")
    logger.info(f"PyTorch Epochs: {config.get('models.pytorch_nn.epochs')}")
    logger.info(f"Input Features: {config.get('features.input_features')}")
