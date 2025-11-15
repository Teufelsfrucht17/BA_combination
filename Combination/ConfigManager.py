"""
ConfigManager.py - Config-System für BA_combination
Lädt und verwaltet die zentrale config.yaml Konfiguration
"""

import yaml
from pathlib import Path
from typing import Any


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

        self.config = self.load_config()

    def load_config(self) -> dict:
        """
        Lädt Config-Datei

        Returns:
            Dictionary mit Konfiguration

        Raises:
            FileNotFoundError: Wenn config.yaml nicht gefunden wird
            yaml.YAMLError: Wenn YAML-Syntax fehlerhaft ist
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Config-Datei nicht gefunden: {self.path}")

        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Fehler beim Laden der Config-Datei: {e}")

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
        self.config = self.load_config()

    def __repr__(self) -> str:
        return f"ConfigManager(path='{self.path}')"


if __name__ == "__main__":
    # Test
    config = ConfigManager()
    print("Config erfolgreich geladen!")
    print(f"Universe: {config.get('data.universe')}")
    print(f"PyTorch Epochs: {config.get('models.pytorch_nn.epochs')}")
    print(f"Input Features: {config.get('features.input_features')}")
