"""
logger_config.py - Zentrale Logging-Konfiguration für BA_combination
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Konfiguriert das Logging-System für die gesamte Anwendung

    Args:
        log_level: Logging-Level (default: INFO)
        log_file: Optionaler Pfad zu Log-Datei
        format_string: Optionaler Format-String für Logs

    Returns:
        Konfigurierter Root Logger
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Basis-Formatierung
    formatter = logging.Formatter(format_string)

    # Root Logger konfigurieren
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Entferne vorhandene Handler
    root_logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # File bekommt alle Logs
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Gibt einen Logger für ein spezifisches Modul zurück

    Args:
        name: Name des Moduls (normalerweise __name__)

    Returns:
        Logger-Instanz
    """
    return logging.getLogger(name)

