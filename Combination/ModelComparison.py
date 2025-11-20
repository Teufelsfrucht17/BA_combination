"""
ModelComparison.py - Hauptmodul für Modellvergleich
Trainiert alle Modelle und erstellt detaillierten Vergleichsbericht
"""

import pandas as pd
import numpy as np
import time
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback: Dummy tqdm
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Import eigener Module
from Datagrabber import DataGrabber
from Dataprep import DataPrep, time_series_split
from ConfigManager import ConfigManager
from Models_Wrapper import (
    train_pytorch_model,
    train_sklearn_nn,
    train_ols,
    train_ridge,
    train_random_forest,
    train_naive_baseline
)
from logger_config import get_logger

logger = get_logger(__name__)


class ModelComparison:
    """Vergleicht alle Machine Learning Modelle für Daily vs Intraday Daten"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialisiert ModelComparison

        Args:
            config_path: Pfad zur Config-Datei
        """
        self.config = ConfigManager(config_path)
        self.results = {}

    def run_full_comparison(self):
        """Führt kompletten Vergleich durch: Alle Portfolios, Daily vs Intraday, alle Modelle"""

        logger.info("="*70)
        logger.info("BA TRADING SYSTEM - PORTFOLIO-BASIERTER MODELLVERGLEICH")
        logger.info("="*70)
        print("\n" + "="*70)
        print("BA TRADING SYSTEM - PORTFOLIO-BASIERTER MODELLVERGLEICH")
        print("="*70)

        # 1. Daten holen (Portfolio-basiert)
        logger.info("[SCHRITT 1/4] DATENABRUF")
        print("\n[SCHRITT 1/4] DATENABRUF")
        grabber = DataGrabber(self.config.path)
        all_data = grabber.fetch_all_data()  # {"dax": {"daily": df, "intraday": df}, "sdax": {...}}

        # 2. Dataprep
        prep = DataPrep(self.config.path)

        # Für jedes Portfolio
        portfolios_iter = tqdm(all_data.items(), desc="Portfolios", leave=True) if TQDM_AVAILABLE else all_data.items()
        for portfolio_name, portfolio_data in portfolios_iter:
            portfolio_config = self.config.get(f"data.portfolios.{portfolio_name}")
            portfolio_display_name = portfolio_config.get("name", portfolio_name.upper())

            # Für beide Zeitperioden
            periods_iter = tqdm(portfolio_data.items(), desc=f"{portfolio_display_name} Perioden", leave=False) if TQDM_AVAILABLE else portfolio_data.items()
            for period_type, data in periods_iter:
                print("\n" + "="*70)
                print(f"TRAINING: {portfolio_display_name} - {period_type.upper()}")
                print("="*70)

                # Prepare data (mit portfolio_name für korrekte Index-Features)
                X, y = prep.prepare_data(data, portfolio_name=portfolio_name, period_type=period_type)

                # Train-Test Split (chronologisch, kein Shuffle!)
                test_split = self.config.get("training.test_split", 0.2)
                X_train, X_test, y_train, y_test = time_series_split(X, y, test_size=test_split)

                # ========================================
                # WICHTIG: Zentrale Skalierung hier!
                # ========================================
                # Scaler wird NUR auf X_train gefittet, dann auf beide Sets angewendet.
                # Dies verhindert Data Leakage (Test-Daten beeinflussen nicht das Training).
                # Alle Modelle erhalten bereits skalierte Daten!
                scaler_method = self.config.get("training.scaling.method", "StandardScaler")
                scaler = MinMaxScaler() if scaler_method == "MinMaxScaler" else StandardScaler()
                scaler.fit(X_train)  # Fit nur auf Trainingsset!

                X_train = pd.DataFrame(
                    scaler.transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index
                )
                X_test = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )

                logger.info(f"Train Size: {len(X_train)} samples, Test Size: {len(X_test)} samples")
                print(f"\nTrain Size: {len(X_train)} samples")
                print(f"Test Size: {len(X_test)} samples")

                # Trainiere alle Modelle
                results_key = f"{portfolio_name}_{period_type}"
                self.results[results_key] = self.train_all_models(
                    X_train, X_test, y_train, y_test, portfolio_name, period_type
                )

        # 3. Vergleich erstellen
        logger.info("[SCHRITT 4/4] ERSTELLE VERGLEICHSBERICHT")
        print("\n[SCHRITT 4/4] ERSTELLE VERGLEICHSBERICHT")
        self.create_comparison_report()

    def train_all_models(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train: pd.Series, 
        y_test: pd.Series, 
        portfolio_name: str, 
        period_type: str
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Trainiert alle aktivierten Modelle

        Args:
            X_train, X_test, y_train, y_test: Train/Test Splits
            portfolio_name: Name des Portfolios (z.B. "dax", "sdax")
            period_type: "daily" oder "intraday"

        Returns:
            Dictionary mit Ergebnissen aller Modelle
        """
        results = {}

        portfolio_config = self.config.get(f"data.portfolios.{portfolio_name}")
        portfolio_display = portfolio_config.get("name", portfolio_name.upper())

        logger.info(f"[SCHRITT 2/4] FEATURE ENGINEERING ABGESCHLOSSEN")
        logger.info(f"[SCHRITT 3/4] MODELL-TRAINING ({portfolio_display} - {period_type.upper()})")
        print(f"\n[SCHRITT 2/4] FEATURE ENGINEERING ABGESCHLOSSEN")
        print(f"[SCHRITT 3/4] MODELL-TRAINING ({portfolio_display} - {period_type.upper()})")

        # Definiere Modell-Konfigurationen
        model_configs = {
            "naive_baseline": {
                "enabled": True,  # Immer aktiviert
                "train_func": train_naive_baseline,
                "display_name": "Baseline Model (Naive Predictor)",
                "get_kwargs": lambda: {},
                "extra_info": "Baseline dient als Vergleichsmaßstab (sollte von ML-Modellen übertroffen werden)"
            },
            "pytorch_nn": {
                "enabled": self.config.get("models.pytorch_nn.enabled", False),
                "train_func": train_pytorch_model,
                "display_name": "PyTorch Neural Network",
                "get_kwargs": lambda: {
                    "hidden1": self.config.get("models.pytorch_nn.hidden1", 64),
                    "hidden2": self.config.get("models.pytorch_nn.hidden2", 32),
                    "epochs": self.config.get("models.pytorch_nn.epochs", 200),
                    "batch_size": self.config.get("models.pytorch_nn.batch_size", 64),
                    "lr": self.config.get("models.pytorch_nn.learning_rate", 0.001),
                    "validation_split": self.config.get("models.pytorch_nn.validation_split", 0.2),
                    "early_stopping_patience": self.config.get("models.pytorch_nn.early_stopping_patience", 20),
                    "use_scheduler": True,
                    "scheduler_patience": self.config.get("models.pytorch_nn.scheduler_patience", 10),
                    "weight_decay": self.config.get("models.pytorch_nn.weight_decay", 0.0),
                    "portfolio_name": portfolio_name,
                    "period_type": period_type
                }
            },
            "sklearn_nn": {
                "enabled": self.config.get("models.sklearn_nn.enabled", False),
                "train_func": train_sklearn_nn,
                "display_name": "Sklearn Neural Network",
                "get_kwargs": lambda: {
                    "hidden_layer_sizes": tuple(self.config.get("models.sklearn_nn.hidden_layer_sizes", [64, 32])),
                    "max_iter": self.config.get("models.sklearn_nn.max_iter", 500),
                    "n_splits": self.config.get("training.cross_validation.n_splits", 5),
                    "use_gridsearch": self.config.get("training.cross_validation.enabled", True)
                }
            },
            "ols": {
                "enabled": self.config.get("models.ols.enabled", False),
                "train_func": train_ols,
                "display_name": "OLS Linear Regression",
                "get_kwargs": lambda: {}
            },
            "ridge": {
                "enabled": self.config.get("models.ridge.enabled", False),
                "train_func": train_ridge,
                "display_name": "Ridge Regression",
                "get_kwargs": lambda: {
                    "alpha_values": self.config.get("models.ridge.alpha_values", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
                }
            },
            "random_forest": {
                "enabled": self.config.get("models.random_forest.enabled", False),
                "train_func": train_random_forest,
                "display_name": "Random Forest",
                "get_kwargs": lambda: {
                    "n_estimators": self.config.get("models.random_forest.n_estimators", 300),
                    "max_depth": self.config.get("models.random_forest.max_depth", 10),
                    "min_samples_split": self.config.get("models.random_forest.min_samples_split", 5),
                    "n_splits": self.config.get("training.cross_validation.n_splits", 5),
                    "use_gridsearch": self.config.get("training.cross_validation.enabled", True)
                }
            }
        }

        # Trainiere alle Modelle
        for model_name, config in model_configs.items():
            if not config["enabled"]:
                continue

            self._train_single_model(
                model_name=model_name,
                config=config,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                results=results
            )

        return results

    def _train_single_model(
        self,
        model_name: str,
        config: Dict[str, Any],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        results: Dict[str, Optional[Dict[str, Any]]]
    ) -> None:
        """
        Trainiert ein einzelnes Modell (Helper-Methode zur Reduzierung von Code-Duplikation)

        Args:
            model_name: Name des Modells
            config: Modell-Konfiguration (enabled, train_func, display_name, get_kwargs, extra_info)
            X_train, X_test, y_train, y_test: Train/Test Splits
            results: Dictionary in das Ergebnisse geschrieben werden
        """
        print(f"\n{'─'*60}")
        print(config["display_name"])
        print(f"{'─'*60}")

        start = time.time()

        try:
            # Hole kwargs
            kwargs = config.get("get_kwargs", lambda: {})()

            # Trainiere Modell
            model, metrics = config["train_func"](X_train, y_train, X_test, y_test, **kwargs)

            training_time = time.time() - start

            results[model_name] = {
                "model": model,
                "metrics": metrics,
                "training_time": training_time
            }

            # Zeige Ergebnisse
            self._print_model_results(metrics, training_time, model_name, config.get("extra_info"))

        except (RuntimeError, ValueError) as e:
            logger.error(f"Fehler beim Training von {model_name}: {e}", exc_info=True)
            print(f"  ✗ Fehler: {e}")
            results[model_name] = None
        except Exception as e:
            logger.critical(f"Unerwarteter Fehler beim Training von {model_name}: {e}", exc_info=True)
            print(f"  ✗ Unerwarteter Fehler: {e}")
            results[model_name] = None

    def _print_model_results(
        self,
        metrics: Dict[str, Any],
        training_time: float,
        model_name: str,
        extra_info: Optional[str] = None
    ) -> None:
        """
        Druckt Modell-Ergebnisse einheitlich

        Args:
            metrics: Dictionary mit Metriken
            training_time: Trainingszeit in Sekunden
            model_name: Name des Modells
            extra_info: Optional zusätzliche Information
        """
        print(f"  ✓ R² Test: {metrics['r2']:.4f}")
        print(f"  ✓ MSE: {metrics['mse']:.6f}")
        print(f"  ✓ MAE: {metrics['mae']:.6f}")
        
        # Zusätzliche Metriken falls vorhanden
        if 'best_alpha' in metrics:
            print(f"  ✓ Best Alpha: {metrics['best_alpha']}")
        
        print(f"  ✓ Training Zeit: {training_time:.2f}s")
        
        if extra_info:
            print(f"  ℹ️  {extra_info}")

        logger.info(
            f"{model_name}: R²={metrics['r2']:.4f}, MSE={metrics['mse']:.6f}, "
            f"MAE={metrics['mae']:.6f}, Time={training_time:.2f}s"
        )

    def create_comparison_report(self):
        """Erstellt detaillierten Vergleichsbericht als Excel (Portfolio-basiert)"""

        # Sammle alle Metriken
        comparison_data = []

        for results_key, models in self.results.items():
            # Parse results_key: "dax_daily" -> portfolio="dax", period="daily"
            if "_" in results_key:
                portfolio_name, period = results_key.rsplit("_", 1)
            else:
                portfolio_name, period = "unknown", results_key

            # Hole Portfolio-Anzeigenamen
            portfolio_config = self.config.get(f"data.portfolios.{portfolio_name}")
            if portfolio_config:
                portfolio_display = portfolio_config.get("name", portfolio_name.upper())
            else:
                portfolio_display = portfolio_name.upper()

            for model_name, model_results in models.items():
                if model_results is None:
                    continue

                comparison_data.append({
                    "Portfolio": portfolio_display,
                    "Period": period,
                    "Model": model_name,
                    "R2_Test": model_results["metrics"]["r2"],
                    "R2_Train": model_results["metrics"].get("train_r2", np.nan),
                    "MSE": model_results["metrics"]["mse"],
                    "MAE": model_results["metrics"]["mae"],
                    "Directional_Accuracy": model_results["metrics"].get("directional_accuracy", np.nan),
                    "Directional_Accuracy_Train": model_results["metrics"].get("directional_accuracy_train", np.nan),
                    "Training_Time_s": model_results["training_time"]
                })

        # Erstelle DataFrame
        df_comparison = pd.DataFrame(comparison_data)

        if df_comparison.empty:
            logger.warning("Keine Ergebnisse zum Vergleichen!")
            print("⚠️ Keine Ergebnisse zum Vergleichen!")
            return

        # Pivot für bessere Übersicht
        # Multi-index Pivot: Portfolio+Period als Spalten
        df_comparison['Portfolio_Period'] = df_comparison['Portfolio'] + "_" + df_comparison['Period']
        pivot_r2 = df_comparison.pivot(index='Model', columns='Portfolio_Period', values='R2_Test')
        pivot_mse = df_comparison.pivot(index='Model', columns='Portfolio_Period', values='MSE')

        # Zusätzlich: Portfolio-spezifische Pivots
        pivot_portfolio = df_comparison.pivot_table(
            index='Model',
            columns=['Portfolio', 'Period'],
            values='R2_Test'
        )

        # Speichere als Excel
        output_path = Path("Results") / "model_comparison.xlsx"
        output_path.parent.mkdir(exist_ok=True)

        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            df_comparison.to_excel(writer, sheet_name='Full_Comparison', index=False)
            pivot_r2.to_excel(writer, sheet_name='R2_by_Portfolio_Period')
            pivot_mse.to_excel(writer, sheet_name='MSE_by_Portfolio_Period')
            pivot_portfolio.to_excel(writer, sheet_name='R2_Hierarchical')

        print("\n" + "="*70)
        print("VERGLEICH ABGESCHLOSSEN")
        print("="*70)
        print("\n📊 Beste Modelle nach R² Score:")
        print("─"*70)

        # Zeige beste Modelle pro Portfolio und Period
        for portfolio in df_comparison['Portfolio'].unique():
            for period in df_comparison['Period'].unique():
                subset = df_comparison[
                    (df_comparison['Portfolio'] == portfolio) &
                    (df_comparison['Period'] == period)
                ]
                if subset.empty:
                    continue

                best_idx = subset['R2_Test'].idxmax()
                best_model = subset.loc[best_idx]

                print(f"\n{portfolio} - {period.upper()}:")
                print(f"  🏆 Bestes Modell: {best_model['Model']}")
                print(f"  📈 R² Test Score: {best_model['R2_Test']:.4f}")
                print(f"  📉 MSE: {best_model['MSE']:.6f}")
                print(f"  ⏱️  Training Zeit: {best_model['Training_Time_s']:.2f}s")

        # Speichere Modelle
        if self.config.get("output.save_models"):
            logger.info("Speichere Modelle...")
            print("\n💾 Speichere Modelle...")
            self.save_models()

        logger.info(f"Ergebnisse gespeichert: {output_path}")
        print(f"\n✅ Ergebnisse gespeichert: {output_path}")
        print("="*70 + "\n")

    def save_models(self):
        """Speichert alle trainierten Modelle (Portfolio-basiert)"""
        models_path = Path("Models")
        models_path.mkdir(exist_ok=True)

        for results_key, models in self.results.items():
            # Parse results_key: "dax_daily" -> portfolio="dax", period="daily"
            if "_" in results_key:
                portfolio_name, period = results_key.rsplit("_", 1)
            else:
                portfolio_name, period = "unknown", results_key

            # Erstelle Unterordner: Models/dax_daily/
            portfolio_period_path = models_path / results_key
            portfolio_period_path.mkdir(exist_ok=True)

            for model_name, model_results in models.items():
                if model_results is None or model_results["model"] is None:
                    continue

                model_file = portfolio_period_path / f"{model_name}.pkl"

                try:
                    # Speichere je nach Modelltyp
                    if model_name == "pytorch_nn":
                        # PyTorch speichern
                        import torch
                        torch.save(model_results["model"], portfolio_period_path / f"{model_name}.pt")
                    else:
                        # Sklearn Modelle speichern
                        joblib.dump(model_results["model"], model_file)

                    print(f"  ✓ {results_key}/{model_name} gespeichert")

                except Exception as e:
                    logger.error(f"Fehler beim Speichern von {results_key}/{model_name}: {e}", exc_info=True)
                    print(f"  ✗ Fehler beim Speichern von {results_key}/{model_name}: {e}")


if __name__ == "__main__":
    # Test
    comparison = ModelComparison()
    comparison.run_full_comparison()
