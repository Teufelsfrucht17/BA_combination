"""
ModelComparison.py - Hauptmodul für Modellvergleich
Trainiert alle Modelle und erstellt detaillierten Vergleichsbericht
"""

from typing import Dict

import pandas as pd
import numpy as np
import time
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Import eigener Module
from Datagrabber import DataGrabber
from Dataprep import DataPrep, time_series_split
from ConfigManager import ConfigManager
from Models_Wrapper import (
    train_pytorch_model,
    train_sklearn_nn,
    train_ols,
    train_ridge,
    train_random_forest
)


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
        self.portfolios = self._get_portfolio_configs()

    def run_full_comparison(self):
        """Führt kompletten Vergleich durch: Daily vs 30min, alle Modelle"""

        print("\n" + "="*70)
        print("BA TRADING SYSTEM - MODEL COMPARISON")
        print("="*70)

        # 1. Daten holen
        print("\n[SCHRITT 1/4] DATENABRUF")
        grabber = DataGrabber(self.config.path)
        data_by_portfolio = grabber.fetch_all_data()

        # 2. Dataprep
        prep = DataPrep(self.config.path)

        for portfolio_key, period_dict in data_by_portfolio.items():
            portfolio_name = self.portfolios.get(portfolio_key, {}).get("name", portfolio_key.upper())
            print("\n" + "#" * 70)
            print(f"PORTFOLIO TRAINING: {portfolio_name}")
            print("#" * 70)

            for period_type in ["daily", "intraday"]:
                data = period_dict.get(period_type)
                if data is None or data.empty:
                    print(f"⚠️ Keine {period_type}-Daten für {portfolio_name} gefunden – überspringe.")
                    continue

                print("\n" + "=" * 70)
                print(f"TRAINING MIT {period_type.upper()} DATEN ({portfolio_name})")
                print("=" * 70)

                # Prepare data
                X, y = prep.prepare_data(data, period_type)

                # Train-Test Split (chronologisch, kein Shuffle!)
                test_split = self.config.get("training.test_split", 0.2)
                X_train, X_test, y_train, y_test = time_series_split(X, y, test_size=test_split)

                X_train_unscaled = X_train.copy()
                X_test_unscaled = X_test.copy()

                # Skaliere nur auf Basis des Trainingssets, um Data Leakage zu vermeiden
                scaler_method = self.config.get("training.scaling.method", "StandardScaler")
                scaler = None
                if scaler_method:
                    scaler = MinMaxScaler() if scaler_method == "MinMaxScaler" else StandardScaler()
                    scaler.fit(X_train_unscaled)
                    print(f"Skalierung: {scaler.__class__.__name__} (fit nur auf Train-Set)")

                    X_train = pd.DataFrame(
                        scaler.transform(X_train_unscaled),
                        columns=X_train.columns,
                        index=X_train.index
                    )
                    X_test = pd.DataFrame(
                        scaler.transform(X_test_unscaled),
                        columns=X_test.columns,
                        index=X_test.index
                    )
                else:
                    X_train = X_train_unscaled
                    X_test = X_test_unscaled

                print(f"\nTrain Size: {len(X_train)} samples")
                print(f"Test Size: {len(X_test)} samples")

                portfolio_results = self.train_all_models(
                    X_train, X_test, y_train, y_test, period_type,
                    portfolio_name=portfolio_name,
                    X_train_unscaled=X_train_unscaled,
                    X_test_unscaled=X_test_unscaled
                )

                self.results.setdefault(portfolio_key, {})[period_type] = portfolio_results

        # 3. Vergleich erstellen
        print("\n[SCHRITT 4/4] ERSTELLE VERGLEICHSBERICHT")
        self.create_comparison_report()

    def train_all_models(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        period_type: str,
        portfolio_name: str = "Portfolio",
        X_train_unscaled=None,
        X_test_unscaled=None
    ) -> dict:
        """Trainiert alle aktivierten Modelle und liefert die Metriken."""

        results = {}
        X_train_unscaled = X_train_unscaled if X_train_unscaled is not None else X_train
        X_test_unscaled = X_test_unscaled if X_test_unscaled is not None else X_test
        cv_config = self.config.get("training.cross_validation", {})
        cv_enabled = cv_config.get("enabled", False)
        cv_splits = cv_config.get("n_splits", 5)

        print(f"\n[SCHRITT 2/4] FEATURE ENGINEERING ABGESCHLOSSEN ({portfolio_name})")
        print(f"[SCHRITT 3/4] MODELL-TRAINING ({period_type.upper()} – {portfolio_name})")

        # =============================================
        # Naive Baseline
        # =============================================
        print(f"\n{'─'*60}")
        print("Naive Baseline (lag-1)")
        print(f"{'─'*60}")
        baseline_metrics = self.compute_naive_baseline(y_train, y_test)
        results["naive_baseline"] = {
            "model": None,
            "metrics": baseline_metrics,
            "training_time": 0.0
        }
        print(f"  ✓ R² Test: {baseline_metrics['r2']:.4f}")
        print(f"  ✓ MSE: {baseline_metrics['mse']:.6f}")
        print(f"  ✓ MAE: {baseline_metrics['mae']:.6f}")

        # =============================================
        # PyTorch Neural Network
        # =============================================
        if self.config.get("models.pytorch_nn.enabled"):
            print(f"\n{'─'*60}")
            print("PyTorch Neural Network")
            print(f"{'─'*60}")

            start = time.time()

            try:
                model, metrics = train_pytorch_model(
                    X_train, y_train, X_test, y_test,
                    hidden1=self.config.get("models.pytorch_nn.hidden1", 64),
                    hidden2=self.config.get("models.pytorch_nn.hidden2", 32),
                    epochs=self.config.get("models.pytorch_nn.epochs", 200),
                    batch_size=self.config.get("models.pytorch_nn.batch_size", 64),
                    lr=self.config.get("models.pytorch_nn.learning_rate", 0.001),
                    validation_split=self.config.get("models.pytorch_nn.validation_split", 0.2),
                    patience=self.config.get("models.pytorch_nn.patience", 25),
                    min_delta=self.config.get("models.pytorch_nn.early_stopping_min_delta", 1e-4)
                )

                training_time = time.time() - start

                results["pytorch_nn"] = {
                    "model": model,
                    "metrics": metrics,
                    "training_time": training_time
                }

                print(f"  ✓ R² Test: {metrics['r2']:.4f}")
                print(f"  ✓ MSE: {metrics['mse']:.6f}")
                print(f"  ✓ MAE: {metrics['mae']:.6f}")
                if not np.isnan(metrics.get('best_val_loss', np.nan)):
                    print(f"  ✓ Bestes Val-Loss: {metrics['best_val_loss']:.6f}")
                print(f"  ✓ Trainierte Epochen: {metrics.get('trained_epochs')}")
                print(f"  ✓ Training Zeit: {training_time:.2f}s")

            except Exception as e:
                print(f"  ✗ Fehler: {e}")
                results["pytorch_nn"] = None

        # =============================================
        # Sklearn Neural Network
        # =============================================
        if self.config.get("models.sklearn_nn.enabled"):
            print(f"\n{'─'*60}")
            print("Sklearn MLP Regressor")
            print(f"{'─'*60}")

            start = time.time()

            try:
                model, metrics = train_sklearn_nn(
                    X_train_unscaled, y_train, X_test_unscaled, y_test,
                    hidden_layer_sizes=tuple(self.config.get("models.sklearn_nn.hidden_layer_sizes", [64, 32])),
                    max_iter=self.config.get("models.sklearn_nn.max_iter", 500),
                    use_cv=self.config.get("models.sklearn_nn.hyperparameter_search", False) and cv_enabled,
                    param_grid=self.config.get("models.sklearn_nn.hyperparameter_grid"),
                    cv_splits=cv_splits
                )

                training_time = time.time() - start

                results["sklearn_nn"] = {
                    "model": model,
                    "metrics": metrics,
                    "training_time": training_time
                }

                print(f"  ✓ R² Test: {metrics['r2']:.4f}")
                print(f"  ✓ MSE: {metrics['mse']:.6f}")
                print(f"  ✓ MAE: {metrics['mae']:.6f}")
                if metrics.get('best_params'):
                    print(f"  ✓ Beste Hyperparameter: {metrics['best_params']}")
                print(f"  ✓ Training Zeit: {training_time:.2f}s")

            except Exception as e:
                print(f"  ✗ Fehler: {e}")
                results["sklearn_nn"] = None

        # =============================================
        # OLS
        # =============================================
        if self.config.get("models.ols.enabled"):
            print(f"\n{'─'*60}")
            print("OLS Linear Regression")
            print(f"{'─'*60}")

            start = time.time()

            try:
                model, metrics = train_ols(X_train, y_train, X_test, y_test)

                training_time = time.time() - start

                results["ols"] = {
                    "model": model,
                    "metrics": metrics,
                    "training_time": training_time
                }

                print(f"  ✓ R² Test: {metrics['r2']:.4f}")
                print(f"  ✓ MSE: {metrics['mse']:.6f}")
                print(f"  ✓ MAE: {metrics['mae']:.6f}")
                print(f"  ✓ Training Zeit: {training_time:.2f}s")

            except Exception as e:
                print(f"  ✗ Fehler: {e}")
                results["ols"] = None

        # =============================================
        # Ridge Regression
        # =============================================
        if self.config.get("models.ridge.enabled"):
            print(f"\n{'─'*60}")
            print("Ridge Regression")
            print(f"{'─'*60}")

            start = time.time()

            try:
                model, metrics = train_ridge(
                    X_train, y_train, X_test, y_test,
                    alpha_values=self.config.get("models.ridge.alpha_values", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),
                    cv_splits=cv_splits
                )

                training_time = time.time() - start

                results["ridge"] = {
                    "model": model,
                    "metrics": metrics,
                    "training_time": training_time
                }

                print(f"  ✓ R² Test: {metrics['r2']:.4f}")
                print(f"  ✓ MSE: {metrics['mse']:.6f}")
                print(f"  ✓ MAE: {metrics['mae']:.6f}")
                print(f"  ✓ Best Alpha: {metrics.get('best_alpha', 'N/A')}")
                print(f"  ✓ Training Zeit: {training_time:.2f}s")

            except Exception as e:
                print(f"  ✗ Fehler: {e}")
                results["ridge"] = None

        # =============================================
        # Random Forest
        # =============================================
        if self.config.get("models.random_forest.enabled"):
            print(f"\n{'─'*60}")
            print("Random Forest")
            print(f"{'─'*60}")

            start = time.time()

            try:
                default_params = {
                    'n_estimators': self.config.get("models.random_forest.n_estimators", 300),
                    'max_depth': self.config.get("models.random_forest.max_depth", 10),
                    'min_samples_split': self.config.get("models.random_forest.min_samples_split", 5),
                    'min_samples_leaf': self.config.get("models.random_forest.min_samples_leaf", 1),
                    'max_features': self.config.get("models.random_forest.max_features", 'sqrt')
                }

                model, metrics = train_random_forest(
                    X_train, y_train, X_test, y_test,
                    default_params=default_params,
                    param_grid=self.config.get("models.random_forest.param_grid"),
                    use_cv=self.config.get("models.random_forest.hyperparameter_search", False) and cv_enabled,
                    cv_splits=cv_splits
                )

                training_time = time.time() - start

                results["random_forest"] = {
                    "model": model,
                    "metrics": metrics,
                    "training_time": training_time
                }

                print(f"  ✓ R² Test: {metrics['r2']:.4f}")
                print(f"  ✓ MSE: {metrics['mse']:.6f}")
                print(f"  ✓ MAE: {metrics['mae']:.6f}")
                if metrics.get('best_params'):
                    print(f"  ✓ Beste Hyperparameter: {metrics['best_params']}")
                print(f"  ✓ Training Zeit: {training_time:.2f}s")

            except Exception as e:
                print(f"  ✗ Fehler: {e}")
                results["random_forest"] = None

        return results

    def create_comparison_report(self):
        """Erstellt detaillierten Vergleichsbericht als Excel"""

        # Sammle alle Metriken
        comparison_data = []

        for portfolio_key, period_dict in self.results.items():
            portfolio_name = self.portfolios.get(portfolio_key, {}).get("name", portfolio_key.upper())
            for period, models in period_dict.items():
                if models is None:
                    continue

                for model_name, model_results in models.items():
                    if model_results is None:
                        continue

                    comparison_data.append({
                        "Portfolio": portfolio_name,
                        "Period": period,
                        "Model": model_name,
                        "R2_Test": model_results["metrics"]["r2"],
                        "R2_Train": model_results["metrics"].get("train_r2", np.nan),
                        "MSE": model_results["metrics"]["mse"],
                        "MAE": model_results["metrics"]["mae"],
                        "Training_Time_s": model_results["training_time"]
                    })

        # Erstelle DataFrame
        df_comparison = pd.DataFrame(comparison_data)

        if df_comparison.empty:
            print("⚠️ Keine Ergebnisse zum Vergleichen!")
            return

        # Pivot für bessere Übersicht
        pivot_r2 = df_comparison.pivot_table(index=['Portfolio', 'Model'], columns='Period', values='R2_Test')
        pivot_mse = df_comparison.pivot_table(index=['Portfolio', 'Model'], columns='Period', values='MSE')

        # Speichere als Excel
        output_path = Path("Results") / "model_comparison.xlsx"
        output_path.parent.mkdir(exist_ok=True)

        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            df_comparison.to_excel(writer, sheet_name='Full_Comparison', index=False)
            pivot_r2.to_excel(writer, sheet_name='R2_Comparison')
            pivot_mse.to_excel(writer, sheet_name='MSE_Comparison')

        print("\n" + "="*70)
        print("VERGLEICH ABGESCHLOSSEN")
        print("="*70)
        print("\n📊 Beste Modelle nach R² Score:")
        print("─"*70)

        for portfolio_name in df_comparison['Portfolio'].unique():
            sub_df = df_comparison[df_comparison['Portfolio'] == portfolio_name]
            for period in ["daily", "intraday"]:
                period_data = sub_df[sub_df['Period'] == period]
                if period_data.empty:
                    continue

                best_idx = period_data['R2_Test'].idxmax()
                best_model = period_data.loc[best_idx]

                print(f"\n{portfolio_name} – {period.upper()} Daten:")
                print(f"  🏆 Bestes Modell: {best_model['Model']}")
                print(f"  📈 R² Test Score: {best_model['R2_Test']:.4f}")
                print(f"  📉 MSE: {best_model['MSE']:.6f}")
                print(f"  ⏱️  Training Zeit: {best_model['Training_Time_s']:.2f}s")

        # Speichere Modelle
        if self.config.get("output.save_models"):
            print("\n💾 Speichere Modelle...")
            self.save_models()

        print(f"\n✅ Ergebnisse gespeichert: {output_path}")
        print("="*70 + "\n")

    def compute_naive_baseline(self, y_train: pd.Series, y_test: pd.Series) -> dict:
        """Berechnet eine simple Lag-1 Baseline für Vergleichszwecke."""
        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(y_train)
        if not isinstance(y_test, pd.Series):
            y_test = pd.Series(y_test)

        combined = pd.concat([y_train, y_test])
        baseline = combined.shift(1).loc[y_test.index]
        if not baseline.empty and not y_train.empty:
            baseline.iloc[0] = y_train.iloc[-1]
        baseline = baseline.fillna(0.0)

        metrics = {
            'r2': r2_score(y_test, baseline),
            'mse': mean_squared_error(y_test, baseline),
            'mae': mean_absolute_error(y_test, baseline)
        }
        return metrics

    def save_models(self):
        """Speichert alle trainierten Modelle"""
        models_path = Path("Models")
        models_path.mkdir(exist_ok=True)

        for portfolio_key, period_dict in self.results.items():
            portfolio_name = self.portfolios.get(portfolio_key, {}).get("name", portfolio_key)
            for period, models in period_dict.items():
                if models is None:
                    continue

                period_path = models_path / portfolio_name / period
                period_path.mkdir(parents=True, exist_ok=True)

                for model_name, model_results in models.items():
                    if model_results is None or model_name == "naive_baseline":
                        continue

                    model_file = period_path / f"{model_name}.pkl"

                    try:
                        # Speichere je nach Modelltyp
                        if model_name == "pytorch_nn":
                            # PyTorch speichern
                            import torch
                            torch.save(model_results["model"], period_path / f"{model_name}.pt")
                        else:
                            # Sklearn Modelle speichern
                            joblib.dump(model_results["model"], model_file)

                        print(f"  ✓ {portfolio_name}/{period}/{model_name} gespeichert")

                    except Exception as e:
                        print(f"  ✗ Fehler beim Speichern von {portfolio_name}/{period}/{model_name}: {e}")


    def _get_portfolio_configs(self) -> Dict[str, dict]:
        portfolios = self.config.get("data.portfolios") or {}
        if not portfolios:
            return {
                "default": {
                    "name": "Portfolio",
                    "universe": self.config.get("data.universe", []),
                    "indices": self.config.get("data.indices", [])
                }
            }

        normalized = {}
        for key, cfg in portfolios.items():
            normalized[key] = {
                "name": cfg.get("name", key.upper()),
                "universe": cfg.get("universe", []),
                "indices": cfg.get("indices", [])
            }
        return normalized


if __name__ == "__main__":
    # Test
    comparison = ModelComparison()
    comparison.run_full_comparison()
