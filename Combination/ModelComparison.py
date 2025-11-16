"""
ModelComparison.py - Hauptmodul für Modellvergleich
Trainiert alle Modelle und erstellt detaillierten Vergleichsbericht
"""

import pandas as pd
import numpy as np
import time
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

        print("\n" + "="*70)
        print("BA TRADING SYSTEM - PORTFOLIO-BASIERTER MODELLVERGLEICH")
        print("="*70)

        # 1. Daten holen (Portfolio-basiert)
        print("\n[SCHRITT 1/4] DATENABRUF")
        grabber = DataGrabber(self.config.path)
        all_data = grabber.fetch_all_data()  # {"dax": {"daily": df, "intraday": df}, "sdax": {...}}

        # 2. Dataprep
        prep = DataPrep(self.config.path)

        # Für jedes Portfolio
        for portfolio_name, portfolio_data in all_data.items():
            portfolio_config = self.config.get(f"data.portfolios.{portfolio_name}")
            portfolio_display_name = portfolio_config.get("name", portfolio_name.upper())

            # Für beide Zeitperioden
            for period_type, data in portfolio_data.items():
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

                print(f"\nTrain Size: {len(X_train)} samples")
                print(f"Test Size: {len(X_test)} samples")

                # Trainiere alle Modelle
                results_key = f"{portfolio_name}_{period_type}"
                self.results[results_key] = self.train_all_models(
                    X_train, X_test, y_train, y_test, portfolio_name, period_type
                )

        # 3. Vergleich erstellen
        print("\n[SCHRITT 4/4] ERSTELLE VERGLEICHSBERICHT")
        self.create_comparison_report()

    def train_all_models(self, X_train, X_test, y_train, y_test, portfolio_name: str, period_type: str) -> dict:
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

        print(f"\n[SCHRITT 2/4] FEATURE ENGINEERING ABGESCHLOSSEN")
        print(f"[SCHRITT 3/4] MODELL-TRAINING ({portfolio_display} - {period_type.upper()})")

        # =============================================
        # Baseline Model (Naive Predictor)
        # =============================================
        print(f"\n{'─'*60}")
        print("Baseline Model (Naive Predictor)")
        print(f"{'─'*60}")

        start = time.time()

        try:
            model, metrics = train_naive_baseline(X_train, y_train, X_test, y_test)

            training_time = time.time() - start

            results["naive_baseline"] = {
                "model": model,
                "metrics": metrics,
                "training_time": training_time
            }

            print(f"  ✓ R² Test: {metrics['r2']:.4f}")
            print(f"  ✓ MSE: {metrics['mse']:.6f}")
            print(f"  ✓ MAE: {metrics['mae']:.6f}")
            print(f"  ✓ Training Zeit: {training_time:.2f}s")
            print("  ℹ️  Baseline dient als Vergleichsmaßstab (sollte von ML-Modellen übertroffen werden)")

        except Exception as e:
            print(f"  ✗ Fehler: {e}")
            results["naive_baseline"] = None

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
                    early_stopping_patience=20,
                    use_scheduler=True
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
                print(f"  ✓ Training Zeit: {training_time:.2f}s")

            except Exception as e:
                print(f"  ✗ Fehler: {e}")
                results["pytorch_nn"] = None

        # =============================================
        # Sklearn Neural Network
        # =============================================
        if self.config.get("models.sklearn_nn.enabled"):
            print(f"\n{'─'*60}")
            print("Sklearn Neural Network")
            print(f"{'─'*60}")

            start = time.time()

            try:
                model, metrics = train_sklearn_nn(
                    X_train, y_train, X_test, y_test,
                    hidden_layer_sizes=tuple(self.config.get("models.sklearn_nn.hidden_layer_sizes", [64, 32])),
                    max_iter=self.config.get("models.sklearn_nn.max_iter", 500),
                    n_splits=self.config.get("training.cross_validation.n_splits", 5),
                    use_gridsearch=self.config.get("training.cross_validation.enabled", True)
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
                    alpha_values=self.config.get("models.ridge.alpha_values", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
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
                model, metrics = train_random_forest(
                    X_train, y_train, X_test, y_test,
                    n_estimators=self.config.get("models.random_forest.n_estimators", 300),
                    max_depth=self.config.get("models.random_forest.max_depth", 10),
                    min_samples_split=self.config.get("models.random_forest.min_samples_split", 5),
                    n_splits=self.config.get("training.cross_validation.n_splits", 5),
                    use_gridsearch=self.config.get("training.cross_validation.enabled", True)
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
                print(f"  ✓ Training Zeit: {training_time:.2f}s")

            except Exception as e:
                print(f"  ✗ Fehler: {e}")
                results["random_forest"] = None

        return results

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
                    "Training_Time_s": model_results["training_time"]
                })

        # Erstelle DataFrame
        df_comparison = pd.DataFrame(comparison_data)

        if df_comparison.empty:
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
            print("\n💾 Speichere Modelle...")
            self.save_models()

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
                    print(f"  ✗ Fehler beim Speichern von {results_key}/{model_name}: {e}")


if __name__ == "__main__":
    # Test
    comparison = ModelComparison()
    comparison.run_full_comparison()
