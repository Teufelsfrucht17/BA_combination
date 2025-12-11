"""
ModelComparison.py - Core module for model comparison
Trains all models and builds a detailed comparison report
"""

import pandas as pd
import numpy as np
import time
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Datagrabber import DataGrabber
from Dataprep import DataPrep, time_series_split
from ConfigManager import ConfigManager
from FamaFrench import FamaFrenchFactorModel, calculate_fama_french_factors
from Models_Wrapper import (
    train_pytorch_model,
    tune_pytorch_model_optuna,
    train_ols,
    train_ridge,
    train_random_forest
)
import LSEG as LS
from logger_config import get_logger


logger = get_logger(__name__)


class ModelComparison:
    """Compare machine learning models for daily vs intraday data"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize ModelComparison

        Args:
            config_path: Path to the config file
        """
        self.config = ConfigManager(config_path)
        self.results = {}

    def run_full_comparison(self):
        """Run the full comparison across portfolios, periods, and models"""

        ffc_runs_enabled = bool(self.config.get("training.ffc_runs", False))

        grabber = DataGrabber(self.config.path)
        all_data = grabber.fetch_all_data()

        all_company_data = grabber.fetch_company_data()

        prep = DataPrep(self.config.path)

        for portfolio_name, portfolio_data in all_data.items():
            portfolio_config = self.config.get(f"data.portfolios.{portfolio_name}")
            portfolio_display_name = portfolio_config.get("name", portfolio_name.upper())
            
            company_df = all_company_data.get(portfolio_name, pd.DataFrame())
            
            if not isinstance(company_df, pd.DataFrame):
                company_df = pd.DataFrame()

            for period_type, data in portfolio_data.items():
                ff_factors = None
                if not company_df.empty:
                    if period_type == "intraday":
                        data_daily = data.copy()
                        if isinstance(data_daily.index, pd.DatetimeIndex):
                            data_daily.index = data_daily.index.normalize()
                        data_daily = data_daily.groupby(data_daily.index).first()
                        
                        ff_model = FamaFrenchFactorModel(self.config.path)
                        portfolio_config = self.config.get(f"data.portfolios.{portfolio_name}")
                        index_col = f"{portfolio_config.get('index', '.GDAXI')}_TRDPRC_1"

                        ff_factors_daily = ff_model.calculate_factors(
                            price_df=data_daily,
                            company_df=company_df,
                            index_col=index_col,
                            portfolio_name=portfolio_name
                        )
                        
                        if not ff_factors_daily.empty and isinstance(data.index, pd.DatetimeIndex):
                            ff_factors = pd.DataFrame(index=data.index, columns=ff_factors_daily.columns)
                            for date in data.index:
                                date_normalized = date.normalize()
                                if date_normalized in ff_factors_daily.index:
                                    ff_factors.loc[date] = ff_factors_daily.loc[date_normalized]
                    else:
                        ff_model = FamaFrenchFactorModel(self.config.path)
                        portfolio_config = self.config.get(f"data.portfolios.{portfolio_name}")
                        index_col = f"{portfolio_config.get('index', '.GDAXI')}_TRDPRC_1"

                        ff_factors = ff_model.calculate_factors(
                            price_df=data,
                            company_df=company_df,
                            index_col=index_col,
                            portfolio_name=portfolio_name
                        )
                    if ff_factors is None or ff_factors.empty:
                        ff_factors = None

                use_ffc_options = [False, True] if ffc_runs_enabled else [False]
                for use_ffc in use_ffc_options:
                    if use_ffc and (ff_factors is None or ff_factors.empty):
                        continue
                    
                    suffix = "_FFC" if use_ffc else ""
                    results_key = f"{portfolio_name}_{period_type}{suffix}"
                    
                    ff_factors_to_use = ff_factors if use_ffc else None
                    X, y = prep.prepare_data(
                        data, 
                        portfolio_name=portfolio_name, 
                        period_type=period_type,
                        ff_factors=ff_factors_to_use
                    )

                    if use_ffc:
                        ff_cols_in_x = [c for c in ['Mkt_Rf', 'SMB', 'HML', 'WML'] if c in X.columns]
                        if ff_cols_in_x:
                            logger.info(
                                "FFC-Faktoren in Features für %s_%s_FFC: %s",
                                portfolio_name,
                                period_type,
                                ff_cols_in_x
                            )
                        else:
                            logger.warning(
                                "FFC-Run ohne FFC-Faktoren in X für %s_%s_FFC",
                                portfolio_name,
                                period_type
                            )

                    test_split = self.config.get("training.test_split", 0.2)
                    X_train, X_test, y_train, y_test = time_series_split(X, y, test_size=test_split)

                    scaler_method = self.config.get("training.scaling.method", "StandardScaler")
                    scaler = MinMaxScaler() if scaler_method == "MinMaxScaler" else StandardScaler()
                    scaler.fit(X_train)

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

                    self.results[results_key] = self.train_all_models(
                        X_train, X_test, y_train, y_test, portfolio_name, period_type, use_ffc=use_ffc
                    )

        self.create_comparison_report()

    def train_all_models(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train: pd.Series, 
        y_test: pd.Series, 
        portfolio_name: str, 
        period_type: str,
        use_ffc: bool = False
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Train all enabled models

        Args:
            X_train, X_test, y_train, y_test: train/test splits
            portfolio_name: Portfolio name (e.g. "dax", "sdax")
            period_type: "daily" or "intraday"

        Returns:
            Dictionary with results of all models
        """
        results = {}

        portfolio_config = self.config.get(f"data.portfolios.{portfolio_name}")
        portfolio_display = portfolio_config.get("name", portfolio_name.upper())

        use_optuna_for_nn = self.config.get("models.pytorch_nn.optuna.enabled", False)

        if use_optuna_for_nn:
            pytorch_train_func = tune_pytorch_model_optuna

            def _pytorch_get_kwargs():
                base_params = {
                    "hidden1": self.config.get("models.pytorch_nn.hidden1", 64),
                    "hidden2": self.config.get("models.pytorch_nn.hidden2", 32),
                    "epochs": self.config.get("models.pytorch_nn.epochs", 200),
                    "batch_size": self.config.get("models.pytorch_nn.batch_size", 64),
                    "lr": self.config.get("models.pytorch_nn.learning_rate", 0.001),
                    "validation_split": self.config.get("models.pytorch_nn.validation_split", 0.2),
                    "early_stopping_patience": self.config.get("models.pytorch_nn.early_stopping_patience", 20),
                    "use_scheduler": True,
                    "scheduler_patience": self.config.get("models.pytorch_nn.scheduler_patience", 10),
                    "weight_decay": self.config.get("models.pytorch_nn.weight_decay", 0.0)
                }
                param_grid = self.config.get("models.pytorch_nn.optuna.param_grid", {})
                n_trials = self.config.get("models.pytorch_nn.optuna.n_trials", None)
                return {
                    "base_params": base_params,
                    "param_grid": param_grid,
                    "n_trials": n_trials,
                    "portfolio_name": portfolio_name,
                    "period_type": period_type,
                    "visualize_model": self.config.get("models.pytorch_nn.visualize_model", False)
                }
        else:
            pytorch_train_func = train_pytorch_model

            def _pytorch_get_kwargs():
                return {
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
                    "period_type": period_type,
                    "visualize_model": self.config.get("models.pytorch_nn.visualize_model", False)
                }

        model_configs = {
            "pytorch_nn": {
                "enabled": self.config.get("models.pytorch_nn.enabled", False),
                "train_func": pytorch_train_func,
                "display_name": "PyTorch Neural Network",
                "get_kwargs": _pytorch_get_kwargs
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

        active_models = self.config.get("models.active_models", [])
        active_models_set = set(active_models) if isinstance(active_models, (list, tuple, set)) else set()

        for model_name, config in model_configs.items():
            if active_models_set and model_name not in active_models_set:
                continue

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
        Train a single model (helper to avoid duplication)

        Args:
            model_name: Model name
            config: Model configuration (enabled, train_func, display_name, get_kwargs, extra_info)
            X_train, X_test, y_train, y_test: Train/test splits
            results: Dictionary to store results
        """
        start = time.time()

        kwargs = config.get("get_kwargs", lambda: {})()

        model, metrics = config["train_func"](X_train, y_train, X_test, y_test, **kwargs)

        training_time = time.time() - start

        results[model_name] = {
            "model": model,
            "metrics": metrics,
            "training_time": training_time
        }

        self._print_model_results(metrics, training_time, model_name, config.get("extra_info"))

    def _print_model_results(
        self,
        metrics: Dict[str, Any],
        training_time: float,
        model_name: str,
        extra_info: Optional[str] = None
    ) -> None:
        """
        Print model results in a uniform format

        Args:
            metrics: Dictionary with metrics
            training_time: Training time in seconds
            model_name: Model name
            extra_info: Optional extra information
        """
        return

    def create_comparison_report(self):
        """Create detailed portfolio-based comparison report as Excel"""

        comparison_data = []

        for results_key, models in self.results.items():
            use_ffc = False
            if results_key.endswith("_FFC"):
                use_ffc = True
                results_key_base = results_key[:-4]
            else:
                results_key_base = results_key
            
            if "_" in results_key_base:
                portfolio_name, period = results_key_base.rsplit("_", 1)
            else:
                portfolio_name, period = "unknown", results_key_base

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
                    "FFC_Factors": "Yes" if use_ffc else "No",
                    "Model": model_name,
                    "R2_Test": model_results["metrics"]["r2"],
                    "R2_Train": model_results["metrics"].get("train_r2", np.nan),
                    "MSE": model_results["metrics"]["mse"],
                    "MAE": model_results["metrics"]["mae"],
                    "Directional_Accuracy": model_results["metrics"].get("directional_accuracy", np.nan),
                    "Directional_Accuracy_Train": model_results["metrics"].get("directional_accuracy_train", np.nan),
                    "Training_Time_s": model_results["training_time"]
                })

        df_comparison = pd.DataFrame(comparison_data)

        if df_comparison.empty:
            return

        output_path = Path("Results") / "model_comparison.xlsx"
        output_path.parent.mkdir(exist_ok=True)

        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            df_comparison.to_excel(writer, sheet_name='Full_Comparison', index=False)

        if self.config.get("output.save_models"):
            self.save_models()

    def save_models(self):
        """Save all trained models (portfolio based)"""
        models_path = Path("Models")
        models_path.mkdir(exist_ok=True)

        for results_key, models in self.results.items():
            if results_key.endswith("_FFC"):
                results_key_base = results_key[:-4]
            else:
                results_key_base = results_key
            
            if "_" in results_key_base:
                portfolio_name, period = results_key_base.rsplit("_", 1)
            else:
                portfolio_name, period = "unknown", results_key_base

            portfolio_period_path = models_path / results_key
            portfolio_period_path.mkdir(exist_ok=True)

            for model_name, model_results in models.items():
                if model_results is None or model_results["model"] is None:
                    continue

                model_file = portfolio_period_path / f"{model_name}.pkl"

                if model_name == "pytorch_nn":
                    import torch
                    torch.save(model_results["model"], portfolio_period_path / f"{model_name}.pt")
                else:
                    joblib.dump(model_results["model"], model_file)


if __name__ == "__main__":
    comparison = ModelComparison()
    comparison.run_full_comparison()
