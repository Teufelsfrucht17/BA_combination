from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

import Dataprep2
import GloablVariableStorage
from createScoreModels import createscore


def _split_data(sheet: int):
    X_train, X_val, y_train, y_val = Dataprep2.finalrunner(sheet)
    feature_names = list(X_train.columns)
    return X_train, X_val, y_train, y_val, feature_names


def train_random_forest(
    sheet: int,
    model_out: Path,
) -> tuple[dict, dict]:
    param_grid = {
        "max_depth": [4, 6, 8, None],
        "n_estimators": [100, 200, 400],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["squared_error"],
    }

    X_train, X_val, y_train, y_val, feature_names = _split_data(sheet)

    reg = RandomForestRegressor(random_state=42, n_jobs=-1)
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(reg, param_grid=param_grid, cv=tscv, n_jobs=-1)
    grid.fit(X_train, y_train.values.ravel())

    y_train_pred = grid.predict(X_train)
    y_val_pred = grid.predict(X_val)

    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred) if len(y_val) else float("nan")

    metrics = {
        "train_r2": r2_score(y_train, y_train_pred),
        "val_r2": r2_score(y_val, y_val_pred) if len(y_val) else float("nan"),
        "train_rmse": float(np.sqrt(train_mse)),
        "val_rmse": float(np.sqrt(val_mse)) if not np.isnan(val_mse) else float("nan"),
    }

    print(
        f"RandomForest - Sheet {sheet} | train_r2={metrics['train_r2']:.4f} | "
        f"val_r2={metrics['val_r2']:.4f} | train_rmse={metrics['train_rmse']:.6f} | "
        f"val_rmse={metrics['val_rmse']:.6f}"
    )

    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": grid.best_estimator_,
            "feature_names": feature_names,
            "sheet": sheet,
            "metrics": metrics,
            "best_params": grid.best_params_,
        },
        model_out,
        compress=("gzip", 3),
    )
    print(f"Gespeichert: {model_out}")
    return metrics, grid.best_params_


def RF(
    sheet: int | None,
    report: pd.DataFrame | None = None,
    *,
    model_out: Path | None = None,
) -> pd.DataFrame:
    if sheet is None:
        raise ValueError("Sheet index muss gesetzt sein")
    if report is None:
        report = createscore()

    model_path = model_out or Path(f"data_output/random_forest_{sheet}.joblib")
    metrics, params = train_random_forest(sheet=sheet, model_out=model_path)

    report.loc[len(report)] = [
        "Random Forest",
        sheet,
        metrics["train_r2"],
        metrics["val_r2"],
        params,
        "N/A",
    ]
    return report


def Run_RandomForest(*, model_dir: Path | None = None) -> pd.DataFrame:
    report = createscore()
    try:
        for i in range(len(GloablVariableStorage.Portfolio)):
            model_out = (model_dir / f"random_forest_{i}.joblib") if model_dir else None
            report = RF(i, report, model_out=model_out)
    except Exception as e:
        print(f"RandomForest run failed: {e}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RandomForest-Regressor auf Dataprep2-Daten trainieren")
    parser.add_argument("--sheet", type=int, default=0, help="Sheet-Index")
    parser.add_argument(
        "--model_out",
        type=Path,
        default=None,
        help="Speicherpfad für Modell und Metadaten",
    )
    cli_args = parser.parse_args()
    RF(cli_args.sheet, model_out=cli_args.model_out)
