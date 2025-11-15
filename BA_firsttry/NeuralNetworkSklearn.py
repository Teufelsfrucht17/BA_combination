from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import Dataprep2
import GloablVariableStorage
from createScoreModels import createscore


def _load_splits(sheet: int):
    """Lade Trainings-/Validierungssplits aus Dataprep2 mit Future-Target."""

    X_train, X_val, y_train, y_val = Dataprep2.finalrunner(sheet)
    feature_names = list(X_train.columns)
    return X_train, X_val, y_train, y_val, feature_names


def _build_pipeline(hidden_layer_sizes: Iterable[int], alpha: float, learning_rate_init: float) -> Pipeline:
    """Erzeuge eine Pipeline, die nur auf Trainingsdaten skaliert und dann ein MLP lernt."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=tuple(hidden_layer_sizes),
                    activation="relu",
                    solver="adam",
                    learning_rate="adaptive",
                    learning_rate_init=learning_rate_init,
                    alpha=alpha,
                    max_iter=500,
                    early_stopping=True,
                    n_iter_no_change=25,
                    random_state=42,
                ),
            ),
        ]
    )


def train_sklearn_mlp(sheet: int, model_out: Path) -> tuple[dict, dict]:
    """Trainiere ein Sklearn-MLP auf den gelaggten Features und future Returns."""

    X_train, X_val, y_train, y_val, feature_names = _load_splits(sheet)

    param_grid = {
        "mlp__hidden_layer_sizes": [
            (64,),
            (64, 32),
            (128, 64),
        ],
        "mlp__alpha": [1e-5, 1e-4, 1e-3],
        "mlp__learning_rate_init": [1e-3, 5e-4],
    }

    base = _build_pipeline(hidden_layer_sizes=(64,), alpha=1e-4, learning_rate_init=1e-3)
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=tscv,
        scoring="r2",
        n_jobs=-1,
        refit=True,
    )

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
        f"SklearnMLP - Sheet {sheet} | train_r2={metrics['train_r2']:.4f} | "
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


def runSklearnMLP(sheet: int | None = None, *, model_out: Path | None = None) -> pd.DataFrame:
    """Trainiere ein Sklearn-MLP für ein einzelnes Sheet und erfasse Kennzahlen."""

    if sheet is None:
        raise ValueError("Sheet index muss gesetzt sein")

    report = createscore()
    metrics, params = train_sklearn_mlp(
        sheet=sheet,
        model_out=model_out or Path(f"data_output/sklearn_mlp_{sheet}.joblib"),
    )

    report.loc[len(report)] = [
        "Sklearn MLP",
        sheet,
        metrics["train_r2"],
        metrics["val_r2"],
        params,
        "scaled + future-target",
    ]
    return report


def run_all_sheets(*, model_dir: Path | None = None) -> pd.DataFrame:
    """Trainiere das Sklearn-MLP über alle Portfolio-Sheets und sammle Scores."""

    report = createscore()
    try:
        for idx in range(len(GloablVariableStorage.Portfolio)):
            model_path = (model_dir / f"sklearn_mlp_{idx}.joblib") if model_dir else None
            result = runSklearnMLP(idx, model_out=model_path)
            report = pd.concat([report, result], ignore_index=True)
    except Exception as exc:
        print(f"Sklearn MLP run failed: {exc}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Trainiert ein Sklearn-MLP auf den Dataprep2-Splits")
    parser.add_argument("--sheet", type=int, default=0, help="Sheet-Index")
    parser.add_argument(
        "--model_out",
        type=Path,
        default=None,
        help="Speicherpfad für Modell und Metadaten",
    )
    args = parser.parse_args()

    runSklearnMLP(args.sheet, model_out=args.model_out)


if __name__ == "__main__":
    main()
