import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import Dataprep2
import GloablVariableStorage
from createScoreModels import createscore


def ridge_classification(sheet_index: int, report: pd.DataFrame | None = None) -> pd.DataFrame:
    """Train a Ridge model on one sheet and append metrics to the report."""

    X_train, X_test, y_train, y_test = Dataprep2.finalrunner(sheet_index)


    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    param_grid = {
        "model__alpha": [
            1e-5, 1e-4, 1e-3, 1e-2,
            3e-2, 1e-1, 3e-1,
            1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0,
        ],
        "model__fit_intercept": [True, False],
        "model__solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag"],
        "model__tol": [1e-4, 1e-3],
        "model__max_iter": [5000],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=tscv,
        scoring="r2",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train.values.ravel())

    y_train_pred = grid.predict(X_train)
    y_test_pred = grid.predict(X_test)

    acc_train = r2_score(y_train, y_train_pred)
    acc_test = r2_score(y_test, y_test_pred)

    print(f"Sheet {sheet_index}: Ridge r2 train = {acc_train:.4f}")
    print(f"Sheet {sheet_index}: Ridge r2 test  = {acc_test:.4f}")
    print(f"Sheet {sheet_index}: Best alpha = {grid.best_params_['model__alpha']}")

    report.loc[len(report)] = [
        "Ridge",
        sheet_index,
        acc_train,
        acc_test,
        grid.best_params_['model__alpha'],
        f"ridge_classification(sheet_index={sheet_index})",
    ]
    joblib.dump(grid, "./data_output/RR/OLSModel" + str(sheet_index) + ".pkl")

    return report


def runRidgeRegession() -> pd.DataFrame:
    """Run Ridge on all sheets and collect the combined score report."""
    report = createscore()
    try:
        for i in range(len(GloablVariableStorage.Portfolio)):
            report = ridge_classification(i, report)
    except Exception as e:
        print(f"Ridge run failed: {e}")

    return report

