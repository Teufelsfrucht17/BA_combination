import joblib
import pandas as pd
from pandas.core.common import random_state


import Dataprep2
import GloablVariableStorage
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from createScoreModels import createscore


def OLSRegression(sheet:int,report:pd.DataFrame) -> pd.DataFrame:



    X_train_OLS, X_test_OLS, Y_train_OLS, Y_test_OLS = Dataprep2.finalrunner(sheet)

    param_grid = {'fit_intercept': [True, False]}
    # Set model specs
    ols_model = LinearRegression()
    tscv = TimeSeriesSplit(n_splits=5)
    CV_olsmodel = GridSearchCV(
        estimator=ols_model,
        param_grid=param_grid,
        cv=tscv,
        n_jobs=-1,
    )
    CV_olsmodel.fit(X_train_OLS, Y_train_OLS.values.ravel())

    # Prediction and result
    y_train_pred = CV_olsmodel.predict(X_train_OLS)
    y_test_pred = CV_olsmodel.predict(X_test_OLS)

    r2_train = r2_score(Y_train_OLS, y_train_pred)
    r2_test = r2_score(Y_test_OLS, y_test_pred)

    print("Sheet:"+str(sheet)+" in-sample R2 =", r2_train) #in-sample R2
    print("Sheet:"+str(sheet)+" Out-of-sample R2 =", r2_test) #Out-of-sample R2

    report.loc[len(report)] = [
        "OLS",
        sheet,
        r2_train,
        r2_test,
        CV_olsmodel.best_params_['fit_intercept'],
        "N/A",
    ]
    payload = {
        "model": CV_olsmodel.best_estimator_,
        "feature_names": list(X_train_OLS.columns),
        "sheet": sheet,
        "best_params": CV_olsmodel.best_params_,
    }
    joblib.dump(payload, "./data_output/OLS/OLSModel"+str(sheet)+".pkl")
    return report

def runOLS() -> pd.DataFrame:

    report = createscore()

    try:
        for i in range(len(GloablVariableStorage.Portfolio)):
            report = OLSRegression(i, report)
    except Exception as e:
        print(f"Ridge run failed: {e}")

    return report

runOLS()
