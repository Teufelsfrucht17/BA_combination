import joblib
import pandas as pd

import Dataprep2


def  predictOLS(sheet: int):

    abc= joblib.load("./data_output/OLS/OLSModel"+str(sheet)+".pkl")
    X_future = Dataprep2.runningcycle(sheet)
    X_future = X_future[abc["feature_names"]]
    pred_future = abc.predict(X_future)
    print(pred_future)

predictOLS(0)