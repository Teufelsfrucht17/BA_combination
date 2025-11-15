import lseg.data as ld
import pandas as pd
from lseg.data.discovery import Chain
import datetime
from IPython.display import display, clear_output

import GloablVariableStorage


def getHistoryData(universe : list[str], fields: list[str] | None , start :datetime.datetime , end : datetime.datetime, interval : str) -> pd.DataFrame:
    ld.open_session()

    df = ld.get_history(
        universe= universe,
        fields=fields,
        start=start,
        end=end,
        interval=interval,
    )
    print(df)

    ld.close_session()

    return df


#test = getHistoryData(universe=GloablVariableStorage.Portfolio ,fields=["OPEN_PRC"], start=datetime.datetime(2015, 1, 1), end=datetime.datetime(2025, 11, 1), interval="30min")
test = getHistoryData(universe=[".GDAXI"] ,fields=None, start=datetime.datetime(2015, 1, 1), end=datetime.datetime(2025, 11, 1), interval="30min")