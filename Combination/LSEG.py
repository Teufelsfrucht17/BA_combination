import lseg.data as ld
import pandas as pd
from lseg.data.discovery import Chain
import datetime
from IPython.display import display, clear_output

import GloablVariableStorage


def getHistoryData(universe :  list[str], fields: list[str], start :datetime.datetime , end : datetime.datetime, interval : str) -> pd.DataFrame:
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


# Keine Tests - wird von Datagrabber aufgerufen