import datetime
from pathlib import Path

import pandas as pd

import GloablVariableStorage
import LSEG as LS


def exceltextwriter(df: pd.DataFrame, name: str) -> None:
    if df is None or df.empty:
        print("⚠️ Keine Daten zurückgegeben – keine Excel erstellt.")
        return

    # Datumsspalte aus dem Index herstellen, falls nötig
    d = df.copy()
    if isinstance(d.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        d = d.reset_index()
        # Nach reset_index heißt die Spalte entweder nach Index-Name oder 'index'
        if "index" in d.columns:
            d = d.rename(columns={"index": "Date"})
        elif d.columns[0] not in ("Date", "Datetime"):
            d = d.rename(columns={d.columns[0]: "Date"})
    else:
        # Alternativ: vorhandene Datums-/Zeitspalten erkennen und vereinheitlichen
        for c in ("Date", "Datetime", "DATE", "timestamp"):
            if c in d.columns:
                if c != "Date":
                    d = d.rename(columns={c: "Date"})
                break

    out_dir = Path("DataStorage")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.xlsx"

    with pd.ExcelWriter(
        out_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm:ss"
    ) as writer:
        for column in df.columns:
            sheet_name = str(column).strip()[:31] or "sheet"
            if "Date" in d.columns:
                frame = d[["Date", column]].copy()
            else:
                # Fallback: ohne Datums-Spalte
                frame = d[[column]].copy()
            # Spaltennamen vereinheitlichen, damit keine MultiIndex-Header entstehen
            if "Date" in frame.columns and frame.shape[1] == 2:
                frame.columns = ["Date", "Price"]
            elif frame.shape[1] == 1:
                frame.columns = ["Price"]

            frame.to_excel(writer, index=False, sheet_name=sheet_name)

    print(f"✅ Excel gespeichert: {out_path}")
    return

def createExcel(
    universe: list[str],
    fields: list[str],
    start: datetime.datetime,
    end: datetime.datetime,
    interval: str,
    name: str,
) -> None:
    """Daten holen und als Excel speichern (kombiniert + je Aktie eigenes Sheet)."""
    df = LS.getHistoryData(
        universe=universe, fields=fields, start=start, end=end, interval=interval )
    exceltextwriter(df, name)
    return



if __name__ == "__main__":
    createExcel(
        universe= [".V1XI",".GDAXI"],
        fields=["OPEN_PRC"],
        start=datetime.datetime(2015, 1, 1),
        end=datetime.datetime(2025, 10, 25),
        interval="D",
        name="INDEX",
    )
