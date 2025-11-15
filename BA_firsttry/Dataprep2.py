from __future__ import annotations

import numpy as np
import pandas as pd


def Excelloader(
    path: str,
    sheet: int | str = 0,
) -> pd.DataFrame:
    """Sehr simpler Excel-Loader: ein Sheet -> DataFrame.

    - `path`: Pfad zur Excel-Datei (.xlsx, .xls, ...)
    - `sheet`: Sheet-Index oder -Name (Default: 0 = erstes Sheet)
    - Rückgabe: DataFrame genau so, wie `pd.read_excel` es liefert
    """
    df = pd.read_excel(path, sheet_name=sheet)
    df = df.dropna(how="any")

    return df


def price_plus_minus(df: pd.DataFrame) -> pd.DataFrame:
    """Erzeuge 0/1-Signal je Zeile basierend auf der Preisänderung zur Vorzeile.

    - Positiver Change -> 1
    - Negativer oder unveränderter Change -> 0
    - Nutzt 'Price' (falls vorhanden), sonst erste numerische Spalte.
    - Sortiert nach 'Date', falls vorhanden, bevor die Änderung berechnet wird.
    """
    out = df.copy()

    # Nach Datum sortieren, falls vorhanden
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.sort_values("Date").reset_index(drop=True)

    # Preisspalte identifizieren
    if "Price" in out.columns:
        price_col = "Price"
    else:
        num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
        price_col = num_cols[0] if num_cols else None

    if price_col is None:
        raise ValueError("Keine numerische 'Price'-Spalte gefunden.")

    # Numerisch sicherstellen
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")

    # Änderung zur Vorzeile und 0/1-Signal
    chg = out[price_col].pct_change(periods=1)
    out["change"] = chg
    # out["change"] = (chg > 0).astype(int)  # Hier der Fix für change %

    # Debug: erste Zeilen ausgeben
   # print(out.head())
    return out

# Anpassung: Momentum nur aus Vergangenheitsinformationen ableiten.
def createmomentum(
    df: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Fügt eine Momentum-Spalte hinzu (periodenbasiert, nur int-Window).

    Definition:
    momentum_t = Price_{t-1} / Price_{t-1-window} - 1

    Parameter:
    - window: Anzahl Perioden (z. B. 60)
    Voraussetzungen:
    - DataFrame enthält eine Spalte 'Price'.
    """
    out = df.copy()

    # Nach Datum sortieren, falls vorhanden
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.sort_values("Date").reset_index(drop=True)

    # Price numerisch sicherstellen
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")

    # Periodenbasiertes Momentum ausschließlich aus Preisen bis einschließlich t-1
    lagged_price = out["Price"].shift(1)
    out["momentum"] = lagged_price.pct_change(periods=window)
    out = out.dropna(how="any")
   # print(out.head())
    return out


# Anpassung: Future-Return erzeugen und sämtliche Features auf Vergangenheitswerte begrenzen.
def runningcycle(sheetstock:int) -> pd.DataFrame:
    """Compile the modelling frame with lagged-only inputs for forecasting."""

    df = Excelloader("DataStorage/Portfolio.xlsx", sheetstock)
    df = price_plus_minus(df)
    df = createmomentum(df, 5)
    df = df.drop(columns=["Price"])

    # Externe Märkte laden und auf vergangene Informationen begrenzen
    dax = Excelloader("DataStorage/INDEX.xlsx", 1)
    dax = price_plus_minus(dax)
    dax = dax[["Date", "change"]].rename(columns={"change": "change_DAX"})

    vdax = Excelloader("DataStorage/INDEX.xlsx", 0)
    vdax = price_plus_minus(vdax)
    vdax = vdax[["Date", "change"]].rename(columns={"change": "change_VDAX"})

    df = df.merge(dax, how="inner", on="Date")
    df = df.merge(vdax, how="inner", on="Date")
    # Nach dem Mergen sicherheitshalber nach Datum sortieren
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)

    # Zusätzliche abgeleitete Features
    df["change_lag1"] = df["change"].shift(1)
    df["change_lag5"] = df["change"].shift(5)
    df["change_roll_mean5"] = df["change"].shift(1).rolling(window=5).mean()
    df["change_roll_std5"] = df["change"].shift(1).rolling(window=5).std()

    df["momentum_lag1"] = df["momentum"].shift(1)

    df.rename(
        columns={
            "change_DAX": "change_dax",
            "change_VDAX": "change_vdax",
        },
        inplace=True,
    )
    for col in ("change_dax", "change_vdax"):
        if col not in df.columns:
            continue
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_roll_mean5"] = df[col].shift(1).rolling(window=5).mean()

    # Future-Return als Zielvariable für echtes Forecasting nutzen
    df["change_future"] = df["change"].shift(-1)

    df = df.dropna(how="any").reset_index(drop=True)
    return df


# Anpassung: Zielvariable auf zukünftigen Return verschoben und gleichzeitige Index-Informationen entfernt.
def splitdataXY(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Teilt df in Featurematrix X und Ziel Y.

    'change_future' dient als Target, alle übrigen Spalten (außer Datum und
    unverzögerte Index-Renditen) bilden X.
    """
    df = df.copy()
    # Toleranz für Groß-/Kleinschreibung bei DAX/VDAX
    df.rename(columns={
        "change_DAX": "change_dax",
        "change_VDAX": "change_vdax",
    }, inplace=True)

    if "change_future" not in df.columns:
        raise ValueError("'change_future' fehlt in den Daten. runningcycle zuerst aufrufen.")

    required = ["change", "momentum", "change_future"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten: {missing}. Vorhanden: {list(df.columns)}")

    Y = df[["change_future"]].copy()

    # Alle Features außer Ziel und Datum verwenden
    drop_cols = {"change", "change_future", "Date", "date", "change_dax", "change_vdax"}
    X_cols = [c for c in df.columns if c not in drop_cols]
    X = df[X_cols].copy()
    return X, Y

def time_series_split(X: np.ndarray, y: np.ndarray):
    """Chronologischer Split: frühe Daten -> Training, späte -> Validation."""
    val_split = 0.2
    n_samples = X.shape[0]
    if n_samples == 0:
        raise ValueError("Keine Datenpunkte vorhanden.")

    val_size = int(n_samples * val_split)
    train_size = n_samples - val_size
    if train_size <= 0:
        raise ValueError("Trainingssplit ergibt keine Trainingsdaten. val_split anpassen.")

    X_train = X[:train_size]
    X_val = X[train_size:]
    y_train = y[:train_size]
    y_val = y[train_size:]
    return X_train, X_val, y_train, y_val

# finalrunner nutzt nun change_future als Y, behält aber das bestehende Rückgabeformat bei.
def finalrunner(sheet:int):
    """Return train/validation splits with change_future as the target."""

    dataset = runningcycle(sheet)
    X, Y = splitdataXY(dataset)
    X_train, X_val, y_train, y_val = time_series_split(X, Y)
    return X_train, X_val, y_train, y_val
