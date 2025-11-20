# BA Trading System - Unified Version

Eine vereinheitlichte Version des BA_combination Trading-Systems, die die Stärken beider Versionen kombiniert und ein Portfolio-basiertes Machine Learning System für Aktienprognosen implementiert.

## 🎯 Projektziel

Dieses System kombiniert:
- **Version 1 (BA_firsttry)**: Basis-Implementation mit selbst entwickelten Komponenten
- **Version 2 (BA_secondtry)**: Professionelle Features und Struktur

**Hauptfeatures:**
- **Portfolio-basiertes Machine Learning**: Separate Portfolios für DAX (Large Cap) und SDAX (Small Cap)
- **Multi-Perioden Analyse**: Vergleich von Daily vs. Intraday (30-Min) Daten
- **5 verschiedene ML-Modelle**: PyTorch NN, Sklearn NN, OLS, Ridge, Random Forest
- **Konfigurierbare Features**: Flexible Feature-Auswahl über `config.yaml`
- **Automatischer Modellvergleich**: Excel-Reports mit detaillierten Metriken
- **Zeitreihen-gerechte Validierung**: TimeSeriesSplit ohne Data Leakage
- **Hyperparameter-Optimierung**: GridSearchCV für Random Forest und MLP

## 📁 Projektstruktur

```
Combination/
├── config.yaml                 # Zentrale Konfiguration (Portfolios, Features, Modelle)
├── DataStorage/               # Excel/CSV Dateien (Rohdaten)
│   ├── combined_daily.xlsx
│   ├── combined_intraday.xlsx
│   ├── dax_daily.xlsx
│   ├── dax_intraday.xlsx
│   ├── sdax_daily.xlsx
│   └── sdax_intraday.xlsx
├── Models/                    # Gespeicherte Modelle (Portfolio-basiert)
│   ├── DAX/
│   │   ├── daily/
│   │   └── intraday/
│   ├── SDAX/
│   │   ├── daily/
│   │   └── intraday/
│   ├── dax_daily/
│   ├── dax_intraday/
│   ├── sdax_daily/
│   └── sdax_intraday/
├── Results/                   # Vergleichsergebnisse
│   ├── model_comparison.xlsx  # Haupt-Report
│   └── pytorch_training_*.csv # Training-Logs
│
├── main.py                    # ⭐ Hauptprogramm (CLI-Interface)
├── ConfigManager.py           # Config-System (YAML-Loader)
├── Datagrabber.py            # Datenabruf (LSEG/Refinitiv API)
├── Dataprep.py               # Datenaufbereitung & Feature Engineering
├── Models_Wrapper.py         # ML-Modell Wrapper (alle 5 Modelle)
├── ModelComparison.py        # Modellvergleich & Evaluation (Orchestrator)
│
├── LSEG.py                   # LSEG API Interface
├── GloablVariableStorage.py  # Globale Variablen
│
├── requirements.txt          # Python Dependencies
└── README.md                 # Diese Dokumentation
```

## 🚀 Installation

### 1. Python-Umgebung erstellen

```bash
# Mit venv (empfohlen)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate  # Windows

# Mit conda
conda create -n ba_trading python=3.10
conda activate ba_trading
```

### 2. Dependencies installieren

```bash
pip install -r requirements.txt
```

**Wichtig:** Für LSEG/Refinitiv Datenzugriff benötigst du:
- LSEG Data API Zugangsdaten
- Entsprechende API-Konfiguration in `LSEG.py`
- Netzwerkverbindung zum LSEG-Server

## ⚙️ Konfiguration

Die zentrale Konfiguration erfolgt über `config.yaml`. Alle Einstellungen können hier angepasst werden.

### Portfolio-Definition

Das System unterstützt mehrere Portfolios mit jeweils eigenem Aktien-Universum und Index:

```yaml
data:
  portfolios:
    dax:
      name: "DAX Portfolio"
      universe:
        - RHMG.DE
        - ENR1n.DE
        - TKAG.DE
        - FTKn.DE
        - ACT1.DE
        - DEZG.DE
      index: ".GDAXI"  # DAX Index (Large Cap)
      index_feature: "change_dax"

    sdax:
      name: "SDAX Portfolio"
      universe:
        - 1U1.DE
        - ADNGk.DE
        - AOFG.DE
        - COKG.DE
        - CWCG.DE
        - DMPG.DE
      index: ".SDAXI"  # SDAX Index (Small Cap)
      index_feature: "change_sdax"

  common_indices:
    - ".V1XI"  # VDAX (Volatilitätsindex) für alle Portfolios
```

### Zeitperioden

```yaml
data:
  periods:
    daily:
      interval: "daily"
      start: "2024-01-01"
      end: "2025-11-15"

    intraday:
      interval: "30min"
      start: "2024-01-01"
      end: "2025-11-15"
```

### Features konfigurieren

```yaml
features:
  input_features:
    - momentum_5              # 5-Perioden Momentum
    - momentum_10             # 10-Perioden Momentum
    - momentum_20             # 20-Perioden Momentum
    - portfolio_index_change  # Automatisch: change_dax oder change_sdax
    - vdax_absolute           # VDAX absoluter Wert
    - volume_ratio            # Volume Verhältnis
    - rolling_volatility_10   # Volatilität über 10 Perioden
    - rolling_volatility_20   # Volatilität über 20 Perioden
    - hour_sin                # Intraday: zyklische Stunde (Sinus)
    - hour_cos                # Intraday: zyklische Stunde (Cosinus)
    - dow_sin                 # Wochentag (Sinus)
    - dow_cos                 # Wochentag (Cosinus)

  target: "price_change_next"  # Y-Variable: Nächste Perioden-Return
```

### Modelle aktivieren/deaktivieren

```yaml
models:
  pytorch_nn:
    enabled: true
    hidden1: 128
    hidden2: 64
    epochs: 400
    batch_size: 64
    learning_rate: 0.0005
    validation_split: 0.2
    early_stopping_patience: 40
    scheduler_patience: 15

  sklearn_nn:
    enabled: true
    hidden_layer_sizes: [64, 32]
    max_iter: 1500

  ols:
    enabled: true

  ridge:
    enabled: true
    alpha_values: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

  random_forest:
    enabled: true
    n_estimators: 300
    max_depth: 10
    min_samples_split: 5
```

### Training-Parameter

```yaml
training:
  test_split: 0.2  # 20% für Test-Set (chronologisch)
  cross_validation:
    enabled: true
    n_splits: 5
    type: "TimeSeriesSplit"  # Zeitreihen-gerechte CV

  scaling:
    method: "StandardScaler"  # oder "MinMaxScaler"
```

## 📊 Verwendung

### Standard-Ausführung

```bash
# Vergleiche alle Portfolios (DAX + SDAX) und beide Perioden (Daily + Intraday)
python main.py
```

Dies führt aus:
- DAX Portfolio: Daily + Intraday Training
- SDAX Portfolio: Daily + Intraday Training
- Alle aktivierten Modelle werden trainiert
- Excel-Report wird erstellt

### Nur bestimmte Periode

```bash
# Nur tägliche Daten für alle Portfolios
python main.py --mode daily

# Nur 30-Minuten Daten für alle Portfolios
python main.py --mode intraday
```

### Custom Features

```bash
# Nur bestimmte Features verwenden (überschreibt config.yaml)
python main.py --features momentum_5 momentum_10 portfolio_index_change vdax_absolute
```

### Nur bestimmte Modelle

```bash
# Nur Neural Networks trainieren
python main.py --models pytorch_nn sklearn_nn

# Nur OLS und Ridge
python main.py --models ols ridge

# Nur Random Forest
python main.py --models random_forest
```

### Custom Config-Datei

```bash
python main.py --config my_custom_config.yaml
```

### Modelle nicht speichern

```bash
# Für schnelle Tests ohne Modell-Speicherung
python main.py --no-save
```

### Alle Optionen kombinieren

```bash
python main.py \
  --mode daily \
  --features momentum_5 momentum_10 portfolio_index_change \
  --models random_forest pytorch_nn \
  --no-save
```

## 📈 Ausgabe & Ergebnisse

### Konsolen-Ausgabe während der Ausführung

```
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║            BA TRADING SYSTEM - UNIFIED VERSION                       ║
║                                                                      ║
║  Kombiniert die Stärken von Version 1 und Version 2                 ║
║  Portfolio-basiertes Machine Learning Trading System                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

======================================================================
KONFIGURATION
======================================================================
Config-Datei:     config.yaml
Modus:            compare
Portfolios:       dax, sdax
Portfolio Größe:  12 Aktien (6 DAX + 6 SDAX)
Features:         ['momentum_5', 'momentum_10', 'portfolio_index_change', ...]
Aktive Modelle:   ['pytorch_nn', 'sklearn_nn', 'ols', 'ridge', 'random_forest']
======================================================================

[SCHRITT 1/4] DATENABRUF
Lade DAX Portfolio...
Lade SDAX Portfolio...

======================================================================
TRAINING: DAX Portfolio - DAILY
======================================================================

[SCHRITT 2/4] DATENAUFBEREITUNG
Erstelle Features...
  ✓ Features erstellt: (1234, 12)
Erstelle X und Y...
  ✓ X Shape: (1234, 11)
  ✓ y Shape: (1234,)

Train Size: 987 samples
Test Size: 247 samples

[SCHRITT 3/4] MODELLTRAINING
Training naive_baseline...
Training pytorch_nn...
Epoch 1/400: Train Loss: 0.001234, Val Loss: 0.001456, LR: 0.0005
...
Training sklearn_nn...
Training ols...
Training ridge...
Training random_forest...

[SCHRITT 4/4] ERGEBNISSE
📊 Beste Modelle nach R² Score:

DAX Portfolio - DAILY:
  🏆 Bestes Modell: random_forest
  📈 R² Test Score: 0.1234
  ⏱️  Training Time: 12.34s

DAX Portfolio - INTRADAY:
  🏆 Bestes Modell: pytorch_nn
  📈 R² Test Score: 0.2345
  ⏱️  Training Time: 45.67s

SDAX Portfolio - DAILY:
  ...

✅ Erfolgreich abgeschlossen!
📁 Ergebnisse: Results/model_comparison.xlsx
💾 Modelle: Models/dax_daily/, Models/dax_intraday/, ...
```

### Excel-Report (`Results/model_comparison.xlsx`)

Nach der Ausführung wird eine detaillierte Excel-Datei erstellt mit mehreren Sheets:

**Sheet 1: Full_Comparison**
```
Portfolio | Period   | Model        | R2_Test | R2_Train | MSE      | MAE    | Training_Time_s
----------|----------|--------------|---------|----------|----------|--------|----------------
dax       | daily    | pytorch_nn   | 0.6543  | 0.7234   | 0.000234 | 0.0123 | 45.67
dax       | daily    | sklearn_nn   | 0.6321  | 0.7012   | 0.000245 | 0.0129 | 12.34
dax       | daily    | ols          | 0.6123  | 0.6890   | 0.000256 | 0.0131 | 0.12
dax       | daily    | ridge        | 0.6234  | 0.6923   | 0.000251 | 0.0128 | 0.45
dax       | daily    | random_forest| 0.7012  | 0.7654   | 0.000187 | 0.0109 | 89.12
dax       | intraday | pytorch_nn   | 0.7123  | 0.7890   | 0.000165 | 0.0098 | 234.56
...
sdax      | daily    | ...
```

**Sheet 2: R2_by_Portfolio_Period** - Pivot-Tabelle nach R² Score

**Sheet 3: MSE_by_Portfolio_Period** - Pivot-Tabelle nach MSE

**Sheet 4: R2_Hierarchical** - Hierarchische Pivot-Tabelle (Portfolio → Period → Model)

### Gespeicherte Modelle

Falls `output.save_models: true` in config.yaml:

```
Models/
├── dax_daily/
│   ├── pytorch_nn.pt
│   ├── sklearn_nn.pkl
│   ├── ols.pkl
│   ├── ridge.pkl
│   └── random_forest.pkl
├── dax_intraday/
│   └── ... (gleiche Struktur)
├── sdax_daily/
│   └── ...
└── sdax_intraday/
    └── ...
```

### PyTorch Training-Logs

Für PyTorch-Modelle werden zusätzlich CSV-Logs erstellt:
- `Results/pytorch_training_dax_daily.csv`
- `Results/pytorch_training_dax_intraday.csv`
- `Results/pytorch_training_sdax_daily.csv`
- `Results/pytorch_training_sdax_intraday.csv`

Diese enthalten pro Epoch: Train Loss, Val Loss, Learning Rate, etc.

## 🔍 Komponenten im Detail

### 1. ConfigManager.py

Verwaltet die zentrale Konfiguration aus `config.yaml`:

```python
from ConfigManager import ConfigManager

config = ConfigManager("config.yaml")

# Werte lesen
epochs = config.get("models.pytorch_nn.epochs")  # 400
features = config.get("features.input_features")  # ['momentum_5', ...]

# Werte setzen
config.set("models.pytorch_nn.epochs", 500)
config.set("features.input_features", ["momentum_5", "change_dax"])

# Config neu laden
config.reload()
```

**Features:**
- Punkt-Notation für verschachtelte Keys (`"models.pytorch_nn.epochs"`)
- Automatisches Speichern bei `set()`
- Fehlerbehandlung bei fehlenden Dateien

### 2. Datagrabber.py

Holt Daten von LSEG/Refinitiv API:

```python
from Datagrabber import DataGrabber

grabber = DataGrabber("config.yaml")

# Alle Portfolios und Perioden laden
all_data = grabber.fetch_all_data()
# Returns: {"dax": {"daily": df, "intraday": df}, "sdax": {...}}

# Einzelnes Portfolio laden
dax_daily = grabber.fetch_portfolio_data("dax", "daily")
```

**Funktionalität:**
- Lädt Aktien-Daten aus Portfolio-Universe
- Lädt Index-Daten (DAX, SDAX, VDAX)
- Kombiniert alle Daten in einem DataFrame
- Speichert in `DataStorage/` (optional)

### 3. Dataprep.py

Feature Engineering & Datenaufbereitung:

```python
from Dataprep import DataPrep, time_series_split

prep = DataPrep("config.yaml")

# Features erstellen
X, y = prep.prepare_data(df, portfolio_name="dax", period_type="daily")

# Chronologischer Train-Test Split
X_train, X_test, y_train, y_test = time_series_split(
    X, y, test_size=0.2
)
```

**Erstellt automatisch:**
- **Momentum-Features**: 5, 10, 20 Perioden
- **Index-Features**: `change_dax`, `change_sdax` (portfolio-spezifisch)
- **VDAX-Features**: `vdax_absolute`
- **Volume-Features**: `volume_ratio`
- **Volatilitäts-Features**: `rolling_volatility_10`, `rolling_volatility_20`
- **Zeit-Features**: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` (nur Intraday)

**Wichtig:**
- Alle Features basieren auf vergangenen Informationen (kein Look-Ahead Bias)
- Portfolio-spezifisches Feature-Mapping (`portfolio_index_change` → `change_dax`/`change_sdax`)
- Automatische Behandlung fehlender Werte

### 4. Models_Wrapper.py

Vereinfachte Wrapper für alle ML-Modelle:

```python
from Models_Wrapper import (
    train_pytorch_model,
    train_sklearn_nn,
    train_ols,
    train_ridge,
    train_random_forest,
    train_naive_baseline
)

# PyTorch Neural Network
model, metrics = train_pytorch_model(
    X_train, y_train, X_test, y_test,
    hidden1=128, hidden2=64, epochs=400
)

# Sklearn MLP
model, metrics = train_sklearn_nn(
    X_train, y_train, X_test, y_test,
    hidden_layer_sizes=[64, 32]
)

# OLS (Ordinary Least Squares)
model, metrics = train_ols(X_train, y_train, X_test, y_test)

# Ridge Regression
model, metrics = train_ridge(
    X_train, y_train, X_test, y_test,
    alpha_values=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Random Forest
model, metrics = train_random_forest(
    X_train, y_train, X_test, y_test,
    n_estimators=300, max_depth=10
)

# Naive Baseline (Vergleichsmaßstab)
model, metrics = train_naive_baseline(X_train, y_train, X_test, y_test)
```

**Alle Funktionen retournieren:**
- `model`: Trainiertes Modell
- `metrics`: Dictionary mit `{"r2_train": ..., "r2_test": ..., "mse": ..., "mae": ..., "training_time": ...}`

**Besondere Features:**
- **PyTorch**: Early Stopping, Learning Rate Scheduler, Validation Split
- **Sklearn MLP**: GridSearchCV mit TimeSeriesSplit
- **Random Forest**: GridSearchCV mit TimeSeriesSplit
- **Ridge**: Automatische Alpha-Optimierung

### 5. ModelComparison.py

Orchestriert den kompletten Workflow:

```python
from ModelComparison import ModelComparison

comparison = ModelComparison("config.yaml")
comparison.run_full_comparison()
```

**Workflow:**
1. **Datenabruf**: Lädt alle Portfolios und Perioden
2. **Datenaufbereitung**: Feature Engineering für jedes Portfolio/Periode
3. **Skalierung**: Zentrale Skalierung (verhindert Data Leakage)
4. **Modelltraining**: Trainiert alle aktivierten Modelle
5. **Evaluation**: Berechnet Metriken (R², MSE, MAE)
6. **Report**: Erstellt Excel-Report mit allen Ergebnissen

**Wichtig:**
- Skalierung erfolgt **nur auf X_train**, dann auf beide Sets angewendet
- Chronologischer Train-Test Split (kein Shuffle)
- Portfolio-basierte Modell-Speicherung

## 🎓 Wichtige Konzepte

### Portfolio-basiertes Training

Im Gegensatz zu aktienspezifischen Modellen trainieren wir hier **ein Modell pro Portfolio**:

- **Y-Variable**: Durchschnittliche Return aller Aktien im Portfolio
- **Vorteil**: 
  - Robustere Predictions durch mehr Trainingsdaten
  - Weniger Overfitting
  - Berücksichtigt Portfolio-Diversifikation
- **Nachteil**: 
  - Keine aktienspezifischen Predictions
  - Annahme: Aktien im Portfolio verhalten sich ähnlich

**Aktuell implementiert:**
- DAX Portfolio: 6 Large-Cap Aktien
- SDAX Portfolio: 6 Small-Cap Aktien

### Time Series Split

Für Zeitreihen verwenden wir **chronologisches Splitting** (kein Random Shuffle):

```
|←―――――――――― Train (80%) ――――――――→|←― Test (20%) ―→|
|                                 |                |
2024-01-01                   2024-12-31     2025-11-15
```

**Warum chronologisch?**
- Finanzdaten haben Zeitabhängigkeit
- Random Shuffle würde zukünftige Informationen ins Training bringen (Data Leakage)
- Realistischere Evaluation: Modelle werden auf zukünftigen Daten getestet

### Feature Engineering

Alle Features basieren auf **vergangenen Informationen** (Look-Ahead Bias vermeiden):

```python
# ✓ Korrekt: t-1 Information
features['momentum_5'] = prices.pct_change(5).shift(1)  # Vergangene 5 Perioden

# ✗ Falsch: würde zukünftige Info verwenden
features['future_return'] = prices.pct_change().shift(-1)  # Nächste Periode
```

**Implementierte Features:**
- **Momentum**: Prozentuale Änderung über N Perioden
- **Index-Änderung**: DAX/SDAX Return als Marktindikator
- **Volatilität**: Rolling Standard Deviation
- **Volume**: Verhältnis zum Durchschnitt
- **Zeit-Features**: Zyklische Encoding (Sinus/Cosinus) für Stunde/Wochentag

### Data Leakage Prävention

**Zentrale Skalierung:**
```python
# Scaler wird NUR auf X_train gefittet
scaler.fit(X_train)

# Dann auf beide Sets angewendet
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Warum wichtig?**
- Test-Daten dürfen das Training nicht beeinflussen
- Scaler-Parameter (Mittelwert, Std) werden nur aus Trainingsdaten berechnet
- Realistischere Evaluation

### Hyperparameter-Optimierung

**TimeSeriesSplit für CV:**
- Random Forest und Sklearn MLP nutzen GridSearchCV
- TimeSeriesSplit statt KFold (chronologisch)
- Verhindert Data Leakage bei Hyperparameter-Suche

**PyTorch:**
- Early Stopping auf Validation Set
- Learning Rate Scheduler (ReduceLROnPlateau)
- Validation Split aus Trainingsdaten

## 🛠️ Erweiterungen & Anpassungen

### Neue Features hinzufügen

1. **In `config.yaml`:**
   ```yaml
   features:
     input_features:
       - my_new_feature
   ```

2. **In `Dataprep.py` → `create_features()`:**
   ```python
   # Berechne neues Feature (nur vergangene Informationen!)
   if 'my_new_feature' in self.config.get('features.input_features', []):
       features['my_new_feature'] = df['TRDPRC_1'].rolling(20).std().shift(1)
   ```

3. **Testen:**
   ```bash
   python main.py --features momentum_5 my_new_feature
   ```

### Neues Modell hinzufügen

1. **In `Models_Wrapper.py`:**
   ```python
   def train_my_model(X_train, y_train, X_test, y_test, **kwargs):
       """Trainiert mein neues Modell"""
       from sklearn.my_model import MyModel
       
       model = MyModel(**kwargs)
       model.fit(X_train, y_train)
       
       # Predictions
       y_pred_train = model.predict(X_train)
       y_pred_test = model.predict(X_test)
       
       # Metriken
       from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
       metrics = {
           "r2_train": r2_score(y_train, y_pred_train),
           "r2_test": r2_score(y_test, y_pred_test),
           "mse": mean_squared_error(y_test, y_pred_test),
           "mae": mean_absolute_error(y_test, y_pred_test),
           "training_time": training_time
       }
       
       return model, metrics
   ```

2. **In `config.yaml`:**
   ```yaml
   models:
     my_model:
       enabled: true
       param1: value1
       param2: value2
   ```

3. **In `ModelComparison.py` → `train_all_models()`:**
   ```python
   if self.config.get("models.my_model.enabled"):
       from Models_Wrapper import train_my_model
       
       model_kwargs = self.config.get("models.my_model", {})
       model_kwargs.pop("enabled", None)  # Entferne enabled-Flag
       
       model, metrics = train_my_model(
           X_train, y_train, X_test, y_test, **model_kwargs
       )
       
       results["my_model"] = {
           "model": model,
           "metrics": metrics,
           ...
       }
   ```

### Neues Portfolio hinzufügen

1. **In `config.yaml`:**
   ```yaml
   data:
     portfolios:
       my_portfolio:
         name: "Mein Portfolio"
         universe:
           - STOCK1.DE
           - STOCK2.DE
         index: ".MYINDEX"
         index_feature: "change_myindex"
   ```

2. **System lädt automatisch:**
   - Aktien-Daten aus Universe
   - Index-Daten
   - Gemeinsame Indizes (VDAX, etc.)

3. **Training erfolgt automatisch** für alle Perioden

## 📝 Unterschiede zu Version 1 & 2

### vs. Version 1 (BA_firsttry)

| Aspekt | Version 1 | Unified |
|--------|-----------|---------|
| Training | Pro Aktie (ein Modell pro Aktie) | Portfolio-basiert (ein Modell pro Portfolio) |
| Config | Hardcoded im Code | YAML-basiert (flexibel) |
| Features | Fest programmiert | Konfigurierbar über Config |
| Perioden | Nur Daily | Daily + Intraday |
| Vergleich | Manuell | Automatisch (Excel-Report) |
| Portfolios | Ein Portfolio | Mehrere Portfolios (DAX, SDAX) |
| Validierung | Standard Split | TimeSeriesSplit + CV |

### vs. Version 2 (BA_secondtry)

| Aspekt | Version 2 | Unified |
|--------|-----------|---------|
| Basis | Neu implementiert | Basiert auf V1 (Code-Reuse) |
| Modelle | Sklearn-fokussiert | PyTorch + Sklearn (beide) |
| Struktur | Sehr modular | Einfacher, direkter |
| Portfolio | Ein Portfolio | Multi-Portfolio Support |
| Features | Weniger Features | Erweiterte Feature-Liste |

## 🐛 Troubleshooting

### Fehler: "Config-Datei nicht gefunden"

```bash
# Stelle sicher, dass du im richtigen Verzeichnis bist
cd Combination
python main.py

# Oder gib absoluten Pfad an
python main.py --config /absoluter/pfad/config.yaml
```

### Fehler: "LSEG API Connection Failed"

**Mögliche Ursachen:**
- Keine Netzwerkverbindung
- Falsche API-Zugangsdaten
- API-Limits erreicht
- Firewall blockiert Verbindung

**Lösung:**
- Überprüfe `LSEG.py` Konfiguration
- Teste API-Verbindung separat
- Prüfe API-Limits im LSEG-Portal

### Fehler: "Keine Features gefunden"

**Mögliche Ursachen:**
- Feature-Name in Config stimmt nicht mit Implementierung überein
- Spaltennamen im DataFrame sind anders als erwartet
- Portfolio-Name fehlt (für portfolio_index_change)

**Lösung:**
- Prüfe `Dataprep.py` → `create_features()` für verfügbare Features
- Überprüfe Spaltennamen im DataFrame
- Stelle sicher, dass Portfolio-Name korrekt übergeben wird

### Low R² Scores

**Wichtig:** Bei Finanzdaten sind niedrige R²-Werte **normal**!

- **R² < 0.1**: Sehr schwierig zu prognostizieren (normal bei Returns)
- **R² 0.1-0.3**: Akzeptabel für Finanzdaten
- **R² > 0.3**: Sehr gut für Finanzdaten
- **R² > 0.5**: Ausgezeichnet (selten bei Returns)

**Verbesserungsmöglichkeiten:**
- Mehr Features hinzufügen
- Längere Trainingsperiode
- Hyperparameter optimieren
- Andere Modelle testen
- Feature-Engineering verbessern

### PyTorch Training sehr langsam

**Optimierungen:**
- Reduziere `epochs` in Config
- Reduziere `batch_size`
- Deaktiviere Validation Split für schnelleres Training
- Nutze GPU falls verfügbar (automatisch erkannt)

### Memory-Fehler bei großen Datasets

**Lösungen:**
- Reduziere Anzahl Aktien im Portfolio
- Kürze Zeitperiode
- Nutze `--mode daily` statt beide Perioden
- Erhöhe System-RAM

## 📚 Weiterführende Ressourcen

### Dokumentation
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [LSEG Data API](https://developers.lseg.com/)

### Machine Learning für Finanzen
- Time Series Cross-Validation: [sklearn.TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- Feature Engineering für Zeitreihen
- Data Leakage in Machine Learning

### Best Practices
- Reproduzierbarkeit (Seeds, Versionierung)
- Model Evaluation für Zeitreihen
- Hyperparameter-Tuning

## 📊 Beispiel-Ergebnisse

### Typische R²-Werte (Finanzdaten)

| Portfolio | Period | Bestes Modell | R² Test | MSE |
|-----------|--------|---------------|---------|-----|
| DAX | Daily | Random Forest | 0.12-0.25 | 0.0002-0.0004 |
| DAX | Intraday | PyTorch NN | 0.15-0.30 | 0.0001-0.0003 |
| SDAX | Daily | Random Forest | 0.10-0.20 | 0.0003-0.0005 |
| SDAX | Intraday | PyTorch NN | 0.12-0.25 | 0.0002-0.0004 |

**Hinweis:** Diese Werte sind Beispiele. Tatsächliche Ergebnisse hängen von vielen Faktoren ab (Zeitperiode, Features, Hyperparameter, etc.).

## 👥 Credits

- **Version 1**: BA_firsttry - Basis-Implementation
- **Version 2**: BA_secondtry - Professionelle Features
- **Unified Version**: Kombination beider Ansätze mit Portfolio-Support

## 📄 Lizenz

Für akademische Zwecke im Rahmen der Bachelorarbeit.

## 🔄 Changelog

Siehe `CHANGELOG.md` für detaillierte Änderungshistorie.

**Wichtige Updates:**
- **2025-11-16 Part 2**: Portfolio-basiertes System (DAX + SDAX)
- **2025-11-16 Part 1**: Data Leakage Fix & ML Improvements

---

**Viel Erfolg mit deinem Trading System! 🚀📈**

Bei Fragen oder Problemen, siehe Troubleshooting-Sektion oder überprüfe die Code-Kommentare in den einzelnen Modulen.
