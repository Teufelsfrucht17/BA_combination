# BA Trading System - Unified Version

Eine vereinheitlichte Version des BA_combination Trading-Systems, die die Stärken beider Versionen kombiniert.

## 🎯 Projektziel

Dieses System kombiniert:
- **Version 1 (BA_firsttry)**: Basis-Implementation mit selbst entwickelten Komponenten
- **Version 2 (BA_secondtry)**: Professionelle Features und Struktur

**Hauptfeatures:**
- Portfolio-basiertes Machine Learning (alle Aktien zusammen)
- Vergleich von Daily vs. Intraday (30-Min) Daten
- 5 verschiedene ML-Modelle: PyTorch NN, Sklearn NN, OLS, Ridge, Random Forest
- Konfigurierbare Features über `config.yaml`
- Automatischer Modellvergleich mit Excel-Reports

## 📁 Projektstruktur

```
Combination/
├── config.yaml                 # Zentrale Konfiguration
├── DataStorage/               # Excel/CSV Dateien
├── Models/                    # Gespeicherte Modelle
│   ├── daily/                # Modelle für tägliche Daten
│   └── intraday/             # Modelle für 30-Min Daten
├── Results/                   # Vergleichsergebnisse
├── Logs/                      # Log-Dateien
│
├── main.py                    # ⭐ Hauptprogramm
├── ConfigManager.py           # Config-System
├── Datagrabber.py            # Datenabruf (LSEG/Refinitiv)
├── Dataprep.py               # Datenaufbereitung & Feature Engineering
├── Models_Wrapper.py         # ML-Modell Wrapper
├── ModelComparison.py        # Modellvergleich & Evaluation
│
├── LSEG.py                   # LSEG API Interface
├── GloablVariableStorage.py  # Globale Variablen
│
└── requirements.txt          # Dependencies
```

## 🚀 Installation

### 1. Python-Umgebung erstellen

```bash
# Mit venv
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
- Entsprechende API-Konfiguration

## ⚙️ Konfiguration

Die zentrale Konfiguration erfolgt über `config.yaml`:

### Aktien-Portfolio anpassen

```yaml
data:
  universe:
    - "SAP.DE"
    - "SIE.DE"
    - "ALV.DE"
    # ... weitere Aktien
```

### Features konfigurieren

```yaml
features:
  input_features:
    - momentum_5      # 5-Perioden Momentum
    - momentum_10     # 10-Perioden Momentum
    - change_dax      # DAX Änderung
    - vdax_absolute   # VDAX absoluter Wert
    - volume_ratio    # Volume Verhältnis
    # - rsi_14        # Optional: RSI
```

### Modelle aktivieren/deaktivieren

```yaml
models:
  pytorch_nn:
    enabled: true
    hidden1: 64
    hidden2: 32
    epochs: 200

  sklearn_nn:
    enabled: true

  ols:
    enabled: true

  ridge:
    enabled: true

  random_forest:
    enabled: true
```

## 📊 Verwendung

### Standard-Ausführung

```bash
# Vergleiche beide Perioden (Daily + Intraday), alle Modelle
python main.py
```

### Nur bestimmte Periode

```bash
# Nur tägliche Daten
python main.py --mode daily

# Nur 30-Minuten Daten
python main.py --mode intraday
```

### Custom Features

```bash
# Nur bestimmte Features verwenden
python main.py --features momentum_5 change_dax vdax_absolute
```

### Nur bestimmte Modelle

```bash
# Nur Neural Networks trainieren
python main.py --models pytorch_nn sklearn_nn

# Nur OLS und Ridge
python main.py --models ols ridge
```

### Custom Config-Datei

```bash
python main.py --config my_custom_config.yaml
```

### Alle Optionen kombinieren

```bash
python main.py \
  --mode daily \
  --features momentum_5 momentum_10 change_dax \
  --models random_forest pytorch_nn \
  --no-save
```

## 📈 Ausgabe & Ergebnisse

### Während der Ausführung

```
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║            BA TRADING SYSTEM - UNIFIED VERSION                       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

======================================================================
KONFIGURATION
======================================================================
Config-Datei:     config.yaml
Modus:            compare
Portfolio Größe:  10 Aktien
Features:         ['momentum_5', 'momentum_10', 'change_dax', ...]
Aktive Modelle:   ['pytorch_nn', 'sklearn_nn', 'ols', 'ridge', ...]
======================================================================

[SCHRITT 1/4] DATENABRUF
...
```

### Excel-Report

Nach der Ausführung wird `Results/model_comparison.xlsx` erstellt mit:

**Sheet 1: Full_Comparison**
```
Period    | Model        | R2_Test | R2_Train | MSE      | MAE    | Training_Time_s
----------|--------------|---------|----------|----------|--------|----------------
daily     | pytorch_nn   | 0.6543  | 0.7234   | 0.000234 | 0.0123 | 45.67
daily     | sklearn_nn   | 0.6321  | 0.7012   | 0.000245 | 0.0129 | 12.34
...
intraday  | random_forest| 0.7012  | 0.7654   | 0.000187 | 0.0109 | 89.12
```

**Sheet 2: R2_Comparison** - Pivot-Tabelle nach R² Score

**Sheet 3: MSE_Comparison** - Pivot-Tabelle nach MSE

### Gespeicherte Modelle

Falls `output.save_models: true` in config.yaml:

```
Models/
├── daily/
│   ├── pytorch_nn.pt
│   ├── sklearn_nn.pkl
│   ├── ols.pkl
│   ├── ridge.pkl
│   └── random_forest.pkl
└── intraday/
    └── ... (gleiche Struktur)
```

## 🔍 Komponenten im Detail

### 1. ConfigManager.py

Verwaltet die zentrale Konfiguration:

```python
from ConfigManager import ConfigManager

config = ConfigManager("config.yaml")
print(config.get("models.pytorch_nn.epochs"))  # 200
config.set("models.pytorch_nn.epochs", 300)
```

### 2. Datagrabber.py

Holt Daten von LSEG/Refinitiv:

```python
from Datagrabber import DataGrabber

grabber = DataGrabber()
daily_data, intraday_data = grabber.fetch_all_data()
```

### 3. Dataprep.py

Feature Engineering & Datenaufbereitung:

```python
from Dataprep import DataPrep

prep = DataPrep()
X, y = prep.prepare_data(daily_data, "daily")
```

**Erstellt automatisch:**
- Momentum-Features (5, 10, 20 Perioden)
- DAX/VDAX Indikatoren
- Volume-Ratios
- Optional: RSI, weitere technische Indikatoren

### 4. Models_Wrapper.py

Vereinfachte Wrapper für alle ML-Modelle:

```python
from Models_Wrapper import train_pytorch_model, train_ols, train_ridge

# PyTorch NN
model, metrics = train_pytorch_model(X_train, y_train, X_test, y_test)

# OLS
model, metrics = train_ols(X_train, y_train, X_test, y_test)
```

### 5. ModelComparison.py

Orchestriert den kompletten Workflow:

```python
from ModelComparison import ModelComparison

comparison = ModelComparison()
comparison.run_full_comparison()
```

## 🎓 Wichtige Konzepte

### Portfolio-basiertes Training

Im Gegensatz zu Version 1 (ein Modell pro Aktie) trainieren wir hier **ein Modell für das gesamte Portfolio**:

- **Y-Variable**: Durchschnittliche Return aller Aktien
- **Vorteil**: Robustere Predictions, weniger Overfitting
- **Nachteil**: Keine aktienspezifischen Predictions

### Time Series Split

Für Zeitreihen verwenden wir **chronologisches Splitting** (kein Random Shuffle):

```
|←―――――――――― Train (80%) ――――――――→|←― Test (20%) ―→|
|                                 |                |
2015-01-01                   2023-12-31     2025-11-15
```

### Feature Engineering

Alle Features basieren auf **vergangenen Informationen** (Look-Ahead Bias vermeiden):

```python
# ✓ Korrekt: t-1 Information
features['momentum_5'] = prices.pct_change(5)

# ✗ Falsch: würde zukünftige Info verwenden
features['future_return'] = prices.pct_change().shift(-1)
```

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
   # Berechne neues Feature
   features['my_new_feature'] = df['TRDPRC_1'].rolling(20).std()
   ```

### Neues Modell hinzufügen

1. **In `Models_Wrapper.py`:**
   ```python
   def train_my_model(X_train, y_train, X_test, y_test, **kwargs):
       model = MyModel(**kwargs)
       model.fit(X_train, y_train)
       # ... predictions & metrics
       return model, metrics
   ```

2. **In `config.yaml`:**
   ```yaml
   models:
     my_model:
       enabled: true
       param1: value1
   ```

3. **In `ModelComparison.py` → `train_all_models()`:**
   ```python
   if self.config.get("models.my_model.enabled"):
       model, metrics = train_my_model(X_train, y_train, X_test, y_test)
       results["my_model"] = {"model": model, "metrics": metrics, ...}
   ```

## 📝 Unterschiede zu Version 1 & 2

### vs. Version 1 (BA_firsttry)

| Aspekt | Version 1 | Unified |
|--------|-----------|---------|
| Training | Pro Aktie | Portfolio-basiert |
| Config | Hardcoded | YAML-basiert |
| Features | Fest | Konfigurierbar |
| Perioden | Nur Daily | Daily + Intraday |
| Vergleich | Manuell | Automatisch |

### vs. Version 2 (BA_secondtry)

| Aspekt | Version 2 | Unified |
|--------|-----------|---------|
| Basis | Neu implementiert | Basiert auf V1 |
| Modelle | Sklearn-fokussiert | PyTorch + Sklearn |
| Struktur | Sehr modular | Einfacher |
| Code-Reuse | Wenig | Maximal (aus V1) |

## 🐛 Troubleshooting

### Fehler: "Config-Datei nicht gefunden"

```bash
# Stelle sicher, dass du im richtigen Verzeichnis bist
cd Combination
python main.py
```

### Fehler: "LSEG API Connection Failed"

- Überprüfe LSEG/Refinitiv Zugangsdaten
- Stelle sicher, dass du mit dem Netzwerk verbunden bist
- Prüfe API-Limits

### Fehler: "Keine Features gefunden"

- Stelle sicher, dass Features in `config.yaml` auch in `Dataprep.py` implementiert sind
- Prüfe, ob Spaltennamen im DataFrame korrekt sind

### Low R² Scores

- **Normal bei Financial Data!** R² von 0.3-0.5 ist bereits gut
- Versuche mehr Features hinzuzufügen
- Verlängere Trainingsperiode
- Experimentiere mit Hyperparametern

## 📚 Weiterführende Ressourcen

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [LSEG Data API](https://developers.lseg.com/)
- Time Series Cross-Validation: [sklearn.TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

## 👥 Credits

- **Version 1**: BA_firsttry - Basis-Implementation
- **Version 2**: BA_secondtry - Professionelle Features
- **Unified Version**: Kombination beider Ansätze

## 📄 Lizenz

Für akademische Zwecke im Rahmen der Bachelorarbeit.

---

**Viel Erfolg mit deinem Trading System! 🚀📈**
