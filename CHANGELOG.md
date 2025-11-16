# Changelog - BA Trading System Verbesserungen

## [2025-11-16] - Data Leakage Fix & ML Improvements

### 🔒 Data Leakage Behebung

**Problem:** Doppelte Skalierung führte zu Data Leakage
- Sklearn MLP hatte StandardScaler in Pipeline, während bereits in ModelComparison.py skaliert wurde

**Lösung:**
- Entfernt StandardScaler aus `train_sklearn_nn()` Pipeline in `Models_Wrapper.py:141`
- Zentrale Skalierung erfolgt nur noch in `ModelComparison.py:69-87`
- Scaler wird **nur auf X_train gefittet**, dann auf beide Sets angewendet
- Klarstellende Kommentare hinzugefügt

### ⚙️ Zeitreihen-gerechte Hyperparameter-Optimierung

**Random Forest (`Models_Wrapper.py:257-339`):**
- GridSearchCV mit TimeSeriesSplit implementiert
- Parameter-Grid: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
- Nutzt `training.cross_validation.n_splits` aus Config (default: 5)
- Optional: `use_gridsearch` Flag zum Deaktivieren

**Sklearn MLP (`Models_Wrapper.py:125-225`):**
- GridSearchCV mit TimeSeriesSplit implementiert
- Parameter-Grid: hidden_layer_sizes, alpha, learning_rate_init
- Kleinerer Grid wegen längerer Trainingszeit
- Nutzt gleiche TimeSeriesSplit-Konfiguration

### 🧠 PyTorch-Modell Verbesserungen (`Models_Wrapper.py:50-190`)

**Neue Features:**
- **Validierungs-Split:** Chronologischer Split (letzten 20% des Trainingssets)
- **Early Stopping:** Stoppt nach 20 Epochen ohne Verbesserung
- **Seeds:** Reproduzierbarkeit durch `np.random.seed(42)`, `torch.manual_seed(42)`
- **Learning Rate Scheduler:** ReduceLROnPlateau (optional, default: True)
- **Bessere Ausgaben:** Zeigt Train Loss, Val Loss und LR während Training

**Konfiguration:**
- `validation_split`: Anteil für Validierung (default: 0.2)
- `early_stopping_patience`: Geduld in Epochen (default: 20)
- `use_scheduler`: Learning Rate Scheduler (default: True)

### 🛡️ Robustheit & Qualitätssicherung

**time_series_split Validierung (`Dataprep.py:192-244`):**
- Prüft minimale Trainingsset-Größe (min_train_size: 50)
- Prüft minimale Testset-Größe (min_test_size: 10)
- Warnt bei sehr kleinen Datasets (< 100 Samples)
- Wirft aussagekräftige Exceptions bei ungültigen Splits

**Baseline-Modell (`Models_Wrapper.py:473-502`):**
- Naive Predictor: y_pred[t] = y[t-1]
- Dient als Vergleichsmaßstab
- Wird automatisch in ModelComparison trainiert
- ML-Modelle sollten Baseline übertreffen

### 📈 SDAX Index Integration

**Config (`config.yaml:12-15, 37-43`):**
- SDAX Index hinzugefügt: `.SDAXI`
- Feature `change_sdax` zu input_features hinzugefügt
- Dokumentation aktualisiert (Large Cap vs Small Cap)

**DataGrabber (`Datagrabber.py:84`):**
- Lädt automatisch alle Indizes aus Config
- Kommentar aktualisiert: "DAX, SDAX, VDAX"

**Feature Engineering (`Dataprep.py:88-114`):**
- `change_sdax`: Prozentuale Änderung des SDAX
- Analog zu `change_dax` implementiert
- Fallback auf 0.0 falls SDAX-Daten fehlen
- Sucht nach Spalten mit "SDAXI" oder "SDAX"

### 📊 Verbesserungen im Modellvergleich

**ModelComparison.py:**
- Baseline-Modell wird als erstes trainiert
- Bessere Ausgabe mit Hinweis auf Baseline-Zweck
- Importiert `train_naive_baseline`

### ✅ Zusammenfassung

**Was wurde behoben:**
- ✅ Data Leakage beim Scaling vollständig eliminiert
- ✅ Zeitreihen-gerechte CV für Random Forest und MLP
- ✅ PyTorch robuster mit Early Stopping und Validation Split
- ✅ Validierung der Split-Größen
- ✅ Baseline-Modell für bessere Evaluation

**Was wurde hinzugefügt:**
- ✅ SDAX Index als zusätzliches Feature
- ✅ change_sdax Feature für Small Cap Marktdynamik
- ✅ Bessere Reproduzierbarkeit durch Seeds
- ✅ Umfassende Dokumentation in Kommentaren

**Erwartete Verbesserungen:**
- 📈 Bessere Modellgüte durch optimierte Hyperparameter
- 🔍 Validere Evaluation durch korrekte Skalierung
- 📊 Mehr Marktinformationen durch SDAX
- 🎯 Klarere Baseline zum Vergleich
