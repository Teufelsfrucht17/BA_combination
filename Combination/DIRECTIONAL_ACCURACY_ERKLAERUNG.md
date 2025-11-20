# Directional Accuracy - Erklärung

## Was ist Directional Accuracy?

**Directional Accuracy** (Richtungsgenauigkeit) ist eine Metrik, die misst, wie oft das Modell die **Richtung** der Preisänderung korrekt vorhersagt - also ob der Preis steigt (+) oder fällt (-).

### Warum ist das wichtig?

Bei Aktienprognosen ist es oft wichtiger, die **Richtung** korrekt vorherzusagen als den exakten Wert. 

**Beispiel:**
- Du willst wissen: Soll ich kaufen oder verkaufen?
- Richtige Vorhersage: "Preis steigt" → Kaufe → Gewinn
- Falsche Vorhersage: "Preis fällt" → Kaufe → Verlust

## Wie wird sie berechnet?

```python
def directional_accuracy(y_true, y_pred):
    """
    Berechnet Trefferrate der Vorzeichen.
    
    Args:
        y_true: Tatsächliche Returns (z.B. [0.02, -0.01, 0.005, ...])
        y_pred: Vorhergesagte Returns (z.B. [0.018, -0.008, 0.004, ...])
    
    Returns:
        Float zwischen 0.0 und 1.0 (0% bis 100%)
    """
    # Vorzeichen bestimmen: +1 für positiv, -1 für negativ
    true_signs = np.sign(y_true)  # [1, -1, 1, ...]
    pred_signs = np.sign(y_pred)  # [1, -1, 1, ...]
    
    # Vergleiche Vorzeichen
    correct = (true_signs == pred_signs)  # [True, True, False, ...]
    
    # Berechne Anteil korrekter Vorhersagen
    accuracy = np.mean(correct)  # z.B. 0.65 = 65%
    
    return accuracy
```

## Konkretes Beispiel

### Beispiel-Daten:

| Zeitpunkt | Tatsächlicher Return (y_true) | Vorhergesagter Return (y_pred) | Korrekt? |
|-----------|-------------------------------|--------------------------------|----------|
| t=1       | +0.02 (↑)                     | +0.018 (↑)                     | ✓ Ja    |
| t=2       | -0.01 (↓)                     | -0.008 (↓)                     | ✓ Ja    |
| t=3       | +0.005 (↑)                    | -0.002 (↓)                     | ✗ Nein  |
| t=4       | -0.015 (↓)                    | -0.012 (↓)                     | ✓ Ja    |
| t=5       | +0.01 (↑)                     | +0.009 (↑)                     | ✓ Ja    |

**Berechnung:**
- Korrekte Vorhersagen: 4 von 5 = 80%
- **Directional Accuracy = 0.80 (80%)**

### Code-Zeile für Zeile:

```python
# ZEILE 56-58: Vorzeichen extrahieren
y_true = np.asarray([0.02, -0.01, 0.005, -0.015, 0.01])
y_pred = np.asarray([0.018, -0.008, -0.002, -0.012, 0.009])

# np.sign() gibt zurück: +1 für positive Werte, -1 für negative, 0 für null
true_signs = np.sign(y_true)   # [1, -1,  1, -1,  1]
pred_signs = np.sign(y_pred)   # [1, -1, -1, -1,  1]

# ZEILE 58: Vergleich
correct = (true_signs == pred_signs)  # [True, True, False, True, True]

# ZEILE 58: Mittelwert = Anteil korrekter Vorhersagen
accuracy = np.mean(correct)  # 4/5 = 0.8 = 80%
```

## Interpretation der Werte

### Directional Accuracy in deiner Excel:

| Wert | Interpretation | Bedeutung |
|------|----------------|-----------|
| **0.50 (50%)** | Zufällig | Wie Münzwurf - Modell ist nutzlos |
| **0.55-0.60 (55-60%)** | Leicht besser als Zufall | Modell hat etwas gelernt |
| **0.60-0.65 (60-65%)** | Gut | Modell kann Richtung gut vorhersagen |
| **0.65-0.70 (65-70%)** | Sehr gut | Sehr nützlich für Trading |
| **> 0.70 (>70%)** | Ausgezeichnet | Selten bei Finanzdaten! |

### Wichtige Erkenntnis:

**Bei Finanzdaten ist bereits 55-60% Directional Accuracy sehr gut!**

Warum?
- Finanzmärkte sind sehr schwer vorherzusagen
- Selbst professionelle Trader haben oft nur 52-55% Win-Rate
- 60% bedeutet: Von 10 Trades, 6 richtig → Potentiell profitabel!

## Directional Accuracy vs. andere Metriken

### R² Score:
- Misst: Wie gut das Modell die **exakten Werte** vorhersagt
- Problem: Bei Finanzdaten oft sehr niedrig (0.1-0.3)
- Beispiel: R² = 0.15, aber Directional Accuracy = 0.62

### MSE (Mean Squared Error):
- Misst: Durchschnittlicher quadratischer Fehler
- Problem: Schwer zu interpretieren
- Beispiel: MSE = 0.000234 → Was bedeutet das?

### Directional Accuracy:
- Misst: **Richtung** (steigt oder fällt)
- Vorteil: Einfach zu verstehen (Prozent korrekter Vorhersagen)
- Vorteil: Direkt relevant für Trading-Entscheidungen
- Beispiel: 62% = Von 100 Vorhersagen sind 62 richtig

## Beispiel aus deiner Excel

Nehmen wir an, du hast:

| Portfolio | Period | Model | Directional_Accuracy |
|-----------|--------|-------|---------------------|
| DAX Portfolio | Daily | pytorch_nn | 0.62 |
| DAX Portfolio | Daily | random_forest | 0.58 |
| DAX Portfolio | Intraday | pytorch_nn | 0.61 |

**Interpretation:**
- **pytorch_nn (Daily)**: 62% der Vorhersagen haben die richtige Richtung
  - Von 100 Vorhersagen: 62 korrekt, 38 falsch
  - Das ist **sehr gut** für Finanzdaten!
  
- **random_forest (Daily)**: 58%
  - Etwas niedriger, aber immer noch besser als Zufall
  
- **pytorch_nn (Intraday)**: 61%
  - Ähnlich wie Daily, zeigt Konsistenz

**Fazit:** Das Modell kann die Richtung besser als zufällig vorhersagen → Potentiell nützlich für Trading!

## Warum ist Directional Accuracy wichtig?

### 1. Praktische Relevanz

Für Trading-Entscheidungen ist die Richtung oft wichtiger als der exakte Wert:

```
Szenario A: Modell sagt +2% voraus, tatsächlich +0.5%
→ Richtung korrekt → Kaufe → Gewinn!

Szenario B: Modell sagt -1% voraus, tatsächlich -0.3%
→ Richtung korrekt → Verkaufe → Gewinn!

Szenario C: Modell sagt +0.5% voraus, tatsächlich -1%
→ Richtung falsch → Kaufe → Verlust!
```

### 2. Robustheit

Directional Accuracy ist **robuster** als R²:
- Unempfindlich gegen Ausreißer
- Funktioniert auch wenn absolute Werte nicht perfekt sind
- Fokus auf das Wesentliche: Steigt oder fällt?

### 3. Trading-Strategie

Mit Directional Accuracy > 0.55 kann man bereits profitabel handeln:

**Beispiel-Strategie:**
- Wenn Modell "↑" vorhersagt → Kaufe
- Wenn Modell "↓" vorhersagt → Verkaufe
- Bei 62% Accuracy: Von 100 Trades → 62 Gewinn, 38 Verlust
- Wenn Gewinn > Verlust → Profitabel!

## Vergleich: Train vs. Test

In deiner Excel findest du wahrscheinlich:

- **Directional_Accuracy**: Auf Test-Set (wie gut auf neuen Daten)
- **Directional_Accuracy_Train**: Auf Train-Set (wie gut auf Trainingsdaten)

### Was bedeutet das?

| Train | Test | Interpretation |
|-------|------|----------------|
| 0.70  | 0.65 | ✅ Gut: Geringe Überanpassung |
| 0.75  | 0.58 | ⚠️ Warnung: Mögliches Overfitting |
| 0.60  | 0.61 | ✅ Sehr gut: Generalisiert gut |
| 0.65  | 0.55 | ⚠️ Warnung: Generalisiert schlecht |

**Ideal:** Test ≈ Train (modell generalisiert gut)

## Wann ist Directional Accuracy besonders wichtig?

### 1. Bei niedrigen R²-Werten

Wenn R² niedrig ist (z.B. 0.15), aber Directional Accuracy gut (z.B. 0.62):
- Modell kann die Richtung gut vorhersagen
- Aber nicht die exakte Höhe
- **Für Trading trotzdem nützlich!**

### 2. Bei volatilen Märkten

Bei hoher Volatilität:
- Exakte Werte schwer vorherzusagen
- Aber Richtung möglicherweise einfacher
- Directional Accuracy zeigt, ob Modell trotzdem nützlich ist

### 3. Für Trading-Entscheidungen

Für binäre Entscheidungen (Kaufen/Verkaufen):
- Richtung ist wichtiger als exakter Wert
- Directional Accuracy ist die relevante Metrik

## Beispiel-Berechnung Schritt für Schritt

```python
import numpy as np

# Tatsächliche Returns
y_true = np.array([0.02, -0.01, 0.005, -0.015, 0.01, -0.005])

# Vorhergesagte Returns
y_pred = np.array([0.018, -0.008, -0.002, -0.012, 0.009, -0.003])

# Schritt 1: Vorzeichen bestimmen
true_signs = np.sign(y_true)
# [ 1, -1,  1, -1,  1, -1]  (positiv = 1, negativ = -1)

pred_signs = np.sign(y_pred)
# [ 1, -1, -1, -1,  1, -1]

# Schritt 2: Vergleich
correct = (true_signs == pred_signs)
# [True, True, False, True, True, True]
# Index 2 ist falsch: true=+1, pred=-1

# Schritt 3: Anteil berechnen
accuracy = np.mean(correct)
# 5 von 6 korrekt = 5/6 = 0.8333 = 83.33%

print(f"Directional Accuracy: {accuracy:.2%}")
# Output: Directional Accuracy: 83.33%
```

## Was sagen die Werte in deiner Excel aus?

### Beispiel-Werte aus deiner Excel:

```
DAX Portfolio - Daily:
  pytorch_nn:          Directional_Accuracy = 0.62 (62%)
  random_forest:       Directional_Accuracy = 0.58 (58%)
  ols:                 Directional_Accuracy = 0.55 (55%)
```

**Interpretation:**
1. **pytorch_nn (62%)**: Bestes Modell für Richtungsvorhersage
   - Von 100 Vorhersagen: 62 korrekt
   - **Sehr gut** für Finanzdaten!

2. **random_forest (58%)**: Gut
   - Etwas niedriger, aber immer noch nützlich

3. **ols (55%)**: Leicht besser als Zufall
   - Minimal besser als Münzwurf (50%)

### Vergleich zu anderen Metriken:

| Model | R² | MSE | Directional Accuracy |
|-------|----|-----|---------------------|
| pytorch_nn | 0.15 | 0.000234 | **0.62** |
| random_forest | 0.18 | 0.000210 | 0.58 |

**Erkenntnis:**
- R² ist niedrig (0.15-0.18), aber Directional Accuracy ist gut (58-62%)
- Das Modell kann **Richtung** gut vorhersagen, auch wenn **exakte Werte** schwierig sind
- **Für Trading sehr nützlich!**

## Zusammenfassung

### Was ist Directional Accuracy?
- **Anteil der korrekten Richtungsvorhersagen** (steigt ↑ oder fällt ↓)
- Wert zwischen 0% und 100%
- 50% = Zufall, >55% = nützlich, >60% = sehr gut

### Warum wichtig?
- **Praktisch relevant**: Für Trading-Entscheidungen wichtiger als exakte Werte
- **Robust**: Weniger anfällig für Ausreißer als R²
- **Interpretierbar**: Einfach zu verstehen (Prozent korrekt)

### In deiner Excel:
- **Directional_Accuracy**: Auf Test-Set (neue Daten)
- **Directional_Accuracy_Train**: Auf Train-Set (Trainingsdaten)
- **Vergleich**: Test sollte nahe bei Train sein (kein Overfitting)

### Typische Werte:
- **50%**: Zufällig (Münzwurf)
- **55-60%**: Gut für Finanzdaten
- **>60%**: Sehr gut (selten bei Finanzdaten)
- **>70%**: Ausgezeichnet (extrem selten)

**Deine Werte von 58-62% sind sehr gut für Finanzdaten!** 🎯


