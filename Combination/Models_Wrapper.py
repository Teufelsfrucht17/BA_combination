"""
Models_Wrapper.py - Vereinfachte Wrapper für alle Modelle
Diese Funktionen sind vereinfacht im Vergleich zu Version 1,
da wir jetzt Portfolio-basiert trainieren (alle Aktien zusammen)
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path

# Sklearn Modelle
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================
# PyTorch Neural Network
# ============================================

class SimpleNet(nn.Module):
    """Einfaches MLP mit zwei Hidden Layers"""

    def __init__(self, in_features: int, hidden1: int = 64, hidden2: int = 32, out_features: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_pytorch_model(X_train, y_train, X_test, y_test,
                       hidden1=64, hidden2=32, epochs=200,
                       batch_size=64, lr=0.001, validation_split=0.2,
                       early_stopping_patience=20, use_scheduler=True):
    """
    Trainiert PyTorch Neural Network mit Validation Split und Early Stopping

    Args:
        X_train, y_train, X_test, y_test: Train/Test Splits
        hidden1: Größe des ersten Hidden Layers
        hidden2: Größe des zweiten Hidden Layers
        epochs: Maximale Anzahl Epochen
        batch_size: Batch-Größe
        lr: Learning Rate
        validation_split: Anteil des Trainingssets für Validierung
        early_stopping_patience: Anzahl Epochen ohne Verbesserung vor Abbruch
        use_scheduler: Ob Learning Rate Scheduler verwendet werden soll

    Returns:
        Tuple von (model, metrics)
    """
    # Seeds für Reproduzierbarkeit
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Konvertiere zu Tensoren
    X_train_t = torch.tensor(X_train.values if isinstance(X_train, pd.DataFrame) else X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values if isinstance(y_train, pd.Series) else y_train, dtype=torch.float32).reshape(-1, 1)
    X_test_t = torch.tensor(X_test.values if isinstance(X_test, pd.DataFrame) else X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values if isinstance(y_test, pd.Series) else y_test, dtype=torch.float32).reshape(-1, 1)

    # Interner Validierungs-Split (chronologisch, letzten X% des Trainingssets)
    n_train = len(X_train_t)
    val_idx = int(n_train * (1 - validation_split))

    X_train_inner = X_train_t[:val_idx]
    y_train_inner = y_train_t[:val_idx]
    X_val = X_train_t[val_idx:]
    y_val = y_train_t[val_idx:]

    print(f"    Inner Train Size: {len(X_train_inner)}, Val Size: {len(X_val)}")

    # DataLoader (kein Shuffle für Zeitreihen!)
    train_dataset = TensorDataset(X_train_inner, y_train_inner)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Modell
    n_features = X_train_t.shape[1]
    model = SimpleNet(in_features=n_features, hidden1=hidden1, hidden2=hidden2, out_features=1).to(device)

    # Optimizer und Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Optional: Learning Rate Scheduler
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

    # Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        # Training
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.to(device))
            val_loss = criterion(val_outputs, y_val.to(device)).item()
        model.train()

        # Learning Rate Scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # Print Progress
        if (epoch + 1) % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")

        # Early Stopping
        if patience_counter >= early_stopping_patience:
            print(f"    Early Stopping bei Epoch {epoch+1} (Best Val Loss: {best_val_loss:.6f})")
            break

    # Lade bestes Modell
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Finale Evaluation
    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train_t.to(device)).cpu().numpy().flatten()
        y_test_pred = model(X_test_t.to(device)).cpu().numpy().flatten()

    metrics = {
        'r2': r2_score(y_test, y_test_pred),
        'mse': mean_squared_error(y_test, y_test_pred),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'best_val_loss': best_val_loss,
        'stopped_at_epoch': epoch + 1
    }

    return model, metrics


# ============================================
# Sklearn Neural Network
# ============================================

def train_sklearn_nn(X_train, y_train, X_test, y_test,
                     hidden_layer_sizes=(64, 32), max_iter=1000,
                     n_splits=5, use_gridsearch=True):
    """
    Trainiert Sklearn MLP Regressor mit optionalem Hyperparameter-Tuning

    WICHTIG: Skalierung erfolgt bereits in ModelComparison.py!
    Daher KEINE zusätzliche Skalierung hier, um Data Leakage zu vermeiden.

    Args:
        X_train, y_train, X_test, y_test: Train/Test Splits (bereits skaliert!)
        hidden_layer_sizes: Tuple mit Layer-Größen (wird für GridSearch überschrieben)
        max_iter: Maximale Iterationen
        n_splits: Anzahl Splits für TimeSeriesSplit
        use_gridsearch: Ob GridSearch verwendet werden soll

    Returns:
        Tuple von (model, metrics)
    """
    if use_gridsearch:
        # Hyperparameter-Grid für Suche (kleiner Grid für MLP wegen längerer Trainingszeit)
        param_grid = {
            'hidden_layer_sizes': [(32,), (64,), (64, 32), (128, 64), (64, 32, 16)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        }

        mlp = MLPRegressor(
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=25,
            random_state=42,
            verbose=False
        )

        tscv = TimeSeriesSplit(n_splits=n_splits)

        print(f"    Starte GridSearch mit {n_splits} TimeSeriesSplit-Folds...")

        grid = GridSearchCV(
            mlp,
            param_grid=param_grid,
            cv=tscv,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )

        grid.fit(X_train, y_train)

        print(f"    Beste Parameter: {grid.best_params_}")
        print(f"    Bester CV R² Score: {grid.best_score_:.4f}")

        model = grid.best_estimator_

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            'r2': r2_score(y_test, y_test_pred),
            'mse': mean_squared_error(y_test, y_test_pred),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'best_params': grid.best_params_,
            'cv_best_score': grid.best_score_
        }

    else:
        # Modell OHNE Scaling-Pipeline (Daten sind bereits skaliert)
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            alpha=0.0001,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=25,
            random_state=42,
            verbose=False
        )

        # Trainiere
        model.fit(X_train, y_train)

        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            'r2': r2_score(y_test, y_test_pred),
            'mse': mean_squared_error(y_test, y_test_pred),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
        }

    return model, metrics


# ============================================
# OLS Regression
# ============================================

def train_ols(X_train, y_train, X_test, y_test):
    """
    Trainiert OLS (Ordinary Least Squares) Linear Regression

    Args:
        X_train, y_train, X_test, y_test: Train/Test Splits

    Returns:
        Tuple von (model, metrics)
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        'r2': r2_score(y_test, y_test_pred),
        'mse': mean_squared_error(y_test, y_test_pred),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
    }

    return model, metrics


# ============================================
# Ridge Regression
# ============================================

def train_ridge(X_train, y_train, X_test, y_test, alpha_values=None):
    """
    Trainiert Ridge Regression mit GridSearch

    Args:
        X_train, y_train, X_test, y_test: Train/Test Splits
        alpha_values: Liste von Alpha-Werten für GridSearch

    Returns:
        Tuple von (model, metrics)
    """
    if alpha_values is None:
        alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    param_grid = {
        'alpha': alpha_values,
        'fit_intercept': [True, False]
    }

    ridge = Ridge()
    tscv = TimeSeriesSplit(n_splits=5)

    grid = GridSearchCV(
        ridge,
        param_grid=param_grid,
        cv=tscv,
        scoring='r2',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    y_train_pred = grid.predict(X_train)
    y_test_pred = grid.predict(X_test)

    metrics = {
        'r2': r2_score(y_test, y_test_pred),
        'mse': mean_squared_error(y_test, y_test_pred),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'best_alpha': grid.best_params_['alpha']
    }

    return grid.best_estimator_, metrics


# ============================================
# Random Forest
# ============================================

def train_random_forest(X_train, y_train, X_test, y_test,
                       n_estimators=300, max_depth=10, min_samples_split=5,
                       n_splits=5, use_gridsearch=True):
    """
    Trainiert Random Forest Regressor mit Hyperparameter-Tuning

    Args:
        X_train, y_train, X_test, y_test: Train/Test Splits
        n_estimators: Anzahl Bäume (wird für GridSearch überschrieben)
        max_depth: Maximale Tiefe (wird für GridSearch überschrieben)
        min_samples_split: Minimale Samples für Split (wird für GridSearch überschrieben)
        n_splits: Anzahl Splits für TimeSeriesSplit
        use_gridsearch: Ob GridSearch verwendet werden soll

    Returns:
        Tuple von (model, metrics)
    """
    if use_gridsearch:
        # Hyperparameter-Grid für Suche
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        print(f"    Starte GridSearch mit {n_splits} TimeSeriesSplit-Folds...")

        grid = GridSearchCV(
            rf,
            param_grid=param_grid,
            cv=tscv,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )

        grid.fit(X_train, y_train)

        print(f"    Beste Parameter: {grid.best_params_}")
        print(f"    Bester CV R² Score: {grid.best_score_:.4f}")

        model = grid.best_estimator_

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            'r2': r2_score(y_test, y_test_pred),
            'mse': mean_squared_error(y_test, y_test_pred),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'best_params': grid.best_params_,
            'cv_best_score': grid.best_score_
        }

    else:
        # Einfaches Training ohne GridSearch
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            'r2': r2_score(y_test, y_test_pred),
            'mse': mean_squared_error(y_test, y_test_pred),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
        }

    return model, metrics


# ============================================
# Baseline Model (Naive Predictor)
# ============================================

def train_naive_baseline(X_train, y_train, X_test, y_test):
    """
    Trainiert ein naives Baseline-Modell für Zeitreihen

    Das Modell sagt einfach voraus: y_pred[t] = y[t-1]
    (die letzte beobachtete Änderung wird für die nächste Periode übernommen)

    Args:
        X_train, y_train, X_test, y_test: Train/Test Splits

    Returns:
        Tuple von (None, metrics)
        Model ist None, da kein Training nötig
    """
    # Für Zeitreihen: Nimm einfach den letzten Wert als Vorhersage
    # Trainingsset: Predict mit vorherigem Wert
    y_train_pred = np.concatenate([[0.0], y_train.values[:-1]])  # Erste Vorhersage = 0

    # Testset: Verwende letzten Wert aus Training als erste Vorhersage,
    # dann jeweils den vorherigen echten Wert
    y_test_pred = np.concatenate([[y_train.iloc[-1]], y_test.values[:-1]])

    metrics = {
        'r2': r2_score(y_test, y_test_pred),
        'mse': mean_squared_error(y_test, y_test_pred),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
    }

    return None, metrics


if __name__ == "__main__":
    # Test
    print("Models Wrapper Test")

    # Erstelle Test-Daten
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y_train = pd.Series(np.random.randn(100))
    X_test = pd.DataFrame(np.random.randn(20, 5), columns=[f'feature_{i}' for i in range(5)])
    y_test = pd.Series(np.random.randn(20))

    # Test OLS
    print("\nTest OLS...")
    model, metrics = train_ols(X_train, y_train, X_test, y_test)
    print(f"R²: {metrics['r2']:.4f}, MSE: {metrics['mse']:.6f}")

    # Test Ridge
    print("\nTest Ridge...")
    model, metrics = train_ridge(X_train, y_train, X_test, y_test)
    print(f"R²: {metrics['r2']:.4f}, MSE: {metrics['mse']:.6f}, Best Alpha: {metrics['best_alpha']}")

    print("\n✓ Models Wrapper funktioniert!")
