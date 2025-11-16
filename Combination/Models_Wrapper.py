"""
Models_Wrapper.py - Vereinfachte Wrapper für alle Modelle
Diese Funktionen sind vereinfacht im Vergleich zu Version 1,
da wir jetzt Portfolio-basiert trainieren (alle Aktien zusammen)
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
from copy import deepcopy

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


def train_pytorch_model(
    X_train,
    y_train,
    X_test,
    y_test,
    hidden1=64,
    hidden2=32,
    epochs=200,
    batch_size=64,
    lr=0.001,
    validation_split: float = 0.2,
    patience: int = 25,
    min_delta: float = 1e-4,
):
    """
    Trainiert PyTorch Neural Network

    Args:
        X_train, y_train, X_test, y_test: Train/Test Splits
        hidden1: Größe des ersten Hidden Layers
        hidden2: Größe des zweiten Hidden Layers
        epochs: Anzahl Epochen
        batch_size: Batch-Größe
        lr: Learning Rate

    Returns:
        Tuple von (model, metrics)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # In numpy konvertieren
    X_train_values = X_train.values if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
    y_train_values = y_train.values if isinstance(y_train, pd.Series) else np.asarray(y_train)
    X_test_values = X_test.values if isinstance(X_test, pd.DataFrame) else np.asarray(X_test)
    y_test_values = y_test.values if isinstance(y_test, pd.Series) else np.asarray(y_test)

    if X_train_values.ndim == 1:
        X_train_values = X_train_values.reshape(-1, 1)
        X_test_values = X_test_values.reshape(-1, 1)

    n_train_samples = X_train_values.shape[0]
    val_size = int(max(0, round(n_train_samples * validation_split)))
    if val_size >= n_train_samples:
        val_size = max(0, n_train_samples - 1)

    train_size = n_train_samples - val_size
    if train_size <= 0:
        raise ValueError("Zu wenige Trainingsdaten nach Validation-Split für das PyTorch Modell.")

    X_train_main = X_train_values[:train_size]
    y_train_main = y_train_values[:train_size]
    X_val = X_train_values[train_size:]
    y_val = y_train_values[train_size:]

    # Konvertiere zu Tensoren
    X_train_t = torch.tensor(X_train_main, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_main, dtype=torch.float32).reshape(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32) if val_size > 0 else None
    y_val_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1) if val_size > 0 else None
    X_test_t = torch.tensor(X_test_values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_values, dtype=torch.float32).reshape(-1, 1)

    full_X_train_t = torch.tensor(X_train_values, dtype=torch.float32)

    # DataLoader nur auf Trainingsanteil
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Modell
    n_features = X_train_t.shape[1]
    model = SimpleNet(in_features=n_features, hidden1=hidden1, hidden2=hidden2, out_features=1).to(device)

    # Optimizer und Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    scheduler = None
    if val_size > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=max(1, patience // 3)
        )

    best_val_loss = float('inf')
    best_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0

    # Training
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / max(1, len(train_loader))

        if val_size > 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t.to(device))
                val_loss = criterion(val_outputs, y_val_t.to(device)).item()

            if scheduler is not None:
                scheduler.step(val_loss)

            if val_loss + min_delta < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"    ⏹️  Early Stopping nach {epoch+1} Epochen (bestes Val-Loss {best_val_loss:.6f})")
                    break

            model.train()

        if (epoch + 1) % 50 == 0 and val_size == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")

    epochs_completed = epoch + 1 if 'epoch' in locals() else 0

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_train_pred = model(full_X_train_t.to(device)).cpu().numpy().flatten()
        y_test_pred = model(X_test_t.to(device)).cpu().numpy().flatten()

    metrics = {
        'r2': r2_score(y_test, y_test_pred),
        'mse': mean_squared_error(y_test, y_test_pred),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train_values, y_train_pred),
        'best_val_loss': best_val_loss if val_size > 0 else np.nan,
        'trained_epochs': epochs_completed,
    }

    return model, metrics


# ============================================
# Sklearn Neural Network
# ============================================

def train_sklearn_nn(
    X_train,
    y_train,
    X_test,
    y_test,
    hidden_layer_sizes=(64, 32),
    max_iter=500,
    use_cv: bool = False,
    param_grid: dict  = None,
    cv_splits: int = 5,
):
    """
    Trainiert Sklearn MLP Regressor

    Args:
        X_train, y_train, X_test, y_test: Train/Test Splits
        hidden_layer_sizes: Tuple mit Layer-Größen
        max_iter: Maximale Iterationen

    Returns:
        Tuple von (model, metrics)
    """
    # Pipeline mit Scaling
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
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
        ))
    ])

    best_params = None
    if use_cv and param_grid:
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        search = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
    else:
        model = pipe
        model.fit(X_train, y_train)

    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        'r2': r2_score(y_test, y_test_pred),
        'mse': mean_squared_error(y_test, y_test_pred),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'best_params': best_params,
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

def train_ridge(X_train, y_train, X_test, y_test, alpha_values=None, cv_splits: int = 5):
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
    tscv = TimeSeriesSplit(n_splits=cv_splits)

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

def train_random_forest(
    X_train,
    y_train,
    X_test,
    y_test,
    default_params=None,
    param_grid=None,
    use_cv: bool = False,
    cv_splits: int = 5,
):
    """
    Trainiert Random Forest Regressor

    Args:
        X_train, y_train, X_test, y_test: Train/Test Splits
        n_estimators: Anzahl Bäume
        max_depth: Maximale Tiefe
        min_samples_split: Minimale Samples für Split

    Returns:
        Tuple von (model, metrics)
    """
    best_params = None
    if default_params is None:
        default_params = {
            'n_estimators': 300,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 1,
            'max_features': 'sqrt'
        }

    if use_cv and param_grid:
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        grid = GridSearchCV(
            rf,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        best_params = grid.best_params_
    else:
        model = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            **default_params
        )
        model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        'r2': r2_score(y_test, y_test_pred),
        'mse': mean_squared_error(y_test, y_test_pred),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'best_params': best_params,
    }

    return model, metrics


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
