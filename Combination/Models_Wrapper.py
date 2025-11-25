"""
Models_Wrapper.py - Simplified wrappers for all models
These functions are streamlined compared to version 1 because we train portfolio-wide
"""

import numpy as np
import pandas as pd
import time
import copy
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union, List

# Sklearn models
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

from logger_config import get_logger

logger = get_logger(__name__)

from tqdm import tqdm

# Constants
DEFAULT_RANDOM_SEED = 42
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_EARLY_STOPPING_PATIENCE = 20
DEFAULT_SCHEDULER_PATIENCE = 10
DEFAULT_EPSILON = 1e-8
DEFAULT_DROPOUT = 0.2
DEFAULT_EPOCH_PRINT_INTERVAL = 50


def directional_accuracy(y_true: Union[np.ndarray, pd.Series, List], y_pred: Union[np.ndarray, pd.Series, List]) -> float:
    """Calculate directional accuracy (matching signs)."""
    if len(y_true) == 0:
        return np.nan
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


# ============================================
# PyTorch Neural Network
# ============================================

class SimpleNet(nn.Module):
    """Simple MLP with two hidden layers"""

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
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    hidden1: int = 64,
    hidden2: int = 32,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 0.001,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    use_scheduler: bool = True,
    scheduler_patience: int = DEFAULT_SCHEDULER_PATIENCE,
    weight_decay: float = 0.0,
    standardize_target: bool = True,
    portfolio_name: Optional[str] = None,
    period_type: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train a PyTorch MLP with two hidden layers, dropout, early stopping, and an optional LR scheduler.

    Args:
        X_train: Scaled training features
        y_train: Training target
        X_test: Scaled test features
        y_test: Test target
        hidden1: Size of first hidden layer
        hidden2: Size of second hidden layer
        epochs: Max epochs
        batch_size: Batch size
        lr: Learning rate
        validation_split: Portion of training data reserved for validation
        early_stopping_patience: Epochs without improvement before stopping
        use_scheduler: Enable learning rate scheduler
        scheduler_patience: Patience for scheduler
        weight_decay: L2 regularization
        standardize_target: Whether to standardize the target
        portfolio_name: Optional portfolio name for loss logs
        period_type: Optional period name for loss logs

    Returns:
        model: Trained PyTorch model
        metrics: Dictionary with r2, mse, mae, train_r2, directional_accuracy, best_val_loss, stopped_at_epoch, etc.
    """
    # Seeds for reproducibility
    np.random.seed(DEFAULT_RANDOM_SEED)
    torch.manual_seed(DEFAULT_RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(DEFAULT_RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug("CUDA available, using GPU")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors
    X_train_t = torch.tensor(X_train.values if isinstance(X_train, pd.DataFrame) else X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values if isinstance(y_train, pd.Series) else y_train, dtype=torch.float32).reshape(-1, 1)
    X_test_t = torch.tensor(X_test.values if isinstance(X_test, pd.DataFrame) else X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values if isinstance(y_test, pd.Series) else y_test, dtype=torch.float32).reshape(-1, 1)

    # Optionally standardize target (fit on inner training slice)
    n_train = len(X_train_t)
    val_idx = int(n_train * (1 - validation_split))
    X_train_inner = X_train_t[:val_idx]
    y_train_inner = y_train_t[:val_idx]
    X_val = X_train_t[val_idx:]
    y_val = y_train_t[val_idx:]

    y_mean = y_train_inner.mean()
    y_std = y_train_inner.std()
    if y_std.item() < DEFAULT_EPSILON:
        logger.warning("Sehr kleine Standardabweichung im Target, setze auf 1.0")
        y_std = torch.tensor(1.0)

    if standardize_target:
        y_train_inner_std = (y_train_inner - y_mean) / y_std
        y_val_std = (y_val - y_mean) / y_std
    else:
        y_train_inner_std = y_train_inner
        y_val_std = y_val

    # Internal validation split (chronological)
    print(f"    Inner Train Size: {len(X_train_inner)}, Val Size: {len(X_val)}")

    # DataLoader (no shuffle for time series)
    train_dataset = TensorDataset(X_train_inner, y_train_inner_std)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Model
    n_features = X_train_t.shape[1]
    model = SimpleNet(in_features=n_features, hidden1=hidden1, hidden2=hidden2, out_features=1).to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Optional: learning rate scheduler
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=scheduler_patience
        )

    # Early Stopping
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    train_losses, val_losses = [], []

    # Training Loop
    model.train()
    epochs_iter = tqdm(range(epochs), desc="Training", leave=False)
    for epoch in epochs_iter:
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
            val_loss = criterion(val_outputs, y_val_std.to(device)).item()
        model.train()

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        # Learning Rate Scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
        else:
            patience_counter += 1

        # Print Progress
        if (epoch + 1) % DEFAULT_EPOCH_PRINT_INTERVAL == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")

        # Early Stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.6f})")
            print(f"    Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.6f})")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_train_pred_std = model(X_train_t.to(device)).cpu().numpy().flatten()
        y_test_pred_std = model(X_test_t.to(device)).cpu().numpy().flatten()

    # Transform back to original scale
    y_mean_val = y_mean.item() if hasattr(y_mean, "item") else float(y_mean)
    y_std_val = y_std.item() if hasattr(y_std, "item") else float(y_std)
    if standardize_target:
        y_train_pred = y_train_pred_std * y_std_val + y_mean_val
        y_test_pred = y_test_pred_std * y_std_val + y_mean_val
        y_train_true = y_train_t.cpu().numpy().flatten()
        y_test_true = y_test_t.cpu().numpy().flatten()
    else:
        y_train_pred = y_train_pred_std
        y_test_pred = y_test_pred_std
        y_train_true = y_train_t.cpu().numpy().flatten()
        y_test_true = y_test_t.cpu().numpy().flatten()

    # Optional: Loss-Curves speichern
    if portfolio_name and period_type:
        curves_path = Path("Results") / f"pytorch_training_{portfolio_name}_{period_type}.csv"
        curves_path.parent.mkdir(exist_ok=True)
        pd.DataFrame({
            "epoch": np.arange(1, len(train_losses) + 1),
            "train_loss": train_losses,
            "val_loss": val_losses
        }).to_csv(curves_path, index=False)

    metrics = {
        'r2': r2_score(y_test_true, y_test_pred),
        'mse': mean_squared_error(y_test_true, y_test_pred),
        'mae': mean_absolute_error(y_test_true, y_test_pred),
        'train_r2': r2_score(y_train_true, y_train_pred),
        'directional_accuracy': directional_accuracy(y_test_true, y_test_pred),
        'directional_accuracy_train': directional_accuracy(y_train_true, y_train_pred),
        'best_val_loss': best_val_loss,
        'stopped_at_epoch': best_epoch if best_epoch else epoch + 1,
        'loss_curve_train': train_losses,
        'loss_curve_val': val_losses
    }

    return model, metrics


# ============================================
# Sklearn Neural Network
# ============================================

def train_sklearn_nn(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    hidden_layer_sizes: Tuple[int, ...] = (64, 32),
    max_iter: int = 1000,
    n_splits: int = 5,
    use_gridsearch: bool = True
) -> Tuple[MLPRegressor, Dict[str, Any]]:
    """
    Train an sklearn MLPRegressor with optional hyperparameter tuning.

    Note: scaling happens in ModelComparison.py; do not re-scale here to avoid leakage.

    Args:
        X_train: Scaled training features
        y_train: Training target
        X_test: Scaled test features
        y_test: Test target
        hidden_layer_sizes: Layer sizes (default: (64, 32))
        max_iter: Max iterations
        n_splits: Splits for TimeSeriesSplit
        use_gridsearch: Whether to run GridSearchCV

    Returns:
        model: Trained MLPRegressor
        metrics: Dict with r2, mse, mae, train_r2, directional_accuracy, etc.
    """
    base_layers = tuple(hidden_layer_sizes)

    def scale_layers(factor: float):
        return tuple(max(1, int(round(h * factor))) for h in base_layers)

    candidate_layers = [base_layers, scale_layers(0.5), scale_layers(1.5)]
    # Remove duplicates when layer sizes collapse to same values
    candidate_layers = list(dict.fromkeys(candidate_layers))

    if use_gridsearch:
        # Hyperparameter grid near configured values
        param_grid = {
            'hidden_layer_sizes': candidate_layers,
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
            'directional_accuracy': directional_accuracy(y_test, y_test_pred),
            'directional_accuracy_train': directional_accuracy(y_train, y_train_pred),
            'best_params': grid.best_params_,
            'cv_best_score': grid.best_score_
        }

    else:
        # Modell OHNE Scaling-Pipeline (Daten sind bereits skaliert)
        model = MLPRegressor(
            hidden_layer_sizes=base_layers,
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
            'directional_accuracy': directional_accuracy(y_test, y_test_pred),
            'directional_accuracy_train': directional_accuracy(y_train, y_train_pred),
        }

    return model, metrics


# ============================================
# OLS Regression
# ============================================

def train_ols(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray]
) -> Tuple[LinearRegression, Dict[str, Any]]:
    """
    Train a simple OLS linear regression (no regularization).
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
        'directional_accuracy': directional_accuracy(y_test, y_test_pred),
        'directional_accuracy_train': directional_accuracy(y_train, y_train_pred),
    }

    return model, metrics


# ============================================
# Ridge Regression
# ============================================

def train_ridge(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    alpha_values: Optional[List[float]] = None
) -> Tuple[Ridge, Dict[str, Any]]:
    """
    Train Ridge regression with a grid search over alpha values.
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
        'directional_accuracy': directional_accuracy(y_test, y_test_pred),
        'directional_accuracy_train': directional_accuracy(y_train, y_train_pred),
        'best_alpha': grid.best_params_['alpha']
    }

    return grid.best_estimator_, metrics


# ============================================
# Random Forest
# ============================================

def train_random_forest(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    n_estimators: int = 300,
    max_depth: Optional[int] = 10,
    min_samples_split: int = 5,
    n_splits: int = 5,
    use_gridsearch: bool = True
) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
    """
    Train a RandomForestRegressor with optional hyperparameter tuning.
    """
    if use_gridsearch:
        # Hyperparameter grid for search
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        print(f"    Starting GridSearch with {n_splits} TimeSeriesSplit folds...")

        grid = GridSearchCV(
            rf,
            param_grid=param_grid,
            cv=tscv,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )

        grid.fit(X_train, y_train)

        print(f"    Best params: {grid.best_params_}")
        print(f"    Best CV R² score: {grid.best_score_:.4f}")

        model = grid.best_estimator_

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            'r2': r2_score(y_test, y_test_pred),
            'mse': mean_squared_error(y_test, y_test_pred),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'directional_accuracy': directional_accuracy(y_test, y_test_pred),
            'directional_accuracy_train': directional_accuracy(y_train, y_train_pred),
            'best_params': grid.best_params_,
            'cv_best_score': grid.best_score_
        }

    else:
        # Simple training without GridSearch
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
            'directional_accuracy': directional_accuracy(y_test, y_test_pred),
            'directional_accuracy_train': directional_accuracy(y_train, y_train_pred),
        }

    return model, metrics


# ============================================
# Baseline Model (Naive Predictor)
# ============================================

def train_naive_baseline(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray]
) -> Tuple[None, Dict[str, Any]]:
    """
    Train a naive baseline for time series.

    The model predicts y_pred[t] = y[t-1]; serves as a benchmark.
    """
    # Use previous value as prediction
    y_train_pred = np.concatenate([[0.0], y_train.values[:-1]])
    y_test_pred = np.concatenate([[y_train.iloc[-1]], y_test.values[:-1]])

    metrics = {
        'r2': r2_score(y_test, y_test_pred),
        'mse': mean_squared_error(y_test, y_test_pred),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'directional_accuracy': directional_accuracy(y_test, y_test_pred),
        'directional_accuracy_train': directional_accuracy(y_train, y_train_pred),
    }

    return None, metrics


if __name__ == "__main__":
    # Test
    print("Models Wrapper Test")

    # Create test data
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

    print("\nModels Wrapper working!")
