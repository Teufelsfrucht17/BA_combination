import torch
import torch.nn as nn

from Dataprep2 import finalrunner

# Hyperparameter (einfach anpassbar)
SHEET = 3      # Excel-Sheet Index
EPOCHS = 200   # Trainingsdurchläufe
LR = 0.01      # Lernrate

# Device wählen (CUDA wenn verfügbar)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    try:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# 1) Daten laden: X (N×1), Y (N×3)
X_df, Y_df = finalrunner(SHEET)
X = torch.tensor(X_df.values, dtype=torch.float32, device=device)
Y = torch.tensor(Y_df.values, dtype=torch.float32, device=device)

# 2) Modell: nur eine Linearschicht 1 -> 3
#    y = X * W^T + b  (W: 3×1, b: 3)
model = nn.Linear(1, 3).to(device)

# 3) Verlustfunktion (MSE) und Optimierer (SGD)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# 4) Training: voller Batch (mit Mixed Precision auf CUDA)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        y_pred = model(X)
        loss = criterion(y_pred, Y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if epoch == 1 or epoch % 20 == 0 or epoch == EPOCHS:
        print(f"Epoch {epoch}/{EPOCHS} - loss={loss.item():.6f}")

# 5) Gewichte und Beispielvorhersagen anzeigen
print("Gewichte (W):\n", model.weight.detach().cpu())
print("Bias (b):\n", model.bias.detach().cpu())

with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
    print("Vorhersagen (erste 5 Zeilen):")
    print(model(X[:5]).detach().cpu())
