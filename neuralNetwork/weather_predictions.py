# weather_predictions.py
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# ------------------ Dataset ------------------
class WeatherDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        # Variables de entrada
        self.features = self.data[["year", "month", "day"]].values
        # Variables objetivo
        self.labels = self.data[["T2M_MAX", "T2M_MIN", "PRECTOTCORR"]].values

        # Escaladores (solo en memoria)
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        self.features = self.scaler_x.fit_transform(self.features)
        self.labels = self.scaler_y.fit_transform(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


# ------------------ Modelo ------------------
class WeatherPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.fc(x)


# ------------------ Entrenamiento y Predicción ------------------
def predict_by_date(year, month, day):
    # 1️⃣ Cargar dataset
    dataset = WeatherDataset("datos.csv")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 2️⃣ Crear modelo, pérdida y optimizador
    model = WeatherPrediction()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 3️⃣ Entrenar el modelo
    for epoch in range(80):  # Menos épocas para hacerlo más rápido
        for x, y in dataloader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Época {epoch+1}/30, Pérdida: {loss.item():.4f}")

    # 4️⃣ Predecir
    x_input = np.array([[year, month, day]])
    x_scaled = dataset.scaler_x.transform(x_input)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    y_pred = model(x_tensor).detach().numpy()
    y_real = dataset.scaler_y.inverse_transform(y_pred)

    return y_real[0].tolist()
