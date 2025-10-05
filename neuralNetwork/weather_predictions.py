import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np

# =========================
# Dataset
# =========================
class weatherDataset(Dataset):
    def __init__(self, frame):
        frame = frame.replace(-999.0, np.nan).fillna(method='ffill')
        self.x = frame[['YEAR', 'MO', 'DY']].values.astype(float)
        self.y = frame[['T2M_MAX', 'T2M_MIN','PRECTOTCORR']].values.astype(float)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        X = torch.tensor(self.x[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return X, y

# =========================
# Modelo
# =========================
class WeatherPrediction(nn.Module):
    def __init__(self, output_size):
        super(WeatherPrediction, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.fc(x)

# =========================
# Clase de entrenamiento
# =========================
class TrainModel:
    def __init__(self, frame, output_size=3):
        dataset = weatherDataset(frame)
        self.dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        self.model = WeatherPrediction(output_size=output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def train(self, epochs=80):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X, y in self.dataloader:
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.dataloader)  # ⚡ promedio por batch
            print(f"🌀 Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    def save_model(self, path="weather_model.pt"):
        torch.save(self.model.state_dict(), path)
        print(f"✅ Modelo guardado en {path}")

    def predict(self, year, month, day):
        self.model.eval()
        entrada = torch.tensor([[year, month, day]], dtype=torch.float32)
        with torch.no_grad():
            salida = self.model(entrada).numpy()[0]
        return {
            "T2M_MAX": round(salida[0], 2),
            "T2M_MIN": round(salida[1], 2),
            "PRECTOTCORR": round(salida[2], 2)
        }