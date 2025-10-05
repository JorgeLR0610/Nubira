import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
import joblib

# Dataset
class WeatherDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        # Features: year, monthOfYear, day (día del mes)
        self.features = self.data[["year", "monthOfYear", "day"]].values
        # Labels: T2M_MAX, T2M_MIN, PRECTOTCORR
        self.labels = self.data[["T2M_MAX", "T2M_MIN", "PRECTOTCORR"]].values

        # Escaladores
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

# Modelo
class WeatherPrediction(nn.Module):
    def __init__(self):
        super(WeatherPrediction, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),   # 3 features: year, month, day
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 3)   # 3 salidas: T2M_MAX, T2M_MIN, PRECTOTCORR
        )

    def forward(self, x):
        return self.fc(x)

def train_save(Dataset):
    # Dataset y DataLoader
    dataloader = DataLoader(Dataset, batch_size=27, shuffle=True)

    # Modelo, criterio y optimizador
    model = WeatherPrediction()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Entrenamiento
    for epoch in range(100):
        for x, y in dataloader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "weather_model.pth")
    joblib.dump(Dataset.scaler_x, "scaler_x.save")
    joblib.dump(Dataset.scaler_y, "scaler_y.save")