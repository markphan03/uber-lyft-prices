#!/usr/bin/env python
# coding: utf-8

# ### Prepare dataset

# In[1]:


import pandas as pd
import numpy as np


weather_path = "data/weather.csv"
cab_path = "data/cab_rides.csv"

# In[2]:


df_cab = pd.read_csv(cab_path, on_bad_lines="skip", low_memory=False)
df_weather = pd.read_csv(weather_path, on_bad_lines="skip", low_memory=False)

# ### KNN Imputation
# For missing rain values, use a KNN-based regression/imputation approach with the numeric predictors like temp, clouds, pressure, humidity, and wind. KNNImputer fills missing values by using the mean of the nearest rows.

# In[3]:


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

df_weather = df_weather.copy()

features_num = ['temp', 'clouds', 'pressure', 'humidity', 'wind']
target = 'rain'

# One-hot encode location
df_weather = pd.get_dummies(df_weather, columns=['location'], drop_first=False)

feature_cols = [col for col in df_weather.columns if col != target and col != 'time_stamp']

train_df = df_weather[df_weather[target].notna()].copy()
missing_df = df_weather[df_weather[target].isna()].copy()

X_train = train_df[feature_cols]
y_train = train_df[target]
X_missing = missing_df[feature_cols]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_missing_scaled = scaler.transform(X_missing)

knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn.fit(X_train_scaled, y_train)

df_weather.loc[df_weather[target].isna(), target] = knn.predict(X_missing_scaled)

# In[4]:


# df_cab = pd.read_csv(cab_path, on_bad_lines="skip", low_memory=False)
# df_weather = pd.read_csv(weather_path, on_bad_lines="skip", low_memory=False)
# Convert timestamps (assuming milliseconds)
df_cab['time_stamp'] = pd.to_datetime(df_cab['time_stamp'], unit='ms').astype('datetime64[ns]')
df_weather['time_stamp'] = pd.to_datetime(df_weather['time_stamp'], unit='s').astype('datetime64[ns]')

df_cab = df_cab.sort_values('time_stamp')
df_weather = df_weather.sort_values('time_stamp')

# Merge cab trip with weather information with nearest timestamp within 15 minutes
merged = pd.merge_asof(
    df_cab,
    df_weather,
    on='time_stamp',
    direction='nearest',
    tolerance=pd.Timedelta(minutes=15)
)


# Drop rows where no weather match
merged = merged.dropna()

# ### Feature Engineering

# In[5]:


# Time features
merged['hour'] = merged['time_stamp'].dt.hour
merged['day'] = merged['time_stamp'].dt.day
merged['weekday'] = merged['time_stamp'].dt.weekday

# Example categorical encoding
merged = pd.get_dummies(merged, columns=['cab_type', 'source', 'destination', 'name'], drop_first=False)

# Drop unused
merged = merged.drop(columns=['time_stamp', 'id', 'product_id'])

# ### Prepare Data

# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Target
y = merged['price']
X = merged.drop(columns=['price'])

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ### Train ML models - Linear Regression & Random Forrest

# In[7]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

# In[8]:


from xgboost import XGBRFRegressor

rf = XGBRFRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    device="cuda",
    tree_method="hist"
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# In[9]:


import numpy as np
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_log_error


# Write MAE for both models
print("Linear Regression")

print("RMSE:", np.sqrt(root_mean_squared_error(y_test, y_pred_lr)))
print("R2:", r2_score(y_test, y_pred_lr))
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("MAPE:", mean_absolute_percentage_error(y_test, y_pred_lr))
print("RMSLE:", np.sqrt(root_mean_squared_log_error(y_test, y_pred_lr)))

print("\nRandom Forest")
print("RMSE:", np.sqrt(root_mean_squared_error(y_test, y_pred_rf)))
print("R2:", r2_score(y_test, y_pred_rf))
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("MAPE:", mean_absolute_percentage_error(y_test, y_pred_rf))
print("RMSLE:", np.sqrt(root_mean_squared_log_error(y_test, y_pred_rf)))

# 

# In[10]:


import matplotlib.pyplot as plt

# Compute residuals (error)
errors = y_pred_rf - y_test

plt.scatter(range(len(errors)), errors)
plt.axhline(0, linestyle='--')  # reference line at zero
plt.xlabel("Data Point Index")
plt.ylabel("Prediction Error (Predicted - Actual)")
plt.title("Prediction Errors (Residuals)")
plt.show()

# In[11]:


import torch
import torch.nn as nn

class CabPriceModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# In[12]:


import os
import torch
import torch.nn as nn

checkpoint_path = "cab_price_checkpoint.pth"

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Convert to tensors and move to device
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)

X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

# Model
model = CabPriceModel(X_train.shape[1]).to(device)

# Loss + optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

start_epoch = 0
best_loss = float("inf")

# Load checkpoint if it exists
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint.get("loss", float("inf"))
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

num_epochs = 800

epoch = start_epoch - 1  # fallback if loop doesn't run (training already complete)

for epoch in range(start_epoch, num_epochs):
    model.train()

    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item()
        }, checkpoint_path)

# Final save (only if training ran)
if start_epoch < num_epochs:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item()
    }, checkpoint_path)
    print("Training complete and checkpoint saved.")
else:
    print("Training already complete (checkpoint at max epoch), skipping save.")

# ### 500 epochs

# In[13]:


from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_log_error
import numpy as np

checkpoint_path = "cab_price_checkpoint.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
with torch.no_grad():
    preds = model(X_test_t).detach().cpu().numpy()
    y_true = y_test_t.detach().cpu().numpy()

rmse = np.sqrt(root_mean_squared_error(y_true, preds))
r2 = r2_score(y_true, preds)
mae = np.mean(np.abs(y_true - preds))
mape = np.mean(np.abs((y_true - preds) / y_true)) * 100
rmsle = np.sqrt(root_mean_squared_log_error(y_true, preds))

print("RMSE:", rmse)
print("R2:", r2)
print("MAE:", mae)
print("MAPE:", mape)
print("RMSLE:", rmsle)

# In[14]:


import matplotlib.pyplot as plt

# Compute residuals (error)
errors = preds - y_true

plt.scatter(range(len(errors)), errors)
plt.axhline(0, linestyle='--')  # reference line at zero
plt.xlabel("Data Point Index")
plt.ylabel("Prediction Error (Predicted - Actual)")
plt.title("Prediction Errors (Residuals)")
plt.show()
