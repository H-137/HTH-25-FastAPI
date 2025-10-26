import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from data import parse_divs_to_df
from mapper import CityToDivisionMapper
from climate_model import ClimateGRU


mapper = CityToDivisionMapper('app/backend/CONUS_CLIMATE_DIVISIONS.shp/GIS.OFFICIAL_CLIM_DIVISIONS.shp')
temp_url = 'https://www.ncei.noaa.gov/pub/data/cirs/climdiv/climdiv-tmpcdv-v1.0.0-20250905'
raw_df = parse_divs_to_df(temp_url)

raw_df = raw_df[raw_df['element_code'] == '02'].copy()

raw_df['region'] = raw_df['state_code'] + '_' + raw_df['division']

division_coords = {}

for i, row in mapper.divisions_gdf.iterrows():
    climdiv = str(row['CLIMDIV']).zfill(4)
    state_code = climdiv[:2]
    div_num = climdiv[2:]
    region_id = f"{state_code}_{div_num}"
    
    #gets the coords for each division based on the center
    centroid = row.geometry.centroid
    division_coords[region_id] = {
        'lat': centroid.y,
        'lon': centroid.x
    }

raw_df['lat'] = raw_df['region'].map(lambda x: division_coords.get(x, {}).get('lat', np.nan))
raw_df['lon'] = raw_df['region'].map(lambda x: division_coords.get(x, {}).get('lon', np.nan))
raw_df = raw_df.dropna(subset=['lat', 'lon'])

# Select and rename columns to match your format
df = raw_df[['region', 'year', 'yearly_avg', 'lat', 'lon']].copy()
df = df.rename(columns={'yearly_avg': 'temp'})

print(f"\nPrepared data for {df['region'].nunique()} climate divisions")
print(f"Year range: {df['year'].min():.0f} - {df['year'].max():.0f}")


years = np.arange(1900, 2025)
regions = df['region'].unique()
features = ['temp', 'lat', 'lon']
target = 'temp'
seq_len = 30
horizon = 20

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

X, y, mask, region_ids = [], [], [], []

#sliding window of sequences for training and predicting +20y at one time
for region in df['region'].unique():
    region_df = df[df['region'] == region].sort_values('year')
    data_reg = region_df[features].values
    target_reg = region_df['temp'].values
    for i in range(len(data_reg) - seq_len):
        X_seq = data_reg[i:i + seq_len]
        
        end_idx = min(i + seq_len + horizon, len(data_reg))
        y_seq = target_reg[i + seq_len:end_idx]

        padded = np.full(horizon, np.nan)
        padded[:len(y_seq)] = y_seq

        
        X.append(X_seq)
        y.append(padded)
        mask.append(~np.isnan(padded))


X = np.array(X)
y = np.array(y)
mask = np.array(mask)
region_ids = np.array(region_ids)
#train test split for validation

indices = np.arange(len(X))
train_idx, val_idx = train_test_split(indices, test_size=0.2)

X_train = torch.tensor(X[train_idx], dtype=torch.float32)
y_train = torch.tensor(np.nan_to_num(y[train_idx], nan=0), dtype=torch.float32)
mask_train = torch.tensor(mask[train_idx], dtype=torch.bool)

X_val = torch.tensor(X[val_idx], dtype=torch.float32)
y_val = torch.tensor(np.nan_to_num(y[val_idx], nan=0), dtype=torch.float32)
mask_val = torch.tensor(mask[val_idx], dtype=torch.bool)

input_size = X_train.shape[2]
model = ClimateGRU(input_size)
criterion = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr = .001)
epochs = 50

train_losses = []
val_losses = []
best_val_loss = float('inf')

#classic training loop
for e in range(epochs):
    opt.zero_grad()
    out_train = model(X_train)
    loss_train = ((out_train - y_train) ** 2)
    loss_train = (loss_train * mask_train).sum() / mask_train.sum()
    loss_train.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        out_val = model(X_val)
        loss_val = ((out_val - y_val) ** 2)
        loss_val = (loss_val * mask_val).sum() / mask_val.sum()
    
    train_losses.append(loss_train.item())
    val_losses.append(loss_val.item())
    
    # Save best model
    if loss_val.item() < best_val_loss:
        best_val_loss = loss_val.item()
        torch.save(model.state_dict(), "model_params_best.pth")

    print(f'Epoch {e+1}/{epochs} | Train Loss: {loss_train.item():.4f} | Val Loss: {loss_val.item():.4f}')

torch.save(model.state_dict(), "model_params.pth")

model.eval()
preds = []

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

torch.save(model.state_dict(), 'model_params.pth')

#define the sequence of seq_len length
with torch.no_grad():
    for region in df['region'].unique():
        region_df = df[df['region'] == region].sort_values('year')
        seq = torch.tensor(region_df[features].values[-seq_len:], dtype=torch.float32).unsqueeze(0) 
        region_preds = []

        y_pred = model(seq).squeeze(0).numpy()
        preds.append((region, y_pred))

preds_df = pd.DataFrame({
    'region': np.repeat(regions, horizon),
    'year': np.tile(np.arange(2025, 2025 + horizon), len(regions)),
    'pred_temp': np.concatenate([p[1] for p in preds])
})

#eval with train test split

model.eval()
with torch.no_grad():
   
    out_train = model(X_train)
    train_preds_scaled = out_train.numpy()
    train_actuals_scaled = y_train.numpy()
    train_mask_np = mask_train.numpy()
    
    out_val = model(X_val)
    val_preds_scaled = out_val.numpy()
    val_actuals_scaled = y_val.numpy()
    val_mask_np = mask_val.numpy()

def inverse_transform_temps(scaled_temps, scaler, lat, lon):
    
    n_samples, horizon = scaled_temps.shape
    
    #dummy to match dimensions of original scaler
    dummy = np.column_stack([
        scaled_temps.flatten(),
        np.full(n_samples * horizon, lat),
        np.full(n_samples * horizon, lon)
    ])

    original = scaler.inverse_transform(dummy)[:, 0]  # Extract temp column
    return original.reshape(n_samples, horizon)

mean_lat = df['lat'].iloc[0] if 'lat' in df.columns else 40.0
mean_lon = df['lon'].iloc[0] if 'lon' in df.columns else -95.0

#unscale for predictions
train_preds = inverse_transform_temps(train_preds_scaled, scaler, mean_lat, mean_lon)
train_actuals = inverse_transform_temps(train_actuals_scaled, scaler, mean_lat, mean_lon)
val_preds = inverse_transform_temps(val_preds_scaled, scaler, mean_lat, mean_lon)
val_actuals = inverse_transform_temps(val_actuals_scaled, scaler, mean_lat, mean_lon)

train_mae = np.abs(train_preds - train_actuals)[train_mask_np].mean()
train_rmse = np.sqrt(((train_preds - train_actuals) ** 2)[train_mask_np].mean())

val_mae = np.abs(val_preds - val_actuals)[val_mask_np].mean()
val_rmse = np.sqrt(((val_preds - val_actuals) ** 2)[val_mask_np].mean())

#check for overfitting
if val_mae > train_mae * 1.5:
    print("\n  Warning: Possible overfitting detected (val MAE >> train MAE)")
else:
    print("\n Model generalization looks good")

#plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
print("\nLearning curves saved to training_curves.png")

print("SAMPLE PREDICTIONS VS ACTUALS")

sample_idx = 0
sample_pred = val_preds[sample_idx]  #now in Fahrenheit, not scaled version
sample_actual = val_actuals[sample_idx]  
sample_mask = val_mask_np[sample_idx]

print("\nYear | Predicted (°F) | Actual (°F) | Error (°F)")
print("-" * 55)
for i in range(min(10, horizon)):
    if sample_mask[i]:
        error = sample_pred[i] - sample_actual[i]
        print(f"{i+1:4d} | {sample_pred[i]:13.2f} | {sample_actual[i]:11.2f} | {error:+9.2f}")