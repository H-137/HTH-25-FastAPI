import numpy as np
import torch
import pickle
from mapper import CityToDivisionMapper
from climate_model import load_model

def predict_temperature_from_api_data(city: str, state: str, lat: float, lon: float, 
                                      historical_years: list, historical_temps: list) -> dict:
    
    print(f"\nProcessing prediction for {city}, {state}")
    print(f"Coordinates: ({lat:.4f}, {lon:.4f})")
    print(f"Historical data: {len(historical_years)} years ({min(historical_years)}-{max(historical_years)})")
    
    if len(historical_years) != len(historical_temps):
        raise ValueError("Years and temperatures must have the same length")
    
    seq_len = 30  # Must match training
    if len(historical_temps) < seq_len:
        raise ValueError(f"Need at least {seq_len} years of data, but only have {len(historical_temps)}")
    
    years = np.array(historical_years)
    temps = np.array(historical_temps)
    
    sort_idx = np.argsort(years)
    years = years[sort_idx]
    temps = temps[sort_idx]
    
    recent_years = years[-seq_len:]
    recent_temps = temps[-seq_len:]
    
    #use saved scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    features = np.column_stack([
        recent_temps,
        np.full(seq_len, lat),
        np.full(seq_len, lon)
    ])
    
    features_scaled = scaler.transform(features)
    
    input_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0)
    
    input_size = 3  # temp, lat, lon
    model = load_model(input_size, path='model_params.pth')
    
    with torch.no_grad():
        predictions_scaled = model(input_tensor).squeeze(0).numpy()
    #for scaler dims
    dummy_features = np.column_stack([
        predictions_scaled,
        np.full(len(predictions_scaled), lat),
        np.full(len(predictions_scaled), lon)
    ])
    predictions_original = scaler.inverse_transform(dummy_features)[:, 0]
    
    last_historical_year = int(recent_years[-1])
    future_years = list(range(last_historical_year + 1, last_historical_year + 21))
    
    response = {
        'city': city,
        'state': state,
        'coordinates': {'lat': lat, 'lon': lon},
        'historical_data': {
            'years': recent_years.tolist(),
            'temperatures': recent_temps.tolist(),
            'last_year': last_historical_year,
            'avg_temp': float(recent_temps.mean())
        },
        'predictions': {
            'years': future_years,
            'temperatures': predictions_original.tolist(),
            'avg_temp': float(predictions_original.mean()),
            'change': float(predictions_original.mean() - recent_temps.mean())
        }
    }
    
    return response

if __name__ == "__main__":

    mapper = CityToDivisionMapper('app/backend/CONUS_CLIMATE_DIVISIONS.shp/GIS.OFFICIAL_CLIM_DIVISIONS.shp')
    city_info = mapper.city_to_division("Boston", "MA")
    
    #this is a test one can remove
    boston_years = list(range(1970, 2025))
    boston_temps = [50.0 + i*0.05 + np.random.randn() for i in range(len(boston_years))]
    
    result2 = predict_temperature_from_api_data(
        city="Boston",
        state="MA",
        lat=city_info['latitude'],
        lon=city_info['longitude'],
        historical_years=boston_years,
        historical_temps=boston_temps
    )