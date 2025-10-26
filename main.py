import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import logging
import os

try:
    from mapper import CityToDivisionMapper
    from user_to_model import predict_temperature_from_api_data
except ImportError as e:
    print(f"Error: Failed to import modules. Make sure 'mapper.py' and 'user_to_model.py' are in the same directory.")
    print(f"Details: {e}")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Climate Prediction API (Original Code)",
    description="An API to predict future temperature trends based on historical data, using the user's original scripts.",
    version="1.0.1"
)

SHAPEFILE_PATH = './CONUS_CLIMATE_DIVISIONS.shp/GIS.OFFICIAL_CLIM_DIVISIONS.shp'

try:
    if not os.path.exists(SHAPEFILE_PATH):
        logger.critical(f"--- CRITICAL ERROR ---")
        logger.critical(f"Shapefile not found at: {SHAPEFILE_PATH}")
        logger.critical(f"Your 'mapper.py' requires this file to run.")
        logger.critical(f"Please place the shapefile at that location and restart the server.")
        logger.critical(f"Server will exit.")
        exit(1)
        
    mapper = CityToDivisionMapper(SHAPEFILE_PATH)
    logger.info(f"Successfully loaded CityToDivisionMapper from shapefile: {SHAPEFILE_PATH}")

except Exception as e:
    logger.critical(f"--- CRITICAL ERROR ---")
    logger.critical(f"Failed to initialize CityToDivisionMapper, even though shapefile may exist.")
    logger.critical(f"Error: {e}")
    logger.critical(f"This is likely an issue with 'mapper.py' or its dependencies (e.g., geopandas).")
    logger.critical(f"Server will exit.")
    exit(1)

if not os.path.exists('scaler.pkl'):
    logger.critical(f"--- CRITICAL ERROR ---")
    logger.critical(f"scaler.pkl not found. Your 'user_to_model.py' requires this file.")
    logger.critical(f"Please run 'python create_dummy_scaler.py' first.")
    logger.critical(f"Server will exit.")
    exit(1)
else:
     logger.info("Found 'scaler.pkl'.")
     
if not os.path.exists('model_params.pth'):
    logger.critical(f"--- CRITICAL ERROR ---")
    logger.critical(f"model_params.pth not found. Your 'user_to_model.py' requires this file.")
    logger.critical(f"Server will exit.")
    exit(1)
else:
    logger.info("Found 'model_params.pth'.")


class PredictionRequestAuto(BaseModel):
    """Request model for automatic prediction using just city/state."""
    city: str
    state: str

class PredictionRequestManual(BaseModel):
    """Request model for manual prediction with all data provided."""
    city: str
    state: str
    lat: float
    lon: float
    historical_years: list[int]
    historical_temps: list[float]

@app.get("/", summary="Health Check")
def read_root():
    """A simple health check endpoint to confirm the server is running."""
    return {"status": "ok", "message": "Climate Prediction API is running."}

@app.post("/predict/automatic", summary="Predict from City/State")
async def predict_automatic(request: PredictionRequestAuto):
    """
    Predicts climate data by automatically:
    1. Geocoding the city/state to get coordinates (using your mapper.py).
    2. Generating *DUMMY* historical data (since no real data source is connected).
    3. Running the prediction model (using your user_to_model.py).
    
    **Note:** This endpoint uses *simulated* historical temperatures.
    """
    logger.info(f"Received automatic prediction request for: {request.city}, {request.state}")
    
    try:
        city_info = mapper.city_to_division(request.city, request.state)
        lat = city_info['latitude']
        lon = city_info['longitude']
        logger.info(f"Geocoded {request.city}: ({lat}, {lon})")
    except Exception as e:
        logger.error(f"Geocoding failed for {request.city}, {request.state}: {e}")
        raise HTTPException(status_code=404, detail=f"Could not find coordinates for {request.city}, {request.state}. Error: {e}")

    try:
        historical_years = list(range(1995, 2025)) # 30 years
        base_temp = 50.0 + (lat - 40) * -1.0 
        historical_temps = [base_temp + i*0.05 + np.random.randn()*0.5 for i in range(len(historical_years))]
        logger.info(f"Generated {len(historical_years)} years of dummy historical data.")
    except Exception as e:
        logger.error(f"Failed to generate dummy data: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate dummy historical data.")

    try:
        prediction_result = predict_temperature_from_api_data(
            city=request.city,
            state=request.state,
            lat=lat,
            lon=lon,
            historical_years=historical_years,
            historical_temps=historical_temps
        )
        return prediction_result
    except ValueError as e:
        logger.warning(f"Prediction input validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction model failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

@app.post("/predict/manual", summary="Predict with Manual Data")
async def predict_manual(request: PredictionRequestManual):
    """
    Predicts climate data using user-provided historical data.
    This calls your user_to_model.py function directly.
    """
    logger.info(f"Received manual prediction request for: {request.city}, {request.state}")
    
    try:
        prediction_result = predict_temperature_from_api_data(
            city=request.city,
            state=request.state,
            lat=request.lat,
            lon=request.lon,
            historical_years=request.historical_years,
            historical_temps=request.historical_temps
        )
        return prediction_result
    except ValueError as e:
        logger.warning(f"Prediction input validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction model failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    """Runs the FastAPI server using uvicorn."""
    logger.info("Starting Climate Prediction API server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)