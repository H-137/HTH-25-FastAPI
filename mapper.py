import requests
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd

class CityToDivisionMapper: 
    def __init__(self, shapefile_path='app/backend/CONUS_CLIMATE_DIVISIONS.shp/GIS.OFFICIAL_CLIM_DIVISIONS.shp'):
        self.divisions_gdf = None
        self.load_division_boundaries(shapefile_path)
    
    def load_division_boundaries(self, shapefile_path):
        try:
            self.divisions_gdf = gpd.read_file(shapefile_path)
            
            if 'CLIMDIV' in self.divisions_gdf.columns: #code for state and specific devision type in it
                self.divisions_gdf['CLIMDIV'] = self.divisions_gdf['CLIMDIV'].astype(str).str.zfill(4)
                
                self.divisions_gdf['state_code_check'] = self.divisions_gdf['CLIMDIV'].str[:2]
                #HI and AK are not in contiguous US
                self.divisions_gdf = self.divisions_gdf[
                    ~self.divisions_gdf['state_code_check'].isin(['49', '50'])
                ]
                self.divisions_gdf = self.divisions_gdf.drop(columns=['state_code_check'])
            
            print(f"After filtering: {len(self.divisions_gdf)} CONUS climate divisions")
            
        except Exception as e:
            print(f"Error loading shapefile: {e}")
            print("Make sure all shapefile components (.shp, .shx, .dbf, etc.) are available")
            raise
    
    def get_city_coordinates(self, city_name, state=None):
        
        base_url = "https://nominatim.openstreetmap.org/search" #open street map to get coords
        
       #api query stuff
        if state:
            query = f"{city_name}, {state}, USA"
        else:
            query = f"{city_name}, USA"
        
        params = {
            'q': query,
            'format': 'json',
            'limit': 1,
            'countrycodes': 'us'  
        }
        
        headers = {'User-Agent': 'ClimateApp/1.0'}
        
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        
        results = response.json()
        if not results:
            raise ValueError(f"City not found: {query}")
        
        result = results[0]
        lat = float(result['lat'])
        lon = float(result['lon'])
        display_name = result['display_name']
        
        return lat, lon, display_name
    
    def coords_to_division(self, lat, lon):
        if self.divisions_gdf is None:
            raise ValueError("Division boundaries not loaded")
        
        #check where this point is wrt divs
        point = Point(lon, lat)
        mask = self.divisions_gdf.contains(point)
        matching_divisions = self.divisions_gdf[mask]
        
        if len(matching_divisions) == 0:
            raise ValueError(f"No CONUS division found for coordinates ({lat}, {lon}). "
                           f"Note: Alaska and Hawaii are not supported.")
        
        division = matching_divisions.iloc[0]
        
        climdiv_code = str(division.get('CLIMDIV', ''))
        
        if len(climdiv_code) == 4:
            state_code = climdiv_code[:2]
            division_number = climdiv_code[2:]
        else:
            state_code = str(division.get('STATE_CODE', division.get('STATE', ''))).zfill(2)
            division_number = str(division.get('DIV_NUM', division.get('DIVISION', ''))).zfill(2)
        
        result = {
            'state_code': state_code,
            'division_number': division_number,
            'state_name': division.get('STATE_NAME', division.get('ST_NAME', '')),
            'division_name': division.get('DIV_NAME', division.get('NAME', ''))
        }
        
        return result
    
    def city_to_division(self, city_name, state=None):
        lat, lon, full_name = self.get_city_coordinates(city_name, state)
        
        division_info = self.coords_to_division(lat, lon)
       
        result = {
            'city': city_name,
            'full_name': full_name,
            'latitude': lat,
            'longitude': lon,
            **division_info  
        }
        
        return result

if __name__ == "__main__":
    mapper = CityToDivisionMapper('app/backend/CONUS_CLIMATE_DIVISIONS.shp/GIS.OFFICIAL_CLIM_DIVISIONS.shp')
    
    city = 'userin'
    state ='userin'
    
    try:
        result = mapper.city_to_division(city, state)
        print(f"\n{city}, {state}:")
        print(f"  State Code: {result['state_code']}")
        print(f"  Division: {result['division_number']}")
        print(f"  Division Name: {result.get('division_name', 'N/A')}")
        print(f"  Coordinates: ({result['latitude']:.4f}, {result['longitude']:.4f})")
    except Exception as e:
        print(f"\n{city}, {state}: Error - {e}")