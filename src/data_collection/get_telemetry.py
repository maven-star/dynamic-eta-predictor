import requests
import pandas as pd
import time
from datetime import datetime

class ProductionCorridorDataEngine:
    """
    Advanced Engineering Ingestion Engine: Extracts detailed spatial-temporal 
    feature matrices and continuous weather vectors along 10 route intervals.
    """
    def __init__(self):
        self.routing_url = "http://router.project-osrm.org/route/v1/driving/"
        self.weather_url = "https://api.open-meteo.com/v1/forecast"

    def fetch_comprehensive_features(self, start_coords, end_coords, sample_points=10):
        print(f"[System] Initiating high-resolution feature extraction ({sample_points} checkpoints)...")
        
        # 1. Capture Real-Time Temporal Context from your machine
        now = datetime.now()
        current_hour = now.hour
        day_of_week = now.weekday()  # 0=Monday, 6=Sunday
        month = now.month
        
        # 2. Query Live Routing Graph
        coord_str = f"{start_coords['lng']},{start_coords['lat']};{end_coords['lng']},{end_coords['lat']}"
        route_res = requests.get(f"{self.routing_url}{coord_str}?overview=full&geometries=geojson", timeout=10)
        
        if route_res.status_code != 200:
            raise ConnectionError("Routing server network failure.")
            
        route_data = route_res.json()['routes'][0]
        waypoints = route_data['geometry']['coordinates']
        total_nodes = len(waypoints)
        
        # 3. Sample the entire highway corridor
        indices = [int(i * (total_nodes - 1) / (sample_points - 1)) for i in range(sample_points)]
        feature_matrix = []
        
        for step, idx in enumerate(indices):
            lng, lat = waypoints[idx]
            pct = int((step / (sample_points - 1)) * 100)
            
            # 4. Query Live Continuous Weather Elements
            params = {
                "latitude": lat,
                "longitude": lng,
                "current_weather": "true",
                "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,precipitation"
            }
            wx_res = requests.get(self.weather_url, params=params, timeout=10)
            
            # Default values if network drops
            temp, humidity, wind_spd, precip = 25.0, 50.0, 10.0, 0.0
            
            if wx_res.status_code == 200:
                wx_data = wx_res.json()
                current_wx = wx_data.get("current_weather", {})
                temp = current_wx.get("temperature", temp)
                wind_spd = current_wx.get("windspeed", wind_spd)
                
                # Extract the current hour's continuous humidity and precipitation values
                hourly = wx_data.get("hourly", {})
                if "relativehumidity_2m" in hourly and len(hourly["relativehumidity_2m"]) > 0:
                    humidity = hourly["relativehumidity_2m"][current_hour]
                if "precipitation" in hourly and len(hourly["precipitation"]) > 0:
                    precip = hourly["precipitation"][current_hour]

            feature_matrix.append({
                "checkpoint_pct": f"{pct}%",
                "latitude": round(lat, 4),
                "longitude": round(lng, 4),
                "hour_of_day": current_hour,
                "day_of_week": day_of_week,
                "month_index": month,
                "temp_celsius": temp,
                "relative_humidity_pct": humidity,
                "wind_speed_kmh": wind_spd,
                "precipitation_mm": precip
            })
            time.sleep(0.1) # Polite API spacing
            
        return {
            "distance_km": round(float(route_data['distance']) / 1000.0, 2),
            "base_duration_min": round(float(route_data['duration']) / 60.0, 1),
            "feature_dataframe": pd.DataFrame(feature_matrix)
        }

if __name__ == "__main__":
    delhi_truck_hub = {"lat": 28.7351, "lng": 77.1432}
    kasol_center = {"lat": 32.0100, "lng": 77.3100}
    
    engine = ProductionCorridorDataEngine()
    try:
        data = engine.fetch_comprehensive_features(delhi_truck_hub, kasol_center)
        
        print("\n" + "="*110)
        print(f"HIGH-RESOLUTION FEATURE MATRIX | Distance: {data['distance_km']} km | Base ETA: {data['base_duration_min']} mins")
        print("="*110)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(data['feature_dataframe'].to_string(index=False))
        print("="*110 + "\n")
    except Exception as e:
        print(f"Pipeline Interrupted: {e}")