# src/data_collection/production_ingestion_bus.py
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from geospatial_pipeline import IndustryETAFeatureEngine

class UniversalCorridorDataEngine:
    """
    Production-grade universal data engine. Geocodes arbitrary global text inputs,
    ingests live spatial routing tracks, and queries dynamic local weather metrics
    without any hardcoded regional assumptions.
    """
    def __init__(self):
        self.geocoding_url = "https://nominatim.openstreetmap.org/search"
        self.routing_url = "http://router.project-osrm.org/route/v1/driving/"
        self.weather_url = "https://api.open-meteo.com/v1/forecast"
        
        # Standard user-agent header to comply with OpenStreetMap usage policy
        self.headers = {"User-Agent": "Dynamic-ETA-Production-Pipeline (contact@spatialeta.ai)"}

    def geocode_location(self, address_string):
        """Converts raw global address strings into dynamic coordinates via live API."""
        params = {"q": address_string, "format": "json", "limit": 1}
        res = requests.get(self.geocoding_url, params=params, headers=self.headers, timeout=10)
        
        if res.status_code != 200 or not res.json():
            raise ValueError(f"Could not geocode address: '{address_string}'. Verify global location string.")
            
        location_data = res.json()[0]
        return {
            "lat": float(location_data["lat"]),
            "lng": float(location_data["lon"]),
            "display_name": location_data["display_name"]
        }

    def fetch_live_corridor_features(self, start_address, end_address, sample_points=10):
        """
        Dynamically coordinates geocoding, pathing, and live weather ingestion 
        across any arbitrary global corridor.
        """
        # Step 1: Resolve clean text locations into float inputs dynamically
        print(f"[Ingestion] Geocoding origin: '{start_address}'...")
        origin = self.geocode_location(start_address)
        time.sleep(1.0) # Respect Nominatim's rate limit
        
        print(f"[Ingestion] Geocoding destination: '{end_address}'...")
        destination = self.geocode_location(end_address)
        
        print(f"\n[Route Config] Locked Route: \n From: {origin['display_name']}\n To:   {destination['display_name']}\n")

        # Step 2: Grab live temporal status from engine machine
        now = datetime.now()
        current_hour = now.hour
        day_of_week = now.weekday()
        
        # Step 3: Call live OSRM routing server for path generation
        coord_str = f"{origin['lng']},{origin['lat']};{destination['lng']},{destination['lat']}"
        route_res = requests.get(f"{self.routing_url}{coord_str}?overview=full&geometries=geojson", timeout=10)
        
        if route_res.status_code != 200:
            raise ConnectionError("OSRM Routing API failed to respond or resolve track path.")
            
        route_data = route_res.json()['routes'][0]
        waypoints = route_data['geometry']['coordinates']
        total_nodes = len(waypoints)
        
        # Step 4: Sample the route matrix points uniformly
        indices = [int(i * (total_nodes - 1) / (sample_points - 1)) for i in range(sample_points)]
        raw_ingested_matrix = []
        
        print(f"[Ingestion] Ingesting live weather cells over {sample_points} points...")
        for step, idx in enumerate(indices):
            lng, lat = waypoints[idx]
            pct = int((step / (sample_points - 1)) * 100)
            
            # Query local weather parameters matching current coordinates
            wx_params = {
                "latitude": lat,
                "longitude": lng,
                "current_weather": "true",
                "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,precipitation"
            }
            wx_res = requests.get(self.weather_url, params=wx_params, timeout=10)
            
            temp, humidity, wind_spd, precip = 20.0, 60.0, 12.0, 0.0 # Standard defensive defaults
            
            if wx_res.status_code == 200:
                wx_data = wx_res.json()
                current_wx = wx_data.get("current_weather", {})
                temp = current_wx.get("temperature", temp)
                wind_spd = current_wx.get("windspeed", wind_spd)
                
                hourly = wx_data.get("hourly", {})
                if "relativehumidity_2m" in hourly and len(hourly["relativehumidity_2m"]) > 0:
                    humidity = hourly["relativehumidity_2m"][current_hour]
                if "precipitation" in hourly and len(hourly["precipitation"]) > 0:
                    precip = hourly["precipitation"][current_hour]

            raw_ingested_matrix.append({
                "checkpoint_pct": f"{pct}%",
                "latitude": lat,
                "longitude": lng,
                "hour_of_day": current_hour,
                "day_of_week": day_of_week,
                "temp_celsius": temp,
                "relative_humidity_pct": humidity,
                "wind_speed_kmh": wind_spd,
                "precipitation_mm": precip
            })
            time.sleep(0.05) # Clean API spacing
            
        df_raw = pd.DataFrame(raw_ingested_matrix)
        
        # Step 5: Pipe the raw matrix directly into the structural feature engine
        # This appends H3 spatial hexagons, digital elevations, and incline curves automatically
        feature_processor = IndustryETAFeatureEngine()
        df_finalized = feature_processor.transform_pipeline(df_raw)
        
        return {
            "distance_km": round(float(route_data['distance']) / 1000.0, 2),
            "base_duration_min": round(float(route_data['duration']) / 60.0, 1),
            "feature_dataframe": df_finalized
        }

if __name__ == "__main__":
    # Test Verification Execution: Passing completely production arbitrary strings 
    # Let's test an intense mountainous corridor across the South American Andes: Santiago to Mendoza
    origin_input = "Santiago, Chile"
    destination_input = "Mendoza, Argentina"
    
    engine = UniversalCorridorDataEngine()
    try:
        data = engine.fetch_live_corridor_features(origin_input, destination_input, sample_points=8)
        
        print("\n" + "="*125)
        print(f"UNIVERSAL REAL-TIME INGESTION MATRIX | Track Scale: {data['distance_km']} km | OSRM Base ETA: {data['base_duration_min']} mins")
        print("="*125)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        output_view = data['feature_dataframe'][[
            "checkpoint_pct", "latitude", "longitude", "h3_cell", 
            "elevation_meters", "incline_gradient_pct", "temp_celsius", "precipitation_mm"
        ]]
        print(output_view.to_string(index=False))
        print("="*125 + "\n")
    except Exception as e:
        print(f"Universal Pipeline Execution Interrupted: {e}")