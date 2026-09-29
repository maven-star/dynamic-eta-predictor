# src/data_collection/get_topography.py
import requests
import pandas as pd
import numpy as np

class TerrainProfiler:
    """
    Production-grade generalized terrain profile analyzer. Computes elevation 
    via global DEM queries and determines localized road gradients dynamically 
    using Haversine point-to-point calculations anywhere on Earth.
    """
    def __init__(self):
        # Using the highly reliable Open-Meteo Elevation API (SRTM / Copernicus 90m resolution)
        self.api_url = "https://api.open-meteo.com/v1/elevation"

    def _haversine_distance_meters(self, lat1, lon1, lat2, lon2):
        """Calculates the true great-circle distance between two points in meters."""
        R = 6371000.0  # Earth's radius in meters
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def profile_dataframe(self, df):
        """
        Processes any spatial dataframe containing 'latitude' and 'longitude'.
        Synchronizes exact elevation metrics and calculates custom incline percentages.
        """
        print("[Topography] Constructing generalized vertical terrain layer via global DEM...")
        
        if df.empty:
            return df

        # 1. Fetch Elevation Profiles via Batch POST
        lats = df["latitude"].tolist()
        lons = df["longitude"].tolist()
        try:
            response = requests.get(self.api_url, params={"latitude": ",".join(map(str, lats)), "longitude": ",".join(map(str, lons))}, timeout=15)
            if response.status_code == 200:
                data = response.json()
                df["elevation_meters"] = data.get("elevation", [0.0] * len(df))
            else:
                print(f"[Topography Warning] API error code {response.status_code}. Using 0.0 baseline.")
                df["elevation_meters"] = 0.0
        except Exception as e:
            print(f"[Topography Warning] Network DEM query failed ({e}). Using 0.0 baseline.")
            df["elevation_meters"] = 0.0

        # 2. Dynamic Slope / Incline Generation via Haversine
        slopes = [0.0]  # First coordinate point has zero baseline incoming slope
        
        for i in range(1, len(df)):
            lat1, lon1 = df.loc[i-1, "latitude"], df.loc[i-1, "longitude"]
            lat2, lon2 = df.loc[i, "latitude"], df.loc[i, "longitude"]
            
            # Calculate true horizontal run (distance)
            run_meters = self._haversine_distance_meters(lat1, lon1, lat2, lon2)
            
            # Vertical rise
            rise_meters = df.loc[i, "elevation_meters"] - df.loc[i-1, "elevation_meters"]
            
            if run_meters > 5.0:  # Avoid division by zero or jitter errors on stationary pings
                slope_pct = (rise_meters / run_meters) * 100
                slopes.append(round(slope_pct, 4))
            else:
                slopes.append(0.0)
                
        df["incline_gradient_pct"] = slopes
        return df

if __name__ == "__main__":
    # Test script with your known Delhi to Kasol milestones
    test_route_data = {
        "checkpoint_pct": ["0%", "11%", "22%", "33%", "44%", "55%", "66%", "77%", "88%", "100%"],
        "latitude": [28.7352, 30.1435, 30.8493, 31.2933, 31.5406, 31.7797, 31.8254, 31.8390, 31.9531, 32.0098],
        "longitude": [77.1430, 76.8667, 76.5627, 76.7418, 76.8883, 76.9869, 77.0749, 77.1272, 77.1800, 77.3110]
    }
    df = pd.DataFrame(test_route_data)
    profiler = TerrainProfiler()
    df_profiled = profiler.profile_dataframe(df)
    
    print("\n--- Globally Profiled Spatial Matrix (Delhi to Kasol Verification) ---")
    print(df_profiled[["checkpoint_pct", "latitude", "longitude", "elevation_meters", "incline_gradient_pct"]])
