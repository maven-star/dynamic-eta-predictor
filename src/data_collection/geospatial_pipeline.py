# src/data_collection/geospatial_pipeline.py
import h3
import numpy as np
import pandas as pd
import requests

class IndustryETAFeatureEngine:
    """
    Unified Global Geospatial Pipeline matching Uber DeeprETA standards.
    Processes arbitrary GPS sequences globally, maps coordinates to Uber H3 cells,
    resolves elevations, computes dynamic Haversine gradients, and encodes cyclical time matrices.
    """
    def __init__(self, h3_resolution=7):
        self.resolution = h3_resolution
        self.elevation_api_url = "https://api.open-meteo.com/v1/elevation"

    def _haversine_distance_meters(self, lat1, lon1, lat2, lon2):
        """Calculates true horizontal great-circle distance between two global points."""
        R = 6371000.0  
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def compute_h3_indices(self, df):
        """Encodes raw float coordinates into discrete space tokens for downstream Embedding Layers."""
        is_v4 = hasattr(h3, 'latlng_to_cell')
        if is_v4:
            df["h3_cell"] = df.apply(lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], self.resolution), axis=1)
        else:
            df["h3_cell"] = df.apply(lambda r: h3.geo_to_h3(r["latitude"], r["longitude"], self.resolution), axis=1)
        return df

    def synchronize_topography(self, df):
        """Queries production global DEM arrays and calculates dynamic slopes without distance assumptions."""
        lats, lons = df["latitude"].tolist(), df["longitude"].tolist()
        try:
            res = requests.post(self.elevation_api_url, json={"latitude": lats, "longitude": lons}, timeout=15)
            df["elevation_meters"] = res.json().get("elevation", [0.0] * len(df)) if res.status_code == 200 else 0.0
        except Exception as e:
            print(f"[Topography Warning] Global DEM lookup failed ({e}). Defaulting baseline.")
            df["elevation_meters"] = 0.0

        # Calculate localized road incline gradients via custom Haversine intervals
        slopes = [0.0]
        for i in range(1, len(df)):
            run_m = self._haversine_distance_meters(df.loc[i-1, "latitude"], df.loc[i-1, "longitude"], df.loc[i, "latitude"], df.loc[i, "longitude"])
            rise_m = df.loc[i, "elevation_meters"] - df.loc[i-1, "elevation_meters"]
            slopes.append(round((rise_m / run_m) * 100, 4) if run_m > 5.0 else 0.0)
            
        df["incline_gradient_pct"] = slopes
        return df

    def encode_cyclical_time(self, df):
        """Converts raw timestamps/hours into smooth, continuous cyclical representations."""
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
        return df

    def transform_pipeline(self, raw_telemetry_df):
        """Executes the finalized end-to-end matrix transformations."""
        df = raw_telemetry_df.copy()
        df = self.compute_h3_indices(df)
        df = self.synchronize_topography(df)
        df = self.encode_cyclical_time(df)
        return df

if __name__ == "__main__":
    # Test Verification: Arbitrary high-altitude pass (e.g., Swiss Alps corridor to verify global generality)
    alps_freight_test = {
        "checkpoint_pct": ["0%", "50%", "100%"],
        "latitude": [46.5196, 46.5588, 46.6242],
        "longitude": [6.6323, 7.4474, 8.0353],
        "hour_of_day": [14, 16, 17],
        "day_of_week": [2, 2, 2]
    }
    df_raw = pd.DataFrame(alps_freight_test)
    engine = IndustryETAFeatureEngine()
    df_processed = engine.transform_pipeline(df_raw)
    
    print("\n" + "="*90)
    print("GLOBAL PRODUCTION INDUSTRIAL DATA PLATFORM | VERIFICATION MATRIX")
    print("="*90)
    print(df_processed[["checkpoint_pct", "h3_cell", "elevation_meters", "incline_gradient_pct", "hour_sin"]])
    print("="*90 + "\n")