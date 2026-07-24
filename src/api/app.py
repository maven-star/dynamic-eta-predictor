from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import requests
import torch
import uuid
import json

from src.models.deepreta_system import DeeprETAEndToEndSystem

app = FastAPI()

MODEL_PATH = "src/models/deepreta_trained.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = DeeprETAEndToEndSystem()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Loaded DeeprETA weights from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load weights: {e}")
    model = None


def geocode_city(city_name: str, headers: dict) -> tuple:
    geo_url = "https://nominatim.openstreetmap.org/search"
    res = requests.get(
        geo_url,
        params={"q": city_name, "format": "json", "limit": 1},
        headers=headers,
    ).json()
    if not res:
        raise ValueError(f"Could not find coordinates for '{city_name}'")
    return float(res[0]["lat"]), float(res[0]["lon"])


def reverse_geocode_town(lat: float, lon: float, headers: dict) -> str:
    """Finds the city, town, or district name for a lat/lon point."""
    rev_url = "https://nominatim.openstreetmap.org/reverse"
    try:
        res = requests.get(
            rev_url,
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 10},
            headers=headers,
            timeout=2,
        ).json()
        address = res.get("address", {})
        return (
            address.get("city")
            or address.get("town")
            or address.get("state_district")
            or address.get("county")
            or ""
        )
    except Exception:
        return ""


def get_osrm_route(
    lat_org: float, lon_org: float, lat_dst: float, lon_dst: float, headers: dict
) -> tuple:
    osrm_url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{lon_org},{lat_org};{lon_dst},{lat_dst}"
        f"?overview=full&geometries=geojson"
    )
    res = requests.get(osrm_url).json()

    if "routes" in res and res["routes"]:
        route = res["routes"][0]
        distance_km = round(route["distance"] / 1000.0, 1)
        geometry = route["geometry"]["coordinates"]
        route_coords = [[pt[1], pt[0]] for pt in geometry]

        # Break down route into key intermediate stops (sampling 4 intermediate points)
        num_points = len(route_coords)
        intermediate_names = []

        if num_points > 10:
            sample_indices = [
                int(num_points * 0.2),
                int(num_points * 0.4),
                int(num_points * 0.6),
                int(num_points * 0.8),
            ]
            for idx in sample_indices:
                pt = route_coords[idx]
                town = reverse_geocode_town(pt[0], pt[1], headers)
                if town and (
                    not intermediate_names or town != intermediate_names[-1]
                ):
                    intermediate_names.append(town)

        return distance_km, route_coords, intermediate_names

    return 250.0, [[lat_org, lon_org], [lat_dst, lon_dst]], []


def latlng_to_mock_h3(lat: float, lng: float) -> str:
    return "8826801431fffff"


def predict_eta_with_nn(origin: str, destination: str) -> dict:
    o = origin.strip().title()
    d = destination.strip().title()

    headers = {"User-Agent": f"deepreta_app_{uuid.uuid4().hex[:8]}"}

    lat_org, lon_org = geocode_city(o, headers)
    lat_dst, lon_dst = geocode_city(d, headers)
    distance_km, route_coords, waypoints = get_osrm_route(
        lat_org, lon_org, lat_dst, lon_dst, headers
    )

    # Clean route breakdown: Origin ➔ Waypoints ➔ Destination
    clean_origin = o.split(",")[0]
    clean_dest = d.split(",")[0]

    filtered_waypoints = [
        w
        for w in waypoints
        if w.lower() not in clean_origin.lower()
        and w.lower() not in clean_dest.lower()
    ]

    route_breakdown = [clean_origin] + filtered_waypoints + [clean_dest]
    route_breakdown_str = " ➔ ".join(route_breakdown)

    h3_base = latlng_to_mock_h3(lat_org, lon_org)
    h3_strings = [h3_base] * 8
    continuous_features = torch.full(
        (8, 6), fill_value=10, dtype=torch.long
    ).to(device)
    driver_profile = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(
        device
    )

    with torch.no_grad():
        if model is not None:
            predicted_quantiles = model.predict_eta(
                h3_strings=h3_strings,
                continuous_features=continuous_features,
                driver_profile=driver_profile,
            )

            pred_values = predicted_quantiles.cpu().numpy().flatten()
            raw_p10, raw_p50, raw_p90 = (
                pred_values[0],
                pred_values[1],
                pred_values[2],
            )
            base_time_mins = (distance_km / 60.0) * 60

            eta_10 = int(max(1, base_time_mins * raw_p10))
            eta_50 = int(max(eta_10 + 1, base_time_mins * raw_p50))
            eta_90 = int(max(eta_50 + 1, base_time_mins * raw_p90))
        else:
            raise RuntimeError("Model weights not loaded.")

    return {
        "origin": o,
        "destination": d,
        "route_breakdown": route_breakdown_str,
        "origin_coords": [lat_org, lon_org],
        "destination_coords": [lat_dst, lon_dst],
        "distance": f"{distance_km} km",
        "route_coords": route_coords,
        "eta_10": f"{eta_10 // 60}h {eta_10 % 60}m"
        if eta_10 >= 60
        else f"{eta_10} mins",
        "eta_50": f"{eta_50 // 60}h {eta_50 % 60}m"
        if eta_50 >= 60
        else f"{eta_50} mins",
        "eta_90": f"{eta_90 // 60}h {eta_90 % 60}m"
        if eta_90 >= 60
        else f"{eta_90} mins",
    }


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeeprETA Route Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        :root {{
            --bg-body: #f8fafc;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --card-bg: #ffffff;
            --card-border: #e2e8f0;
            --input-bg: #f1f5f9;
            --input-border: #cbd5e1;
            --input-text: #0f172a;
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --header-text: #4338ca;
            
            --p10-color: #15803d;
            --p50-color: #1d4ed8;
            --p90-color: #b91c1c;
            --card-accent: #4f46e5;
        }}

        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg-body: #0f172a;
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
                --card-bg: #1e293b;
                --card-border: #334155;
                --input-bg: #334155;
                --input-border: #475569;
                --input-text: #ffffff;
                --primary: #6366f1;
                --primary-hover: #4f46e5;
                --header-text: #818cf8;
                
                --p10-color: #4ade80;
                --p50-color: #60a5fa;
                --p90-color: #f87171;
                --card-accent: #6366f1;
            }}
        }}

        body {{
            background-color: var(--bg-body);
            color: var(--text-main);
            font-family: system-ui, -apple-system, sans-serif;
            transition: background-color 0.3s ease, color 0.3s ease;
        }}

        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }}

        .brand-title {{ color: var(--header-text); }}

        .btn-primary {{
            background-color: var(--primary);
            border: none;
            color: #ffffff;
        }}
        .btn-primary:hover {{ background-color: var(--primary-hover); }}

        .form-control {{
            background-color: var(--input-bg);
            border: 1px solid var(--input-border);
            color: var(--input-text);
        }}

        .result-card {{ border-left: 4px solid var(--card-accent) !important; }}
        .text-custom-muted {{ color: var(--text-muted); }}

        .color-p10 {{ color: var(--p10-color); }}
        .color-p50 {{ color: var(--p50-color); }}
        .color-p90 {{ color: var(--p90-color); }}

        .badge-route {{
            background-color: var(--input-bg);
            color: var(--text-main);
            border: 1px solid var(--card-border);
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 0.95rem;
            display: inline-block;
        }}

        #map {{
            height: 380px;
            width: 100%;
            border-radius: 10px;
            margin-top: 15px;
            border: 1px solid var(--card-border);
        }}
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-7 col-md-9">
                
                <div class="card p-4 mb-4">
                    <h3 class="text-center mb-4 brand-title">🧠 DeeprETA Route Predictor</h3>
                    <form action="/" method="POST">
                        <div class="row">
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Origin City</label>
                                <input type="text" name="origin" class="form-control" placeholder="e.g. Delhi" value="{origin}" required>
                            </div>
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Destination City</label>
                                <input type="text" name="destination" class="form-control" placeholder="e.g. Chandigarh" value="{destination}" required>
                            </div>
                        </div>
                        <button type="submit" class="btn btn-primary w-100 py-2">Query Route & Predict ETA</button>
                    </form>
                </div>

                {result_html}

            </div>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    {map_script}
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE.format(
        origin="", destination="", result_html="", map_script=""
    )


@app.post("/", response_class=HTMLResponse)
async def predict(origin: str = Form(...), destination: str = Form(...)):
    try:
        prediction = predict_eta_with_nn(origin, destination)

        result_html = f"""
        <div class="card p-4 result-card">
            <h5 class="mb-3 brand-title">Model Output & Travel Route</h5>
            
            <div class="mb-3">
                <div class="text-custom-muted mb-1"><strong>Route Corridor Breakdown:</strong></div>
                <div class="badge-route">📍 {prediction["route_breakdown"]}</div>
            </div>

            <div class="mb-3 text-custom-muted">
                <strong>Real Road Distance:</strong> <span>{prediction["distance"]}</span>
            </div>
            
            <div id="map"></div>

            <hr style="border-color: var(--card-border);" class="my-4">
            
            <div class="d-flex justify-content-between mb-2">
                <span class="color-p10"><strong>10th Percentile (Optimistic):</strong></span>
                <span class="color-p10"><strong>{prediction["eta_10"]}</strong></span>
            </div>
            <div class="d-flex justify-content-between mb-2 fs-5">
                <span class="color-p50"><strong>50th Percentile (Most Likely):</strong></span>
                <strong class="color-p50">{prediction["eta_50"]}</strong>
            </div>
            <div class="d-flex justify-content-between mb-2">
                <span class="color-p90"><strong>90th Percentile (Pessimistic):</strong></span>
                <span class="color-p90"><strong>{prediction["eta_90"]}</strong></span>
            </div>
        </div>
        """

        map_script = f"""
        <script>
            const routeCoords = {json.dumps(prediction["route_coords"])};
            const originCoords = {json.dumps(prediction["origin_coords"])};
            const destCoords = {json.dumps(prediction["destination_coords"])};

            const map = L.map('map');

            const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            const tileUrl = isDark 
                ? 'https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png'
                : 'https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png';

            L.tileLayer(tileUrl, {{
                maxZoom: 19,
                attribution: '&copy; OpenStreetMap &copy; CARTO'
            }}).addTo(map);

            const polyline = L.polyline(routeCoords, {{
                color: isDark ? '#818cf8' : '#4f46e5',
                weight: 5,
                opacity: 0.85
            }}).addTo(map);

            L.marker(originCoords).addTo(map).bindPopup("<b>Origin:</b> {prediction['origin']}");
            L.marker(destCoords).addTo(map).bindPopup("<b>Destination:</b> {prediction['destination']}");

            map.fitBounds(polyline.getBounds(), {{ padding: [30, 30] }});
        </script>
        """

    except Exception as e:
        result_html = f"""
        <div class="alert alert-danger p-3" role="alert">
            <strong>Inference Failure:</strong> {str(e)}
        </div>
        """
        map_script = ""

    return HTML_TEMPLATE.format(
        origin=origin,
        destination=destination,
        result_html=result_html,
        map_script=map_script,
    )