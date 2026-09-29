"""FastAPI application for route ETA prediction."""

from __future__ import annotations

import hashlib
import html
import json
import os
import threading
import time
from datetime import datetime
from dataclasses import dataclass
from math import cos, pi
from pathlib import Path
from typing import Any

import h3
import requests
import torch
from fastapi import FastAPI, Form, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ValidationError, model_validator

from src.config import load_project_config
from src.data_collection.realtime_provider import LiveRoute, Place, ProviderError, TomTomRealtimeProvider
from src.data_collection.trip_store import TripStore
from src.models.deepreta_system import DeeprETAEndToEndSystem

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("ETA_MODEL_PATH", ROOT / "src/models/deepreta_trained.pth"))
REQUEST_TIMEOUT_SECONDS = 10
settings = load_project_config()
SAMPLE_POINTS = settings.sequence_length
NOMINATIM_INTERVAL_SECONDS = 1.0
USER_AGENT = os.getenv("ETA_USER_AGENT", "dynamic-eta-predictor/1.0 (local demo)")
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "").strip()
TRIP_DB_PATH = Path(os.getenv("ETA_TRIP_DB_PATH", ROOT / "data/eta_events.sqlite3"))
CALIBRATION_REQUESTED = os.getenv("ETA_MODEL_CALIBRATED", "false").lower() == "true"
MODEL_REPORT_PATH = Path(os.getenv("ETA_MODEL_REPORT_PATH", str(MODEL_PATH.with_suffix(".metrics.json"))))

app = FastAPI(title="Dynamic ETA Predictor", version="1.0.0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model() -> DeeprETAEndToEndSystem:
    model = DeeprETAEndToEndSystem(
        vocab_size=settings.h3_vocab_size,
        h3_dim=settings.h3_embedding_dim,
        continuous_dim=settings.continuous_embedding_dim,
        max_seq_len=settings.sequence_length,
        embed_dim=settings.embedding_dim,
        linformer_k=settings.linformer_projection_k,
    )
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for block in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_calibration_status() -> tuple[bool, str]:
    """Require an eligible report bound cryptographically to this checkpoint."""
    if not CALIBRATION_REQUESTED:
        return False, "learned quantiles disabled until ETA_MODEL_CALIBRATED=true"
    try:
        report = json.loads(MODEL_REPORT_PATH.read_text(encoding="utf-8"))
        schema_version = report["schema_version"]
        expected_hash = report["checkpoint"]["sha256"]
        eligible = report["promotion"]["eligible"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        return False, f"calibration report unavailable or invalid: {error}"
    if schema_version != 1:
        return False, "calibration report schema is not supported"
    if eligible is not True:
        return False, "calibration report did not pass promotion gates"
    try:
        if not isinstance(expected_hash, str) or _checkpoint_sha256(MODEL_PATH) != expected_hash:
            return False, "calibration report does not match the configured checkpoint"
    except OSError as error:
        return False, f"unable to verify checkpoint checksum: {error}"
    return True, "eligible chronological test report verified"


try:
    model: DeeprETAEndToEndSystem | None = _load_model()
    model_load_error: str | None = None
except Exception as error:
    model = None
    model_load_error = str(error)

MODEL_CALIBRATED, model_calibration_status = _load_calibration_status()


class RouteRequest(BaseModel):
    origin: str = Field(min_length=2, max_length=160)
    destination: str = Field(min_length=2, max_length=160)
    origin_latitude: float | None = Field(default=None, ge=-90, le=90)
    origin_longitude: float | None = Field(default=None, ge=-180, le=180)
    destination_latitude: float | None = Field(default=None, ge=-90, le=90)
    destination_longitude: float | None = Field(default=None, ge=-180, le=180)
    origin_label: str | None = Field(default=None, min_length=2, max_length=200)
    destination_label: str | None = Field(default=None, min_length=2, max_length=200)
    speed_multiplier: float = Field(default=1.0, ge=0.5, le=1.5)
    harsh_braking: float = Field(default=1.0, ge=0.0, le=10.0)
    aggressive_acceleration: float = Field(default=1.0, ge=0.0, le=10.0)
    vehicle_type: str = Field(default="unknown", min_length=1, max_length=64)
    request_type: str = Field(default="mountain_trip", min_length=1, max_length=64)

    @model_validator(mode="after")
    def _require_complete_selected_places(self) -> "RouteRequest":
        for field_name in ("origin", "destination"):
            latitude = getattr(self, f"{field_name}_latitude")
            longitude = getattr(self, f"{field_name}_longitude")
            if (latitude is None) != (longitude is None):
                raise ValueError(f"{field_name} latitude and longitude must be provided together")
        return self


class PositionEventRequest(BaseModel):
    occurred_at: datetime
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    speed_kph: float | None = Field(default=None, ge=0, le=250)
    heading_degrees: float | None = Field(default=None, ge=0, lt=360)
    source: str = Field(default="manual", min_length=1, max_length=32)


class CompletionRequest(BaseModel):
    completed_at: datetime
    actual_duration_minutes: float = Field(gt=0, le=10_080)
    source: str = Field(default="manual", min_length=1, max_length=32)


@dataclass(frozen=True)
class Route:
    distance_km: float
    base_duration_minutes: float
    coordinates: list[list[float]]
    free_flow_minutes: float | None = None
    traffic_delay_minutes: float | None = None
    provider: str = "osrm_static"


class PublicRouteClient:
    """Bounded public-service client with Nominatim pacing and caching."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self._cache: dict[tuple[str, str], tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self._last_nominatim_at = 0.0

    def get(self, namespace: str, key: str, url: str, params: dict[str, Any], ttl: int = 900) -> Any:
        cache_key = (namespace, key)
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and time.monotonic() - cached[0] < ttl:
                return cached[1]
        try:
            response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            raise HTTPException(status_code=503, detail=f"Upstream {namespace} service is unavailable") from error
        with self._lock:
            self._cache[cache_key] = (time.monotonic(), payload)
        return payload

    def geocode(self, location: str) -> tuple[float, float, str]:
        places = self.search_places(location, limit=1)
        if not places:
            raise HTTPException(status_code=422, detail=f"Could not geocode '{location}'")
        place = places[0]
        return place.latitude, place.longitude, place.label

    def search_places(self, location: str, limit: int = 6) -> list[Place]:
        query = location.strip()
        if len(query) < 2:
            return []
        cache_key = f"search:{query.casefold()}"
        with self._lock:
            cached = self._cache.get(("geocoding", cache_key))
            delay = 0.0 if cached else NOMINATIM_INTERVAL_SECONDS - (time.monotonic() - self._last_nominatim_at)
        if delay > 0:
            time.sleep(delay)
        payload = self.get(
            "geocoding", cache_key, "https://nominatim.openstreetmap.org/search",
            {"q": query, "format": "jsonv2", "limit": max(1, min(limit, 10)), "addressdetails": 1}, ttl=3600,
        )
        with self._lock:
            self._last_nominatim_at = time.monotonic()
        places: list[Place] = []
        for result in payload:
            try:
                places.append(Place(float(result["lat"]), float(result["lon"]), str(result["display_name"])))
            except (KeyError, TypeError, ValueError):
                continue
        return places

    def route(self, origin: tuple[float, float], destination: tuple[float, float]) -> Route:
        lat_o, lon_o = origin
        lat_d, lon_d = destination
        payload = self.get(
            "routing", f"{lat_o:.5f},{lon_o:.5f}:{lat_d:.5f},{lon_d:.5f}",
            f"https://router.project-osrm.org/route/v1/driving/{lon_o},{lat_o};{lon_d},{lat_d}",
            {"overview": "full", "geometries": "geojson"},
        )
        routes = payload.get("routes", [])
        if not routes:
            raise HTTPException(status_code=422, detail="No drivable route was found")
        route = routes[0]
        duration = float(route["duration"]) / 60.0
        return Route(float(route["distance"]) / 1000.0, duration, [[float(lat), float(lon)] for lon, lat in route["geometry"]["coordinates"]], duration, 0.0, "osrm_static")


class RouteFeatureProvider:
    """Derive the model's six real features from sampled route geometry."""

    def __init__(self, client: PublicRouteClient) -> None:
        self.client = client

    @staticmethod
    def sample(coordinates: list[list[float]], count: int = SAMPLE_POINTS) -> list[list[float]]:
        if not coordinates:
            raise ValueError("route geometry is empty")
        if len(coordinates) <= count:
            return coordinates
        return [coordinates[round(index * (len(coordinates) - 1) / (count - 1))] for index in range(count)]

    def elevations(self, points: list[list[float]]) -> list[float]:
        latitudes = ",".join(f"{lat:.5f}" for lat, _ in points)
        longitudes = ",".join(f"{lon:.5f}" for _, lon in points)
        try:
            payload = self.client.get("elevation", latitudes + ":" + longitudes, "https://api.open-meteo.com/v1/elevation", {"latitude": latitudes, "longitude": longitudes}, ttl=86400)
            values = payload.get("elevation", [])
            if len(values) == len(points):
                return [float(value) for value in values]
        except HTTPException:
            pass
        return [0.0] * len(points)

    def weather(self, latitude: float, longitude: float) -> tuple[float, float, float, float]:
        try:
            payload = self.client.get(
                "weather", f"{latitude:.2f},{longitude:.2f}", "https://api.open-meteo.com/v1/forecast",
                {"latitude": latitude, "longitude": longitude, "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"}, ttl=600,
            )
            current = payload.get("current", {})
            return (float(current.get("temperature_2m", 20.0)), float(current.get("relative_humidity_2m", 60.0)), float(current.get("wind_speed_10m", 10.0)), float(current.get("precipitation", 0.0)))
        except HTTPException:
            return (20.0, 60.0, 10.0, 0.0)

    @staticmethod
    def grade(previous: list[float], current: list[float], previous_elevation: float, elevation: float) -> float:
        lat1, lon1 = previous
        lat2, lon2 = current
        north_m = (lat2 - lat1) * 111_320.0
        east_m = (lon2 - lon1) * 111_320.0 * cos((lat1 + lat2) * 0.5 * pi / 180.0)
        run = max((north_m * north_m + east_m * east_m) ** 0.5, 1.0)
        return max(-30.0, min(30.0, (elevation - previous_elevation) / run * 100.0))

    def build(self, route: Route) -> tuple[list[str], torch.Tensor]:
        points = self.sample(route.coordinates)
        elevations = self.elevations(points)
        cells, features = [], []
        for index, (latitude, longitude) in enumerate(points):
            weather = self.weather(latitude, longitude)
            grade = 0.0 if index == 0 else self.grade(points[index - 1], points[index], elevations[index - 1], elevations[index])
            cell = h3.latlng_to_cell(latitude, longitude, settings.h3_resolution) if hasattr(h3, "latlng_to_cell") else h3.geo_to_h3(latitude, longitude, settings.h3_resolution)
            cells.append(cell)
            features.append([elevations[index], grade, *weather])
        return cells, torch.tensor(features, dtype=torch.float32)


route_client = PublicRouteClient()
feature_provider = RouteFeatureProvider(route_client)
tomtom_provider = TomTomRealtimeProvider(TOMTOM_API_KEY) if TOMTOM_API_KEY else None
trip_store = TripStore(TRIP_DB_PATH)


def resolve_route(
    origin: str,
    destination: str,
    origin_coordinates: tuple[float, float] | None = None,
    destination_coordinates: tuple[float, float] | None = None,
    origin_label: str | None = None,
    destination_label: str | None = None,
) -> tuple[Route, str, str]:
    """Use live traffic whenever a TomTom key is configured."""
    if tomtom_provider is not None:
        try:
            origin_place = (
                Place(origin_coordinates[0], origin_coordinates[1], origin_label or "Latest vehicle position")
                if origin_coordinates is not None
                else tomtom_provider.geocode(origin)
            )
            destination_place = (
                Place(destination_coordinates[0], destination_coordinates[1], destination_label or destination)
                if destination_coordinates is not None
                else tomtom_provider.geocode(destination)
            )
            live: LiveRoute = tomtom_provider.calculate_route(origin_place, destination_place)
            return (
                Route(live.distance_km, live.travel_time_minutes, live.coordinates, live.free_flow_minutes, live.traffic_delay_minutes, live.provider),
                origin_place.label,
                destination_place.label,
            )
        except ProviderError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
    if origin_coordinates is not None:
        origin_lat, origin_lon, origin_name = origin_coordinates[0], origin_coordinates[1], origin_label or "Latest vehicle position"
    else:
        origin_lat, origin_lon, origin_name = route_client.geocode(origin)
    if destination_coordinates is not None:
        destination_lat, destination_lon, destination_name = (
            destination_coordinates[0], destination_coordinates[1], destination_label or destination
        )
    else:
        destination_lat, destination_lon, destination_name = route_client.geocode(destination)
    return route_client.route((origin_lat, origin_lon), (destination_lat, destination_lon)), origin_name, destination_name


def predict_route(
    request: RouteRequest, origin_coordinates: tuple[float, float] | None = None
) -> dict[str, Any]:
    selected_origin = origin_coordinates or (
        (request.origin_latitude, request.origin_longitude)
        if request.origin_latitude is not None and request.origin_longitude is not None
        else None
    )
    selected_destination = (
        (request.destination_latitude, request.destination_longitude)
        if request.destination_latitude is not None and request.destination_longitude is not None
        else None
    )
    route, origin_name, destination_name = resolve_route(
        request.origin,
        request.destination,
        selected_origin,
        selected_destination,
        "Latest vehicle position" if origin_coordinates is not None else request.origin_label,
        request.destination_label,
    )
    if model is None:
        raise HTTPException(status_code=503, detail="ETA model is unavailable; see /health")
    # Every inference uses live route geometry, H3 cells, terrain and weather.
    # TomTom's traffic-aware duration is the base time; the network predicts
    # ordered route-condition multipliers around that real-time baseline.
    cells, features = feature_provider.build(route)
    driver = torch.tensor(
        [request.speed_multiplier, request.harsh_braking, request.aggressive_acceleration], dtype=torch.float32
    )
    with torch.inference_mode():
        multipliers = model.predict_eta(cells, features, driver).detach().cpu().tolist()
    raw_quantiles = [max(1, round(route.base_duration_minutes * float(multiplier))) for multiplier in multipliers]
    if MODEL_CALIBRATED:
        quantiles = raw_quantiles
        model_status = "calibrated model quantiles; chronological test report verified"
    else:
        # Never let an unvalidated checkpoint claim a faster median than the
        # live routing provider. This safety anchor prevents the synthetic
        # bundled model from turning a 10-hour mountain route into 2 hours.
        baseline = max(1, round(route.base_duration_minutes))
        quantiles = [max(1, round(baseline * 0.80)), baseline, round(baseline * 1.30)]
        model_status = f"baseline-anchored intervals; {model_calibration_status}"
    flow = None
    if tomtom_provider is not None:
        try:
            flow = tomtom_provider.flow_at(*route.coordinates[0])
        except ProviderError:
            flow = None
    return {
        "origin": origin_name,
        "destination": destination_name,
        "distance_km": round(route.distance_km, 1),
        "eta_minutes": {"p10": quantiles[0], "p50": quantiles[1], "p90": quantiles[2]},
        "traffic_baseline_minutes": round(route.base_duration_minutes),
        "free_flow_minutes": round(route.free_flow_minutes) if route.free_flow_minutes is not None else None,
        "traffic_delay_minutes": round(route.traffic_delay_minutes) if route.traffic_delay_minutes is not None else None,
        "traffic_available": route.provider == "tomtom_live_traffic",
        "traffic_flow": None if flow is None else {"current_speed_kph": flow.current_speed_kph, "free_flow_speed_kph": flow.free_flow_speed_kph, "confidence": flow.confidence},
        "provider": route.provider,
        "route_coordinates": route.coordinates,
        "feature_points": len(cells),
        "feature_snapshot": {
            "h3_cells": cells,
            "continuous_features": features.tolist(),
            "driver_profile": driver.tolist(),
            "routing_eta_minutes": route.base_duration_minutes,
            "traffic_delay_minutes": route.traffic_delay_minutes,
            "vehicle_type": request.vehicle_type,
            "request_type": request.request_type,
        },
        "model_status": model_status,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if model is not None else "degraded",
        "traffic_provider": "tomtom_live_traffic" if tomtom_provider else "osrm_static_fallback",
        "model_loaded": model is not None,
        "model_error": model_load_error,
        "learned_quantiles_enabled": MODEL_CALIBRATED,
        "calibration_status": model_calibration_status,
    }


@app.get("/api/places")
def suggest_places(
    query: str = Query(min_length=2, max_length=160),
    limit: int = Query(default=6, ge=1, le=10),
) -> dict[str, Any]:
    """Return precise selectable places; clients must retain the coordinates."""
    try:
        places = tomtom_provider.search_places(query, limit) if tomtom_provider is not None else route_client.search_places(query, limit)
    except ProviderError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "provider": "tomtom_search" if tomtom_provider is not None else "nominatim_fallback",
        "places": [
            {"label": place.label, "latitude": place.latitude, "longitude": place.longitude}
            for place in places
        ],
    }


@app.post("/api/predict")
def api_predict(request: RouteRequest) -> dict[str, Any]:
    prediction = predict_route(request)
    prediction.pop("feature_snapshot", None)
    return prediction


@app.post("/api/trips")
def create_trip(request: RouteRequest) -> dict[str, Any]:
    """Create a labelled-trip candidate and return its first quantile ETA."""
    import uuid

    prediction = predict_route(request)
    trip_id = str(uuid.uuid4())
    feature_snapshot = prediction.pop("feature_snapshot")
    trip_store.create_trip(trip_id, request.model_dump(mode="json"), prediction, feature_snapshot)
    return {"trip_id": trip_id, **prediction}


@app.post("/api/trips/{trip_id}/positions")
def record_position(trip_id: str, event: PositionEventRequest) -> dict[str, str]:
    try:
        trip_store.add_position(
            trip_id,
            event.occurred_at.isoformat(),
            event.latitude,
            event.longitude,
            event.speed_kph,
            event.heading_degrees,
            event.source,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"status": "recorded"}


@app.post("/api/trips/{trip_id}/complete")
def complete_trip(trip_id: str, completion: CompletionRequest) -> dict[str, str]:
    try:
        trip_store.complete_trip(
            trip_id, completion.completed_at.isoformat(), completion.actual_duration_minutes, completion.source
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"status": "completed"}


@app.get("/api/trips/{trip_id}")
def get_trip(trip_id: str) -> dict[str, Any]:
    trip = trip_store.get_trip(trip_id)
    if trip is None:
        raise HTTPException(status_code=404, detail="trip was not found")
    return trip


@app.post("/api/trips/{trip_id}/eta")
def refresh_remaining_eta(trip_id: str) -> dict[str, Any]:
    """Predict the remaining ETA from the latest persisted vehicle position."""
    trip = trip_store.get_trip(trip_id)
    if trip is None:
        raise HTTPException(status_code=404, detail="trip was not found")
    if trip["status"] != "active":
        raise HTTPException(status_code=409, detail="trip is already completed")
    if not trip["positions"]:
        raise HTTPException(status_code=409, detail="record at least one vehicle position before refreshing ETA")
    latest = trip["positions"][-1]
    request = RouteRequest.model_validate(trip["request"])
    prediction = predict_route(request, (float(latest["latitude"]), float(latest["longitude"])))
    prediction.pop("feature_snapshot", None)
    prediction["trip_id"] = trip_id
    prediction["position_timestamp"] = latest["occurred_at"]
    prediction["remaining_eta"] = True
    return prediction


PAGE = """<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Dynamic ETA Predictor</title>
<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'>
<style>
body{font-family:system-ui;max-width:900px;margin:2rem auto;padding:0 1rem;color:#102a43}
form{display:grid;grid-template-columns:1fr 1fr auto;gap:.65rem;align-items:start}
label{font-weight:650;font-size:.9rem;display:block;margin-bottom:.25rem}
input,button{box-sizing:border-box;padding:.7rem;width:100%;font:inherit;border:1px solid #9fb3c8;border-radius:.35rem}
button{width:auto;background:#1976d2;color:#fff;border:0;cursor:pointer;margin-top:1.65rem}
small{display:block;min-height:1.25rem;margin-top:.25rem;color:#52606d}.invalid{color:#b42318}.selected{color:#00796b}
#map{height:360px;margin-top:1rem}.warning{color:#8a5a00}@media(max-width:680px){form{grid-template-columns:1fr}button{margin-top:0;width:100%}}
</style></head><body>
<h1>Dynamic ETA Predictor</h1>
<p>Select an exact pickup and destination place. A city alone is ambiguous; choosing a suggestion stores its coordinates so routing starts at the right location.</p>
<form id='route-form' method='post'>
  <div><label for='origin'>Origin / pickup point</label><input id='origin' name='origin' list='origin-options' required maxlength='160' autocomplete='off' placeholder='e.g. Connaught Place, Delhi' value='__ORIGIN__'><datalist id='origin-options'></datalist><input id='origin_latitude' name='origin_latitude' type='hidden'><input id='origin_longitude' name='origin_longitude' type='hidden'><input id='origin_label' name='origin_label' type='hidden'><small id='origin-status'>Start typing, then choose a precise suggestion.</small></div>
  <div><label for='destination'>Destination / drop point</label><input id='destination' name='destination' list='destination-options' required maxlength='160' autocomplete='off' placeholder='e.g. Leh Main Market' value='__DESTINATION__'><datalist id='destination-options'></datalist><input id='destination_latitude' name='destination_latitude' type='hidden'><input id='destination_longitude' name='destination_longitude' type='hidden'><input id='destination_label' name='destination_label' type='hidden'><small id='destination-status'>Start typing, then choose a precise suggestion.</small></div>
  <button type='submit'>Predict ETA</button>
</form>
__RESULT__
<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>
<script>
(() => {
  const fields = ['origin', 'destination'];
  const timers = {};
  const status = (name, message, kind = '') => {
    const node = document.getElementById(`${name}-status`);
    node.textContent = message;
    node.className = kind;
  };
  const clearSelection = (name) => {
    for (const suffix of ['latitude', 'longitude', 'label']) document.getElementById(`${name}_${suffix}`).value = '';
  };
  const selectSuggestion = (name) => {
    const input = document.getElementById(name);
    const option = [...document.getElementById(`${name}-options`).options].find((item) => item.value === input.value);
    if (!option) { clearSelection(name); status(name, 'Choose one of the suggested precise places.', 'invalid'); return false; }
    document.getElementById(`${name}_latitude`).value = option.dataset.latitude;
    document.getElementById(`${name}_longitude`).value = option.dataset.longitude;
    document.getElementById(`${name}_label`).value = option.value;
    status(name, `Selected: ${option.value}`, 'selected');
    return true;
  };
  const fetchSuggestions = async (name) => {
    const input = document.getElementById(name);
    const query = input.value.trim();
    if (query.length < 2) { status(name, 'Type at least two characters.', ''); return; }
    status(name, 'Searching precise places...', '');
    try {
      const response = await fetch(`/api/places?query=${encodeURIComponent(query)}&limit=6`);
      if (!response.ok) throw new Error('place search unavailable');
      const payload = await response.json();
      const list = document.getElementById(`${name}-options`);
      list.replaceChildren();
      for (const place of payload.places) {
        const option = document.createElement('option');
        option.value = place.label;
        option.dataset.latitude = place.latitude;
        option.dataset.longitude = place.longitude;
        list.append(option);
      }
      status(name, payload.places.length ? 'Choose a suggestion to lock the exact coordinates.' : 'No precise place found. Try a landmark, neighborhood, or full address.', payload.places.length ? '' : 'invalid');
    } catch (_) { status(name, 'Place search is unavailable. Try again shortly.', 'invalid'); }
  };
  for (const name of fields) {
    const input = document.getElementById(name);
    input.addEventListener('input', () => {
      clearSelection(name);
      clearTimeout(timers[name]);
      timers[name] = setTimeout(() => fetchSuggestions(name), 300);
    });
    input.addEventListener('change', () => selectSuggestion(name));
  }
  document.getElementById('route-form').addEventListener('submit', (event) => {
    const selected = fields.map(selectSuggestion).every(Boolean);
    if (!selected) event.preventDefault();
  });
})();
</script>
__SCRIPT__
</body></html>"""


def render_page(origin: str, destination: str, result: str, script: str) -> str:
    return (
        PAGE.replace("__ORIGIN__", html.escape(origin, quote=True))
        .replace("__DESTINATION__", html.escape(destination, quote=True))
        .replace("__RESULT__", result)
        .replace("__SCRIPT__", script)
    )


def format_minutes(minutes: int) -> str:
    return f"{minutes // 60}h {minutes % 60}m" if minutes >= 60 else f"{minutes} min"


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return render_page("", "", "", "")


@app.post("/", response_class=HTMLResponse)
def predict(
    origin: str = Form(...),
    destination: str = Form(...),
    origin_latitude: float | None = Form(default=None),
    origin_longitude: float | None = Form(default=None),
    destination_latitude: float | None = Form(default=None),
    destination_longitude: float | None = Form(default=None),
    origin_label: str | None = Form(default=None),
    destination_label: str | None = Form(default=None),
) -> str:
    try:
        result = predict_route(RouteRequest(
            origin=origin, destination=destination,
            origin_latitude=origin_latitude, origin_longitude=origin_longitude,
            destination_latitude=destination_latitude, destination_longitude=destination_longitude,
            origin_label=origin_label, destination_label=destination_label,
        ))
        traffic_note = "Live traffic is active." if result["traffic_available"] else "Traffic is unavailable: OSRM static-route fallback is active. Set TOMTOM_API_KEY."
        delay = "" if result["traffic_delay_minutes"] is None else f" · traffic delay: {format_minutes(result['traffic_delay_minutes'])}"
        eta = result["eta_minutes"]
        result_html = f"<h2>{html.escape(result['origin'])} → {html.escape(result['destination'])}</h2><p>{result['distance_km']} km · traffic baseline: {format_minutes(result['traffic_baseline_minutes'])}{delay}</p><p><b>P10:</b> {format_minutes(eta['p10'])} · <b>P50:</b> {format_minutes(eta['p50'])} · <b>P90:</b> {format_minutes(eta['p90'])}</p><p class='warning'>{html.escape(traffic_note)} Model intervals require completed-trip calibration.</p><div id='map'></div>"
        coordinates = json.dumps(result["route_coordinates"])
        script = f"<script>const m=L.map('map');const c={coordinates};const p=L.polyline(c).addTo(m);m.fitBounds(p.getBounds(),{{padding:[20,20]}});L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:19,attribution:'© OpenStreetMap'}}).addTo(m);</script>"
        return render_page(origin, destination, result_html, script)
    except (HTTPException, ValidationError, ValueError) as error:
        detail = error.detail if isinstance(error, HTTPException) else str(error)
        return render_page(origin, destination, f"<p role='alert'>{html.escape(str(detail))}</p>", "")
