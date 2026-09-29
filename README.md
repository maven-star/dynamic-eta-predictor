# Dynamic ETA Predictor

An ETA service with a real-time traffic mode. With a TomTom API key it uses
TomTom's live-traffic routing duration and current road-segment speed as its
ETA source. OSRM remains a clearly labelled static-route fallback for local
development only.

## Status and limits

The API feeds live H3, terrain, weather, driver-profile, and traffic-aware
routing inputs to the neural model on every request. The bundled checkpoint is
still uncalibrated because it was not trained on completed real trips; treat
its p10/p50/p90 intervals as development output until it is replaced and
validated against your collected trip records.

Until then, serving anchors p50 to the live traffic-routing ETA and returns a
conservative development interval. Only set `ETA_MODEL_CALIBRATED=true` after
the candidate's chronological test report passes its promotion gates; the API
also verifies that the report's SHA-256 belongs to the configured checkpoint.
This prevents an untrained or swapped checkpoint from reporting implausibly
short mountain ETAs.

`sim_data/raw_telemetry.csv` is not currently trainable: it lacks route
sequences/H3 cells and usable duration labels. Training rejects it rather than
silently producing a misleading checkpoint.

## Run locally

Python 3.11+ is recommended.

```bash
python3 -m venv env
source env/bin/activate
python -m pip install -r requirements.lock
export TOMTOM_API_KEY='your-free-tomtom-key'
uvicorn src.api.app:app --reload
```

Open `http://127.0.0.1:8000`; API documentation is available at
`http://127.0.0.1:8000/docs`. Create a free key at the TomTom Developer Portal
and keep it outside source control. With `TOMTOM_API_KEY`, `/api/predict` uses
TomTom geocoding, traffic-aware routing, and Traffic Flow Segment Data. Without
it the API reports `traffic_available: false` and uses the OSRM fallback.

Run local checks:

```bash
python test_inference.py
python -m unittest discover -s tests -v
```

## API

`POST /api/predict`

```json
{
  "origin": "Delhi, India",
  "destination": "Chandigarh, India",
  "speed_multiplier": 1.0,
  "harsh_braking": 1.0,
  "aggressive_acceleration": 1.0
}
```

The result includes `eta_minutes.p10`, `eta_minutes.p50`, and
`eta_minutes.p90`, plus the real-time traffic baseline, free-flow duration,
traffic delay, a route-origin traffic-flow observation, and the provider used.
`GET /health` reports whether TomTom live traffic or the OSRM static fallback
is active.

## Collect real mountain-trip outcomes

Use the trip-event endpoints from the first ETA request through completion.
They store the exact model feature snapshot and routing baseline used at
dispatch, so later training compares like with like.

```bash
# Start a trip and save the returned trip_id.
curl -X POST http://127.0.0.1:8000/api/trips \
  -H 'Content-Type: application/json' \
  -d '{"origin":"Manali, India","destination":"Leh, India","vehicle_type":"4x4"}'

# Send GPS pings from a phone/tracker during the trip.
curl -X POST http://127.0.0.1:8000/api/trips/TRIP_ID/positions \
  -H 'Content-Type: application/json' \
  -d '{"occurred_at":"2026-09-29T09:30:00+00:00","latitude":32.24,"longitude":77.19,"speed_kph":34,"heading_degrees":12,"source":"traccar"}'

# Record the actual duration after arrival.
curl -X POST http://127.0.0.1:8000/api/trips/TRIP_ID/complete \
  -H 'Content-Type: application/json' \
  -d '{"completed_at":"2026-09-29T17:45:00+00:00","actual_duration_minutes":495,"source":"driver-app"}'
```

Before completion, request a current **remaining** quantile ETA from the most
recent GPS ping:

```bash
curl -X POST http://127.0.0.1:8000/api/trips/TRIP_ID/eta
```

For free GPS collection, self-host Traccar and use its mobile client or a
compatible tracker. Export completed records and train a replacement model:

```bash
python -m src.training.export_events --output data/training/mountain_routes.jsonl
python -m src.training.train --data data/training/mountain_routes.jsonl --output models/mountain_eta.pth
ETA_MODEL_PATH=models/mountain_eta.pth ETA_MODEL_CALIBRATED=true uvicorn src.api.app:app
```

With the current split fractions, the trainer requires at least 144 completed
trips: 100 older trips for fitting and 20+ newer trips each for validation and
final testing. In practice, collect far more. It sorts by `started_at`, selects
the best epoch only on the validation period, and does not inspect the newest
test period until selection is complete. It writes
`models/mountain_eta.metrics.json`; live learned quantiles remain disabled if
that report fails p50-baseline, MAE, p10, or p90 coverage gates.

## Training-data contract

The trainer consumes a JSONL file. Every row needs a unique `trip_id`, a
timezone-aware `started_at` value from when its feature snapshot was taken, a
non-empty sequence of H3 cells, a matching N×6 feature matrix in this exact
order, a driver profile, and two positive durations in minutes:

1. `elevation_meters`
2. `incline_gradient_pct`
3. `temperature_celsius`
4. `relative_humidity_pct`
5. `wind_speed_kmh`
6. `precipitation_mm`

```json
{"trip_id":"trip-123","started_at":"2026-09-29T09:00:00+00:00","h3_cells":["876...","876..."],"continuous_features":[[210,0.2,29,65,8,0],[220,0.4,29,64,9,0]],"driver_profile":[1.0,0.2,0.3],"base_duration_minutes":120,"actual_duration_minutes":135}
```

Train explicitly to a new destination; this command never overwrites the
bundled checkpoint unless that exact output path is supplied:

```bash
python -m src.training.train --data data/routes.jsonl --output models/eta.pth
```

Serve a newly trained checkpoint by setting `ETA_MODEL_PATH` and explicitly
requesting calibrated mode, for example
`ETA_MODEL_PATH=models/eta.pth ETA_MODEL_CALIBRATED=true uvicorn src.api.app:app`.
The API only enables learned quantiles if the adjacent `.metrics.json` report
is eligible and checksum-matches that exact checkpoint; otherwise it keeps the
traffic baseline anchor. Do not weaken the promotion limits without reviewing
representative, held-out mountain-trip performance.

The project config in `config/project_config.yaml` is loaded by the API and
trainer. Tests cover config/model compatibility, ordered quantiles, variable
sequence lengths, API startup, and rejection of non-model-ready CSV input.
