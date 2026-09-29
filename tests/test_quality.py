import importlib
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

from src.api.app import app, format_minutes, health, home
from src.config import load_project_config
from src.data_collection.realtime_provider import TomTomRealtimeProvider
from src.data_collection.trip_store import TripStore
from src.models.deepreta_system import DeeprETAEndToEndSystem
from src.training.train import EvaluationMetrics, TrainingExample, load_examples, promotion_decision, split_chronologically


class ModelQualityTests(unittest.TestCase):
    def test_outputs_are_ordered_for_variable_sequence_lengths(self):
        model = DeeprETAEndToEndSystem().eval()
        for length in (1, 7, 8, 19):
            with torch.inference_mode():
                prediction = model.predict_eta(["cell"] * length, torch.zeros(length, 6), torch.ones(3))
            self.assertEqual(tuple(prediction.shape), (3,))
            self.assertTrue(bool(prediction[0] < prediction[1] < prediction[2]))

    def test_config_matches_model_contract(self):
        config = load_project_config()
        self.assertEqual(config.sequence_length, 8)
        self.assertEqual(config.linformer_projection_k, 4)

    def test_raw_csv_is_rejected_as_training_input(self):
        with self.assertRaisesRegex(ValueError, "JSONL route sequences"):
            load_examples(Path("sim_data/raw_telemetry.csv"))

    def test_chronological_split_keeps_newest_data_out_of_training(self):
        config = load_project_config()
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        examples = [
            TrainingExample(
                trip_id=f"trip-{index}",
                started_at=start + timedelta(hours=index),
                h3_cells=["cell"],
                features=torch.zeros(1, 6),
                driver=torch.ones(3),
                base_duration_minutes=60,
                actual_duration_minutes=65,
            )
            for index in range(150)
        ]
        train_examples, validation_examples, test_examples = split_chronologically(list(reversed(examples)), config)
        self.assertGreaterEqual(len(train_examples), config.min_train_examples)
        self.assertGreaterEqual(len(validation_examples), config.min_evaluation_examples)
        self.assertGreaterEqual(len(test_examples), config.min_evaluation_examples)
        self.assertLess(train_examples[-1].started_at, validation_examples[0].started_at)
        self.assertLess(validation_examples[-1].started_at, test_examples[0].started_at)

    def test_promotion_rejects_uncalibrated_or_worse_candidate(self):
        config = load_project_config()
        metrics = EvaluationMetrics(
            count=config.min_evaluation_examples,
            pinball_loss=0.1,
            p10_coverage=0.5,
            p50_coverage=0.5,
            p90_coverage=0.5,
            p50_mae_minutes=100,
            baseline_mae_minutes=20,
            p50_skill_vs_baseline=-4.0,
            quantile_crossing_count=0,
        )
        eligible, failures = promotion_decision(metrics, config)
        self.assertFalse(eligible)
        self.assertTrue(failures)


class ApiSmokeTests(unittest.TestCase):
    def test_health_and_home_are_available_without_upstream_calls(self):
        self.assertIn(health()["status"], {"ok", "degraded"})
        page = home()
        self.assertIn("Dynamic ETA Predictor", page)
        self.assertIn("Origin / pickup point", page)
        self.assertIn("/api/places", page)
        self.assertTrue(any(route.path == "/api/predict" for route in app.routes))
        self.assertTrue(any(route.path == "/api/places" for route in app.routes))
        self.assertEqual(format_minutes(61), "1h 1m")

    def test_live_features_are_sent_to_the_quantile_model(self):
        module = importlib.import_module("src.api.app")
        original_route, original_builder, original_model, original_tomtom, original_calibrated = (
            module.resolve_route, module.feature_provider.build, module.model, module.tomtom_provider, module.MODEL_CALIBRATED
        )

        class StubModel:
            def predict_eta(self, cells, features, driver):
                self.cells, self.features, self.driver = cells, features, driver
                # Deliberately implausible values from an uncalibrated checkpoint.
                # Serving must keep p50 on the live routing baseline, not expose a
                # two-hour result for a ten-hour live route.
                return torch.tensor([0.1, 0.2, 0.3])

        stub = StubModel()
        try:
            module.resolve_route = lambda origin, destination, origin_coordinates=None, destination_coordinates=None, origin_label=None, destination_label=None: (module.Route(10, 600, [[28.6, 77.2], [28.7, 77.3]], 80, 20, "tomtom_live_traffic"), "Origin", "Destination")
            module.feature_provider.build = lambda route: (["cell-a", "cell-b"], torch.ones(2, 6))
            module.model, module.tomtom_provider = stub, None
            module.MODEL_CALIBRATED = False
            result = module.predict_route(module.RouteRequest(origin="Origin", destination="Destination"))
        finally:
            module.resolve_route, module.feature_provider.build, module.model, module.tomtom_provider, module.MODEL_CALIBRATED = original_route, original_builder, original_model, original_tomtom, original_calibrated
        self.assertEqual(result["eta_minutes"], {"p10": 480, "p50": 600, "p90": 780})
        self.assertEqual(stub.cells, ["cell-a", "cell-b"])
        self.assertEqual(tuple(stub.features.shape), (2, 6))

    def test_selected_place_coordinates_bypass_ambiguous_geocoding(self):
        module = importlib.import_module("src.api.app")
        original_route, original_builder, original_model, original_tomtom, original_calibrated = (
            module.resolve_route, module.feature_provider.build, module.model, module.tomtom_provider, module.MODEL_CALIBRATED
        )
        captured = {}

        class StubModel:
            def predict_eta(self, cells, features, driver):
                return torch.tensor([0.8, 1.0, 1.3])

        def resolve(origin, destination, origin_coordinates=None, destination_coordinates=None, origin_label=None, destination_label=None):
            captured.update({
                "origin_coordinates": origin_coordinates, "destination_coordinates": destination_coordinates,
                "origin_label": origin_label, "destination_label": destination_label,
            })
            return module.Route(10, 60, [[28.63, 77.22], [28.61, 77.21]], 60, 0, "tomtom_live_traffic"), origin_label, destination_label

        try:
            module.resolve_route, module.feature_provider.build, module.model, module.tomtom_provider = resolve, lambda route: (["cell"], torch.ones(1, 6)), StubModel(), None
            module.MODEL_CALIBRATED = False
            result = module.predict_route(module.RouteRequest(
                origin="Delhi", destination="Leh",
                origin_latitude=28.6315, origin_longitude=77.2167, origin_label="Connaught Place, New Delhi",
                destination_latitude=34.1526, destination_longitude=77.5771, destination_label="Leh Main Market",
            ))
        finally:
            module.resolve_route, module.feature_provider.build, module.model, module.tomtom_provider, module.MODEL_CALIBRATED = original_route, original_builder, original_model, original_tomtom, original_calibrated
        self.assertEqual(captured["origin_coordinates"], (28.6315, 77.2167))
        self.assertEqual(captured["destination_coordinates"], (34.1526, 77.5771))
        self.assertEqual(result["origin"], "Connaught Place, New Delhi")

    def test_partial_selected_place_coordinates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "latitude and longitude"):
            from src.api.app import RouteRequest
            RouteRequest(origin="Connaught Place", destination="Leh", origin_latitude=28.63)

    def test_calibrated_mode_requires_a_matching_eligible_report(self):
        module = importlib.import_module("src.api.app")
        original_requested, original_model_path, original_report_path = (
            module.CALIBRATION_REQUESTED, module.MODEL_PATH, module.MODEL_REPORT_PATH
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "candidate.pth"
                report_path = Path(directory) / "candidate.metrics.json"
                checkpoint.write_bytes(b"candidate-checkpoint")
                report_path.write_text(json.dumps({
                    "schema_version": 1,
                    "checkpoint": {"sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
                    "promotion": {"eligible": True},
                }), encoding="utf-8")
                module.CALIBRATION_REQUESTED = True
                module.MODEL_PATH, module.MODEL_REPORT_PATH = checkpoint, report_path
                self.assertTrue(module._load_calibration_status()[0])
                report_path.write_text(json.dumps({
                    "schema_version": 1,
                    "checkpoint": {"sha256": "wrong"},
                    "promotion": {"eligible": True},
                }), encoding="utf-8")
                self.assertFalse(module._load_calibration_status()[0])
        finally:
            module.CALIBRATION_REQUESTED, module.MODEL_PATH, module.MODEL_REPORT_PATH = (
                original_requested, original_model_path, original_report_path
            )


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _FakeSession:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append((url, params, timeout))
        return _FakeResponse(next(self.payloads))


class LiveProviderTests(unittest.TestCase):
    def test_tomtom_adapter_parses_live_route_and_flow(self):
        session = _FakeSession([
            {"results": [{"position": {"lat": 28.6, "lon": 77.2}, "address": {"freeformAddress": "Delhi"}}]},
            {"results": [{"position": {"lat": 30.7, "lon": 76.8}, "address": {"freeformAddress": "Chandigarh"}}]},
            {"routes": [{"summary": {"lengthInMeters": 250000, "travelTimeInSeconds": 14400, "noTrafficTravelTimeInSeconds": 12600, "trafficDelayInSeconds": 1800}, "legs": [{"points": [{"latitude": 28.6, "longitude": 77.2}, {"latitude": 30.7, "longitude": 76.8}]}]}]},
            {"flowSegmentData": {"currentSpeed": 35, "freeFlowSpeed": 60, "confidence": 0.9}},
        ])
        provider = TomTomRealtimeProvider("test-key", session=session)
        route = provider.calculate_route(provider.geocode("Delhi"), provider.geocode("Chandigarh"))
        flow = provider.flow_at(28.6, 77.2)
        self.assertEqual(route.provider, "tomtom_live_traffic")
        self.assertEqual(route.travel_time_minutes, 240)
        self.assertEqual(route.traffic_delay_minutes, 30)
        self.assertEqual(flow.current_speed_kph, 35)
        self.assertIn("key", session.calls[0][1])

    def test_tomtom_search_returns_selectable_places(self):
        session = _FakeSession([{
            "results": [
                {"position": {"lat": 28.6315, "lon": 77.2167}, "poi": {"name": "Connaught Place"}, "address": {"freeformAddress": "New Delhi, Delhi"}},
                {"position": {"lat": 28.6320, "lon": 77.2170}, "address": {"freeformAddress": "Connaught Place, New Delhi"}},
            ]
        }])
        places = TomTomRealtimeProvider("test-key", session=session).search_places("connaught place")
        self.assertEqual(places[0].label, "Connaught Place, New Delhi, Delhi")
        self.assertEqual(places[0].latitude, 28.6315)
        self.assertEqual(session.calls[0][1]["typeahead"], "true")


class TripStoreTests(unittest.TestCase):
    def test_completed_trip_exports_to_training_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "events.sqlite3"
            output = Path(directory) / "routes.jsonl"
            store = TripStore(database)
            snapshot = {
                "h3_cells": ["a", "b"],
                "continuous_features": [[100, 0, 20, 50, 5, 0], [110, 1, 19, 55, 6, 0]],
                "driver_profile": [1, 0.1, 0.2],
                "routing_eta_minutes": 60,
                "vehicle_type": "4x4",
                "request_type": "mountain_trip",
            }
            store.create_trip("trip-1", {"origin": "A"}, {"eta_minutes": {"p50": 60}}, snapshot)
            store.add_position("trip-1", "2026-01-01T00:00:00+00:00", 28.6, 77.2, 20, 90, "test")
            store.complete_trip("trip-1", "2026-01-01T01:15:00+00:00", 75, "test")
            self.assertEqual(store.export_completed_training_jsonl(output), 1)
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["trip_id"], "trip-1")
            self.assertIn("started_at", record)
            self.assertEqual(record["actual_duration_minutes"], 75)
            self.assertEqual(record["base_duration_minutes"], 60)


if __name__ == "__main__":
    unittest.main()
