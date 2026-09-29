"""Durable local trip-event store for ETA training data collection.

This is deliberately small and dependency-free for a project deployment. The
same event contract can later be moved to Postgres/Kafka without changing the
API or feature schema.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TripStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS trips (
                    trip_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL CHECK (status IN ('active', 'completed')),
                    request_json TEXT NOT NULL,
                    prediction_json TEXT NOT NULL,
                    feature_snapshot_json TEXT NOT NULL,
                    actual_duration_minutes REAL,
                    completion_source TEXT
                );
                CREATE TABLE IF NOT EXISTS position_events (
                    trip_id TEXT NOT NULL REFERENCES trips(trip_id),
                    occurred_at TEXT NOT NULL,
                    latitude REAL NOT NULL CHECK (latitude BETWEEN -90 AND 90),
                    longitude REAL NOT NULL CHECK (longitude BETWEEN -180 AND 180),
                    speed_kph REAL,
                    heading_degrees REAL,
                    source TEXT NOT NULL,
                    PRIMARY KEY (trip_id, occurred_at)
                );
                CREATE INDEX IF NOT EXISTS idx_position_events_trip_time
                    ON position_events(trip_id, occurred_at);
                """
            )

    def create_trip(
        self,
        trip_id: str,
        request: dict[str, Any],
        prediction: dict[str, Any],
        feature_snapshot: dict[str, Any],
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO trips (trip_id, created_at, status, request_json, prediction_json, feature_snapshot_json)
                   VALUES (?, ?, 'active', ?, ?, ?)""",
                (trip_id, utc_now(), json.dumps(request), json.dumps(prediction), json.dumps(feature_snapshot)),
            )

    def add_position(
        self,
        trip_id: str,
        occurred_at: str,
        latitude: float,
        longitude: float,
        speed_kph: float | None,
        heading_degrees: float | None,
        source: str,
    ) -> None:
        with self._connection() as connection:
            active = connection.execute("SELECT 1 FROM trips WHERE trip_id = ? AND status = 'active'", (trip_id,)).fetchone()
            if active is None:
                raise KeyError("active trip was not found")
            connection.execute(
                """INSERT OR IGNORE INTO position_events
                   (trip_id, occurred_at, latitude, longitude, speed_kph, heading_degrees, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (trip_id, occurred_at, latitude, longitude, speed_kph, heading_degrees, source),
            )

    def complete_trip(self, trip_id: str, completed_at: str, actual_duration_minutes: float, source: str) -> None:
        with self._connection() as connection:
            updated = connection.execute(
                """UPDATE trips
                   SET status = 'completed', completed_at = ?, actual_duration_minutes = ?, completion_source = ?
                   WHERE trip_id = ? AND status = 'active'""",
                (completed_at, actual_duration_minutes, source, trip_id),
            )
            if updated.rowcount != 1:
                raise KeyError("active trip was not found")

    def get_trip(self, trip_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM trips WHERE trip_id = ?", (trip_id,)).fetchone()
            if row is None:
                return None
            positions = connection.execute(
                "SELECT occurred_at, latitude, longitude, speed_kph, heading_degrees, source FROM position_events WHERE trip_id = ? ORDER BY occurred_at",
                (trip_id,),
            ).fetchall()
        return {
            "trip_id": row["trip_id"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
            "status": row["status"],
            "request": json.loads(row["request_json"]),
            "prediction": json.loads(row["prediction_json"]),
            "feature_snapshot": json.loads(row["feature_snapshot_json"]),
            "actual_duration_minutes": row["actual_duration_minutes"],
            "positions": [dict(position) for position in positions],
        }

    def export_completed_training_jsonl(self, output_path: Path) -> int:
        """Export only completed, fully-featured records for model training."""
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT trip_id, created_at, completed_at, feature_snapshot_json, actual_duration_minutes
                   FROM trips
                   WHERE status = 'completed' AND actual_duration_minutes IS NOT NULL
                   ORDER BY created_at ASC, trip_id ASC"""
            ).fetchall()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        exported = 0
        with output_path.open("w", encoding="utf-8") as output_file:
            for row in rows:
                snapshot = json.loads(row["feature_snapshot_json"])
                required = {"h3_cells", "continuous_features", "driver_profile", "routing_eta_minutes"}
                if not required.issubset(snapshot) or float(snapshot["routing_eta_minutes"]) <= 0:
                    continue
                record = {
                    "trip_id": row["trip_id"],
                    # This is when the feature snapshot and baseline ETA were
                    # available. It is the only valid timestamp for a leakage-
                    # resistant chronological model split.
                    "started_at": row["created_at"],
                    "completed_at": row["completed_at"],
                    "h3_cells": snapshot["h3_cells"],
                    "continuous_features": snapshot["continuous_features"],
                    "driver_profile": snapshot["driver_profile"],
                    "base_duration_minutes": snapshot["routing_eta_minutes"],
                    "actual_duration_minutes": row["actual_duration_minutes"],
                    "vehicle_type": snapshot.get("vehicle_type", "unknown"),
                    "request_type": snapshot.get("request_type", "mountain_trip"),
                }
                output_file.write(json.dumps(record) + "\n")
                exported += 1
        return exported
