"""Live routing and traffic providers.

TomTom is used when ``TOMTOM_API_KEY`` is configured. It supplies a
traffic-aware route duration and current flow speed; OSRM does not supply live
traffic and is therefore only a fallback for local development.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests


class ProviderError(RuntimeError):
    """A provider could not return a valid response."""


@dataclass(frozen=True)
class Place:
    latitude: float
    longitude: float
    label: str


@dataclass(frozen=True)
class LiveRoute:
    distance_km: float
    travel_time_minutes: float
    free_flow_minutes: float | None
    traffic_delay_minutes: float | None
    coordinates: list[list[float]]
    provider: str


@dataclass(frozen=True)
class FlowObservation:
    current_speed_kph: float
    free_flow_speed_kph: float
    confidence: float | None


class TomTomRealtimeProvider:
    """Thin, testable adapter for TomTom Search, Routing, and Traffic Flow."""

    base_url = "https://api.tomtom.com"

    def __init__(self, api_key: str, session: requests.Session | None = None, timeout_seconds: int = 10) -> None:
        if not api_key.strip():
            raise ValueError("TomTom API key is empty")
        self.api_key = api_key
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.session.get(
                f"{self.base_url}{path}", params={**params, "key": self.api_key}, timeout=self.timeout_seconds
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            raise ProviderError("TomTom service is unavailable") from error
        if not isinstance(payload, dict):
            raise ProviderError("TomTom returned an unexpected response")
        return payload

    def geocode(self, query: str) -> Place:
        payload = self._get(f"/search/2/geocode/{quote(query.strip())}.json", {"limit": 1})
        results = payload.get("results", [])
        if not results:
            raise ProviderError(f"TomTom could not geocode '{query}'")
        result = results[0]
        position = result.get("position", {})
        try:
            return Place(float(position["lat"]), float(position["lon"]), str(result["address"]["freeformAddress"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ProviderError("TomTom returned an incomplete geocoding result") from error

    def search_places(self, query: str, limit: int = 6) -> list[Place]:
        """Return distinct, user-selectable places for an origin/destination field."""
        normalized_query = query.strip()
        if len(normalized_query) < 2:
            return []
        payload = self._get(
            f"/search/2/search/{quote(normalized_query)}.json",
            {"limit": max(1, min(limit, 10)), "typeahead": "true"},
        )
        places: list[Place] = []
        seen: set[tuple[float, float, str]] = set()
        for result in payload.get("results", []):
            position = result.get("position", {})
            address = result.get("address", {})
            poi = result.get("poi", {})
            try:
                latitude, longitude = float(position["lat"]), float(position["lon"])
                address_label = str(address.get("freeformAddress") or "").strip()
                poi_name = str(poi.get("name") or "").strip()
            except (KeyError, TypeError, ValueError):
                continue
            label = ", ".join(part for part in (poi_name, address_label) if part) or f"{latitude:.5f}, {longitude:.5f}"
            key = (round(latitude, 6), round(longitude, 6), label.casefold())
            if key not in seen:
                seen.add(key)
                places.append(Place(latitude, longitude, label))
        return places

    def calculate_route(self, origin: Place, destination: Place) -> LiveRoute:
        locations = f"{origin.latitude},{origin.longitude}:{destination.latitude},{destination.longitude}"
        payload = self._get(
            f"/routing/1/calculateRoute/{locations}/json",
            {"traffic": "true", "routeType": "fastest", "travelMode": "car"},
        )
        routes = payload.get("routes", [])
        if not routes:
            raise ProviderError("TomTom could not find a drivable route")
        route = routes[0]
        summary = route.get("summary", {})
        points = [
            [float(point["latitude"]), float(point["longitude"])]
            for leg in route.get("legs", [])
            for point in leg.get("points", [])
        ]
        if len(points) < 2:
            raise ProviderError("TomTom returned a route without geometry")
        try:
            travel_seconds = float(summary["travelTimeInSeconds"])
            free_flow_seconds = summary.get("noTrafficTravelTimeInSeconds")
            delay_seconds = summary.get("trafficDelayInSeconds")
            return LiveRoute(
                distance_km=float(summary["lengthInMeters"]) / 1000.0,
                travel_time_minutes=travel_seconds / 60.0,
                free_flow_minutes=float(free_flow_seconds) / 60.0 if free_flow_seconds is not None else None,
                traffic_delay_minutes=float(delay_seconds) / 60.0 if delay_seconds is not None else None,
                coordinates=points,
                provider="tomtom_live_traffic",
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ProviderError("TomTom returned an incomplete routing result") from error

    def flow_at(self, latitude: float, longitude: float) -> FlowObservation:
        payload = self._get(
            "/traffic/services/4/flowSegmentData/absolute/10/json",
            {"point": f"{latitude},{longitude}", "unit": "KMPH"},
        )
        flow = payload.get("flowSegmentData", {})
        try:
            return FlowObservation(
                current_speed_kph=float(flow["currentSpeed"]),
                free_flow_speed_kph=float(flow["freeFlowSpeed"]),
                confidence=float(flow["confidence"]) if flow.get("confidence") is not None else None,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ProviderError("TomTom returned incomplete traffic-flow data") from error
