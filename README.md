# Dynamic ETA Predictor: Adapting DeeprETA for Long-Distance Routes

A deep learning pipeline and FastAPI application that adapts **Uber's DeeprETA** framework to predict quantile arrival time windows ($p_{10}, p_{50}, p_{90}$) for long-distance transit corridors across India (e.g., Delhi to Chandigarh, Leh routes).

The system integrates real-world road network geometry from **Open Source Routing Machine (OSRM)** with **OpenStreetMap Nominatim** reverse-geocoding to break down routes into major transit waypoints while evaluating spatial embeddings and continuous feature projections through an $O(N)$ **Linformer Transformer Encoder**.

---

## 🌟 Key Features

* **Spatial-Temporal Hexagonal Indexing (H3):** Converts spatial coordinates into discrete hexagonal cell identifiers to preserve geographical patterns and highway features.
* **Efficient $O(N)$ Attention with Linformer:** Uses low-rank projections on Key and Value matrices ($E$ and $F$) to execute linear-time self-attention across long-distance sequence tokens.
* **Quantile Loss Regression ($p_{10}, p_{50}, p_{90}$):** Predicts optimistic, baseline, and pessimistic travel time estimates using Pinball Loss to model travel variance.
* **OSRM Road Network Routing:** Extracts exact road distance and polyline geometry rather than relying on straight-line Euclidean metrics.
* **Corridor Breakdown & Intermediate Waypoints:** Samples route coordinates and reverse-geocodes intermediate cities/towns along major highways (e.g., `Delhi ➔ Sonipat ➔ Panipat ➔ Karnal ➔ Ambala ➔ Chandigarh`).
* **Interactive Leaflet Web Interface:** Built-in FastAPI web app rendering route polylines with dark/light mode tiles matching system preferences.

---

## 🏗️ Architecture Overview

```text
[User Request: Origin & Destination]
                 │
                 ▼
     1. Geocoding (Nominatim) ──► Lat/Lon Coordinates
                 │
                 ▼
     2. Route Engine (OSRM)   ──► Road Polyline, Real Distance & Waypoints
                 │
                 ▼
     3. Spatial Indexing      ──► Spatial Hexagon Cell Vectors
                 │
                 ▼
     4. Tensor Embedding      ──► H3 Embeddings + Continuous Feature Projection
                 │
                 ▼
     5. Linformer Encoder     ──► O(N) Contextualized Multi-Head Self-Attention
                 │
                 ▼
     6. Quantile Head (MLP)   ──► [p10, p50, p90] Travel Time Multipliers
                 │
                 ▼
     7. Post-Processing       ──► Calculated ETAs + Interactive Map Rendering