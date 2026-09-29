"""Export completed trip events into the JSONL contract consumed by train.py."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_collection.trip_store import TripStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/eta_events.sqlite3"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = TripStore(args.database).export_completed_training_jsonl(args.output)
    print(f"exported {count} completed trips to {args.output}")


if __name__ == "__main__":
    main()
