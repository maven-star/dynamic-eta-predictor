"""Typed project configuration loaded from ``config/project_config.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    h3_resolution: int
    sequence_length: int
    h3_vocab_size: int
    h3_embedding_dim: int
    continuous_embedding_dim: int
    linformer_projection_k: int
    embedding_dim: int
    batch_size: int
    learning_rate: float
    epochs: int
    validation_fraction: float
    test_fraction: float
    min_train_examples: int
    min_evaluation_examples: int
    max_test_p50_mae_minutes: float
    minimum_p50_skill_vs_baseline: float
    p10_coverage_min: float
    p10_coverage_max: float
    p90_coverage_min: float
    p90_coverage_max: float


def load_project_config(path: Path | None = None) -> ProjectConfig:
    config_path = path or Path(__file__).resolve().parents[1] / "config/project_config.yaml"
    with config_path.open("r", encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    features, model, training, promotion = raw["features"], raw["model"], raw["training"], raw["promotion"]
    return ProjectConfig(
        h3_resolution=int(raw["spatial"]["h3_resolution"]),
        sequence_length=int(features["sequence_length"]),
        h3_vocab_size=int(model["h3_vocab_size"]),
        h3_embedding_dim=int(model["h3_embedding_dim"]),
        continuous_embedding_dim=int(model["continuous_embedding_dim"]),
        linformer_projection_k=int(model["linformer_projection_k"]),
        embedding_dim=int(model["embedding_dim"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        epochs=int(training["epochs"]),
        validation_fraction=float(training["validation_fraction"]),
        test_fraction=float(training["test_fraction"]),
        min_train_examples=int(training["min_train_examples"]),
        min_evaluation_examples=int(training["min_evaluation_examples"]),
        max_test_p50_mae_minutes=float(promotion["max_test_p50_mae_minutes"]),
        minimum_p50_skill_vs_baseline=float(promotion["minimum_p50_skill_vs_baseline"]),
        p10_coverage_min=float(promotion["p10_coverage_min"]),
        p10_coverage_max=float(promotion["p10_coverage_max"]),
        p90_coverage_min=float(promotion["p90_coverage_min"]),
        p90_coverage_max=float(promotion["p90_coverage_max"]),
    )
