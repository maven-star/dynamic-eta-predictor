"""Train and evaluate an ETA residual-multiplier model from trip JSONL.

The input is deliberately strict. Every example must have a unique trip ID and
the timestamp at which route features were observed. Splits are chronological:
older trips train the model, newer trips select it, and the newest trips are a
final test. A checkpoint is only eligible for live serving when its adjacent
metrics report passes the promotion gates in ``config/project_config.yaml``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from src.config import ProjectConfig, load_project_config
from src.models.deepreta_system import DeeprETAEndToEndSystem

QUANTILES = (0.10, 0.50, 0.90)
REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrainingExample:
    trip_id: str
    started_at: datetime
    h3_cells: list[str]
    features: torch.Tensor
    driver: torch.Tensor
    base_duration_minutes: float
    actual_duration_minutes: float

    @property
    def target_multiplier(self) -> float:
        return self.actual_duration_minutes / self.base_duration_minutes


@dataclass(frozen=True)
class EvaluationMetrics:
    count: int
    pinball_loss: float
    p10_coverage: float
    p50_coverage: float
    p90_coverage: float
    p50_mae_minutes: float
    baseline_mae_minutes: float
    p50_skill_vs_baseline: float
    quantile_crossing_count: int


def _parse_timestamp(value: object, line_number: int) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"line {line_number}: started_at must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"line {line_number}: started_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"line {line_number}: started_at must include a timezone")
    if parsed > datetime.now(timezone.utc):
        raise ValueError(f"line {line_number}: started_at cannot be in the future")
    return parsed.astimezone(timezone.utc)


def _example_from_record(record: object, line_number: int) -> TrainingExample:
    if not isinstance(record, dict):
        raise ValueError(f"line {line_number}: record must be a JSON object")
    required = {
        "trip_id", "started_at", "h3_cells", "continuous_features", "driver_profile",
        "base_duration_minutes", "actual_duration_minutes",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"line {line_number}: missing required fields: {', '.join(sorted(missing))}")
    trip_id = record["trip_id"]
    if not isinstance(trip_id, str) or not trip_id.strip():
        raise ValueError(f"line {line_number}: trip_id must be a non-empty string")
    started_at = _parse_timestamp(record["started_at"], line_number)
    h3_cells = record["h3_cells"]
    if not isinstance(h3_cells, list) or not h3_cells or not all(isinstance(cell, str) and cell for cell in h3_cells):
        raise ValueError(f"line {line_number}: h3_cells must be a non-empty list of strings")
    try:
        features = torch.tensor(record["continuous_features"], dtype=torch.float32)
        driver = torch.tensor(record["driver_profile"], dtype=torch.float32)
        base, actual = float(record["base_duration_minutes"]), float(record["actual_duration_minutes"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"line {line_number}: numeric fields must contain finite numbers") from error
    if features.ndim != 2 or features.shape != (len(h3_cells), 6):
        raise ValueError(f"line {line_number}: continuous_features must have shape [{len(h3_cells)}, 6]")
    if driver.shape != (3,):
        raise ValueError(f"line {line_number}: driver_profile must contain exactly three values")
    if not torch.isfinite(features).all() or not torch.isfinite(driver).all() or not math.isfinite(base) or not math.isfinite(actual):
        raise ValueError(f"line {line_number}: numeric fields must contain finite numbers")
    if not (0 < base <= 10_080 and 0 < actual <= 10_080):
        raise ValueError(f"line {line_number}: durations must be between 0 and 10,080 minutes")
    return TrainingExample(trip_id.strip(), started_at, h3_cells, features, driver, base, actual)


def load_examples(path: Path, minimum_examples: int = 1) -> list[TrainingExample]:
    if path.suffix.lower() != ".jsonl":
        raise ValueError("training input must be JSONL route sequences; raw delivery CSVs are not model-ready")
    examples: list[TrainingExample] = []
    trip_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON") from error
            example = _example_from_record(record, line_number)
            if example.trip_id in trip_ids:
                raise ValueError(f"line {line_number}: duplicate trip_id '{example.trip_id}'")
            trip_ids.add(example.trip_id)
            examples.append(example)
    if len(examples) < minimum_examples:
        raise ValueError(f"at least {minimum_examples} validated route examples are required for training")
    return examples


def split_chronologically(examples: Sequence[TrainingExample], config: ProjectConfig) -> tuple[list[TrainingExample], list[TrainingExample], list[TrainingExample]]:
    """Return non-overlapping old/train, newer/validation, newest/test partitions."""
    required = config.min_train_examples + 2 * config.min_evaluation_examples
    if len(examples) < required:
        raise ValueError(
            f"need at least {required} completed trips for chronological training, validation, and test partitions"
        )
    ordered = sorted(examples, key=lambda example: (example.started_at, example.trip_id))
    validation_count = max(config.min_evaluation_examples, math.ceil(len(ordered) * config.validation_fraction))
    test_count = max(config.min_evaluation_examples, math.ceil(len(ordered) * config.test_fraction))
    validation_start = len(ordered) - validation_count - test_count
    test_start = len(ordered) - test_count
    # Never split records sharing a feature-observation timestamp.
    while validation_start > 0 and ordered[validation_start - 1].started_at == ordered[validation_start].started_at:
        validation_start -= 1
    while test_start > validation_start and ordered[test_start - 1].started_at == ordered[test_start].started_at:
        test_start -= 1
    train_examples = ordered[:validation_start]
    validation_examples = ordered[validation_start:test_start]
    test_examples = ordered[test_start:]
    if len(train_examples) < config.min_train_examples:
        raise ValueError("chronological split leaves too few older trips for training")
    if len(validation_examples) < config.min_evaluation_examples or len(test_examples) < config.min_evaluation_examples:
        raise ValueError("chronological timestamp groups leave too few trips for validation or test")
    return train_examples, validation_examples, test_examples


class PinballLoss(nn.Module):
    def __init__(self, quantiles: tuple[float, float, float] = QUANTILES) -> None:
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = target - prediction
        return torch.maximum((self.quantiles - 1) * error, self.quantiles * error).mean()


def evaluate(
    model: DeeprETAEndToEndSystem,
    examples: Iterable[TrainingExample],
    loss_fn: PinballLoss,
    device: torch.device,
) -> EvaluationMetrics:
    losses: list[float] = []
    coverage = [0, 0, 0]
    p50_absolute_errors: list[float] = []
    baseline_absolute_errors: list[float] = []
    crossings = 0
    count = 0
    model.eval()
    with torch.inference_mode():
        for example in examples:
            prediction = model.predict_eta(example.h3_cells, example.features, example.driver)
            target = torch.tensor(example.target_multiplier, device=device)
            losses.append(float(loss_fn(prediction, target).item()))
            values = prediction.detach().cpu().tolist()
            crossings += int(not (values[0] < values[1] < values[2]))
            for index, value in enumerate(values):
                coverage[index] += int(example.target_multiplier <= value)
            predicted_p50_minutes = values[1] * example.base_duration_minutes
            p50_absolute_errors.append(abs(predicted_p50_minutes - example.actual_duration_minutes))
            baseline_absolute_errors.append(abs(example.base_duration_minutes - example.actual_duration_minutes))
            count += 1
    if not count:
        raise ValueError("cannot evaluate an empty split")
    p50_mae = sum(p50_absolute_errors) / count
    baseline_mae = sum(baseline_absolute_errors) / count
    skill = (baseline_mae - p50_mae) / baseline_mae if baseline_mae > 0 else 0.0
    return EvaluationMetrics(
        count=count,
        pinball_loss=sum(losses) / count,
        p10_coverage=coverage[0] / count,
        p50_coverage=coverage[1] / count,
        p90_coverage=coverage[2] / count,
        p50_mae_minutes=p50_mae,
        baseline_mae_minutes=baseline_mae,
        p50_skill_vs_baseline=skill,
        quantile_crossing_count=crossings,
    )


def promotion_decision(metrics: EvaluationMetrics, config: ProjectConfig) -> tuple[bool, list[str]]:
    """Return whether a candidate is safe to opt into live model serving."""
    failures: list[str] = []
    if metrics.count < config.min_evaluation_examples:
        failures.append("test split is smaller than the required evaluation sample")
    if metrics.quantile_crossing_count:
        failures.append("quantile crossings were observed")
    if metrics.p50_mae_minutes > config.max_test_p50_mae_minutes:
        failures.append("p50 MAE exceeds the configured production limit")
    if metrics.p50_skill_vs_baseline < config.minimum_p50_skill_vs_baseline:
        failures.append("p50 does not improve on the traffic-routing baseline")
    if not config.p10_coverage_min <= metrics.p10_coverage <= config.p10_coverage_max:
        failures.append("p10 coverage is outside the configured calibration band")
    if not config.p90_coverage_min <= metrics.p90_coverage <= config.p90_coverage_max:
        failures.append("p90 coverage is outside the configured calibration band")
    return not failures, failures


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for block in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomically(path: Path, document: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def train(data_path: Path, output_path: Path, seed: int = 42, report_path: Path | None = None) -> dict[str, object]:
    config = load_project_config()
    required_examples = config.min_train_examples + 2 * config.min_evaluation_examples
    examples = load_examples(data_path, required_examples)
    train_examples, validation_examples, test_examples = split_chronologically(examples, config)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeeprETAEndToEndSystem(
        vocab_size=config.h3_vocab_size,
        h3_dim=config.h3_embedding_dim,
        continuous_dim=config.continuous_embedding_dim,
        max_seq_len=config.sequence_length,
        embed_dim=config.embedding_dim,
        linformer_k=config.linformer_projection_k,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = PinballLoss().to(device)
    rng = random.Random(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    for epoch in range(1, config.epochs + 1):
        model.train()
        shuffled_examples = list(train_examples)
        rng.shuffle(shuffled_examples)
        losses = []
        for start in range(0, len(shuffled_examples), config.batch_size):
            batch = shuffled_examples[start : start + config.batch_size]
            optimizer.zero_grad()
            batch_loss = torch.stack([
                loss_fn(
                    model.predict_eta(example.h3_cells, example.features, example.driver),
                    torch.tensor(example.target_multiplier, device=device),
                )
                for example in batch
            ]).mean()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(batch_loss.item()))
        validation_metrics = evaluate(model, validation_examples, loss_fn, device)
        if validation_metrics.pinball_loss < best_validation_loss:
            best_validation_loss = validation_metrics.pinball_loss
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"epoch={epoch:03d} train_pinball={sum(losses) / len(losses):.5f} "
            f"validation_pinball={validation_metrics.pinball_loss:.5f}"
        )
    if best_state is None:
        raise RuntimeError("training did not produce a candidate checkpoint")
    model.load_state_dict(best_state)
    validation_metrics = evaluate(model, validation_examples, loss_fn, device)
    test_metrics = evaluate(model, test_examples, loss_fn, device)
    eligible, failures = promotion_decision(test_metrics, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(model.cpu().state_dict(), temporary)
    temporary.replace(output_path)
    target_report_path = report_path or output_path.with_suffix(".metrics.json")
    target_report_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"path": str(output_path), "sha256": _sha256(output_path)},
        "data": {
            "total_examples": len(examples), "train_examples": len(train_examples),
            "validation_examples": len(validation_examples), "test_examples": len(test_examples),
            "train_end": train_examples[-1].started_at.isoformat(),
            "validation_start": validation_examples[0].started_at.isoformat(),
            "test_start": test_examples[0].started_at.isoformat(),
        },
        "validation_metrics": asdict(validation_metrics),
        "test_metrics": asdict(test_metrics),
        "promotion": {"eligible": eligible, "failures": failures},
    }
    _write_json_atomically(target_report_path, report)
    print(f"saved candidate checkpoint to {output_path}")
    print(f"saved evaluation report to {target_report_path}")
    print("promotion_eligible=" + str(eligible).lower())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Validated completed-trip JSONL input")
    parser.add_argument("--output", type=Path, required=True, help="Candidate checkpoint destination")
    parser.add_argument("--report", type=Path, help="Metrics report destination (default: OUTPUT.metrics.json)")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed")
    args = parser.parse_args()
    try:
        train(args.data, args.output, args.seed, args.report)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
