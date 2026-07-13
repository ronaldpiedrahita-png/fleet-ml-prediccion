"""Tests del generador de datos sinteticos."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_dataset import FEATURES, TARGET, simulate_fleet  # noqa: E402
from ml_features import LEAKING_FEATURES  # noqa: E402


def test_schema_and_size():
    df = simulate_fleet(n_trucks=200, seed=42)
    assert list(df.columns) == FEATURES + [TARGET, "truck_id"]
    assert len(df) == 200


def test_is_deterministic_given_seed():
    df1 = simulate_fleet(seed=42)
    df2 = simulate_fleet(seed=42)
    assert df1.equals(df2)


def test_generated_data_has_no_leaking_columns():
    df = simulate_fleet(seed=7)
    assert not (set(df.columns) & LEAKING_FEATURES)


def test_failure_rate_is_plausible():
    rate = simulate_fleet(seed=42)[TARGET].mean()
    assert 0.10 < rate < 0.45, f"Tasa de fallos inverosimil: {rate:.2%}"
