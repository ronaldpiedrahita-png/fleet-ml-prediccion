"""Fuente unica de verdad para la seleccion de features del modelo.

Centralizar esto evita que un feature con fuga se cuele en el entrenamiento
sin que lo note ningun archivo. El test tests/test_no_target_leakage.py vigila
que ningun feature seleccionado prediga el target casi perfecto por si solo.
"""
from __future__ import annotations

from collections.abc import Iterable

# Columnas que no son features: identificador y etiqueta.
NON_FEATURES = {"will_fail", "truck_id"}

# Columnas derivadas de la propia etiqueta de fallo -> fuga de datos directa.
# `total_fallos` = conteo de fallos; `ratio_fallos` = fallos/eventos;
# `total_downtime_days` solo acumula en fallos. Cualquiera > 0 equivale a
# will_fail = 1. Jamas deben usarse como feature.
LEAKING_FEATURES = {"total_fallos", "ratio_fallos", "total_downtime_days"}

EXCLUDED = NON_FEATURES | LEAKING_FEATURES


def select_features(columns: Iterable[str]) -> list[str]:
    """Devuelve las columnas utilizables como features (sin target ni fugas)."""
    return [c for c in columns if c not in EXCLUDED]
