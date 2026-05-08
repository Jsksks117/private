from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from dp_evaluator import DPSyntheticEvaluator


def _pick_target_table_and_col(real_tables: Dict[str, pd.DataFrame]) -> Tuple[str, str]:
    preferred = ["is_delayed", "fraud_flag", "is_priority", "oversize_flag", "fare_class", "loyalty_tier"]
    for table_name, df in real_tables.items():
        for col in preferred:
            if col in df.columns:
                unique_count = df[col].dropna().nunique()
                if 2 <= unique_count <= 10:
                    return table_name, col

    for table_name, df in real_tables.items():
        for col in df.columns:
            if col == "id" or col.endswith("_id"):
                continue
            unique_count = df[col].dropna().nunique()
            if 2 <= unique_count <= 10:
                return table_name, col

    first_table = next(iter(real_tables.keys()))
    fallback_col = next(col for col in real_tables[first_table].columns if col != "id")
    return first_table, fallback_col


def evaluate_synthetic_quality(
    real_tables: Dict[str, pd.DataFrame],
    synth_tables: Dict[str, pd.DataFrame],
    relationships: List[Tuple[str, str, str, str]],
) -> Dict[str, object]:
    evaluator = DPSyntheticEvaluator(real_tables=real_tables, synth_tables=synth_tables)

    marginal = evaluator.evaluate_marginal_fidelity()
    relational = evaluator.evaluate_relational_integrity(relationships)

    target_table, target_col = _pick_target_table_and_col(real_tables)

    tstr_result = {}
    tstr_error = None
    privacy_result = {}
    try:
        tstr_result = evaluator.evaluate_tstr(target_table=target_table, target_col=target_col)
    except Exception as ex:
        tstr_error = str(ex)

    try:
        privacy_result = evaluator.evaluate_privacy(target_table=target_table)
    except Exception:
        pass

    fk_keys = [k for k in relational if k.endswith("FK_violation_rate")]
    avg_fk_violation = float(np.mean([relational[k] for k in fk_keys])) if fk_keys else 0.0

    cardinality_keys = [k for k in relational if k.endswith("cardinality_ratio")]
    cardinality_errors = [abs(1.0 - float(relational[k])) for k in cardinality_keys]
    avg_cardinality_error = float(np.mean(cardinality_errors)) if cardinality_errors else 0.0

    ks_values: List[float] = []
    for table_metrics in marginal.values():
        for metric_name, metric_value in table_metrics.items():
            if metric_name.startswith("KS_"):
                ks_values.append(float(metric_value))
    avg_ks_score = float(np.mean(ks_values)) if ks_values else 0.0

    return {
        "marginal": marginal,
        "relational": relational,
        "tstr": tstr_result,
        "privacy": privacy_result,
        "target_table": target_table,
        "target_col": target_col,
        "tstr_error": tstr_error,
        "summary": {
            "average_fk_violation_rate": avg_fk_violation,
            "average_cardinality_error": avg_cardinality_error,
            "average_ks_score": avg_ks_score,
        },
    }
