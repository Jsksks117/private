import time
from typing import Dict

import pandas as pd

from modules.module1_schema_parser import parse_schema
from modules.module2_privacy_allocator import allocate_privacy
from modules.module3_synthesis_engine import DPMultiTableSynthesizer
from modules.module5_evaluation_engine import evaluate_synthetic_quality, _pick_target_table_and_col


def _run_case(real_tables: Dict[str, pd.DataFrame], eps: float, num_rows: int = 120, focus_table: str | None = None, seed: int = 42) -> Dict:
    t0 = time.time()
    schema = parse_schema(real_tables)
    alloc = allocate_privacy(schema["schema_def"], eps)
    synth = DPMultiTableSynthesizer(epsilon=eps, seed=seed, focus_runtime_threshold_seconds=90.0)
    synth_tables = synth.generate(
        real_tables=real_tables,
        schema_def=schema["schema_def"],
        primary_keys=schema["primary_keys"],
        relationships=schema["relationships"],
        num_rows=num_rows,
        epsilon_allocation=alloc["epsilon_allocation"],
        focus_table=focus_table,
        table_row_counts={k: len(v) for k, v in real_tables.items()},
    )
    metrics = evaluate_synthetic_quality(real_tables, synth_tables, schema["relationships"]) 
    return {
        "eps": eps,
        "runtime_sec": round(time.time() - t0, 1),
        "target_table": metrics.get("target_table"),
        "target_col": metrics.get("target_col"),
        "tstr_acc": metrics.get("tstr", {}).get("TSTR_Accuracy"),
        "tstr_auc": metrics.get("tstr", {}).get("TSTR_AUC"),
        "avg_ks": metrics.get("summary", {}).get("average_ks_score"),
    }


def run_repro_benchmark(real_tables: Dict[str, pd.DataFrame], num_rows: int = 120, low_eps: float = 2.0, high_eps: float = 20.0, focus_table: str | None = None, seed: int = 42) -> Dict[str, Dict]:
    """Run reproducible low/high epsilon benchmark and return both results.

    If focus_table is None, auto-detects the best target table for TSTR (same as UI).
    
    Returns a dict with keys 'low' and 'high'.
    """
    # Auto-detect focus table if not provided (matches UI behavior)
    if focus_table is None:
        focus_table, _ = _pick_target_table_and_col(real_tables)
    
    low = _run_case(real_tables, low_eps, num_rows=num_rows, focus_table=focus_table, seed=seed)
    high = _run_case(real_tables, high_eps, num_rows=num_rows, focus_table=focus_table, seed=seed)
    return {"low": low, "high": high}

