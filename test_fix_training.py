import os
import pandas as pd
from modules.module1_schema_parser import parse_schema
from modules.module2_privacy_allocator import allocate_privacy
from modules.module3_synthesis_engine import DPMultiTableSynthesizer
from modules.module5_evaluation_engine import evaluate_synthetic_quality

DATADIR = "sample_data_extreme"

def _normalize_table_name(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    base = base.replace("confidential_", "")
    base = base.replace("-", "_").replace(" ", "_")
    return base.title()

files = [f for f in os.listdir(DATADIR) if f.endswith('.csv')]
real_tables = {}
for f in files:
    path = os.path.join(DATADIR, f)
    name = _normalize_table_name(f)
    real_tables[name] = pd.read_csv(path)

print("\n" + "="*70)
print("TEST: Fixed training with reduced batch size + min steps")
print("="*70)

epsilon = 10
num_rows = 300
focus_table = "Baggage"

schema_info = parse_schema(real_tables)
alloc = allocate_privacy(schema_info["schema_def"], epsilon)

print(f"epsilon={epsilon}, num_rows={num_rows}, focus_table={focus_table}\n")

synthesizer = DPMultiTableSynthesizer(
    epsilon=epsilon,
    db_connection_string="sqlite:///poc_modular.db",
    seed=42,
    focus_runtime_threshold_seconds=90.0,
)

synth_tables = synthesizer.generate(
    real_tables=real_tables,
    schema_def=schema_info["schema_def"],
    primary_keys=schema_info["primary_keys"],
    relationships=schema_info["relationships"],
    num_rows=int(num_rows),
    epsilon_allocation=alloc["epsilon_allocation"],
    focus_table=focus_table,
    table_row_counts={table_name: len(df) for table_name, df in real_tables.items()},
)

print("\nEvaluating quality...")
metrics = evaluate_synthetic_quality(
    real_tables=real_tables,
    synth_tables=synth_tables,
    relationships=schema_info["relationships"],
)

avg_ks = metrics["summary"].get("average_ks_score", 0.0)
tstr_acc = metrics["tstr"].get("TSTR_Accuracy") if metrics["tstr"] else None
tstr_auc = metrics["tstr"].get("TSTR_AUC") if metrics["tstr"] else None
avg_card = metrics["summary"].get("average_cardinality_error", 0.0)

print("\n" + "="*70)
print("RESULTS WITH FIX:")
print("="*70)
print(f"Average KS:       {avg_ks:.3f}")
print(f"TSTR Accuracy:    {(tstr_acc * 100):.1f}%" if tstr_acc else "TSTR Accuracy:    N/A")
print(f"TSTR AUC:         {tstr_auc:.3f}" if tstr_auc else "TSTR AUC:         N/A")
print(f"Cardinality Err:  {avg_card:.3f}")
print("="*70)
print("\nExpected: metrics should now VARY with epsilon (not all identical)")
print("="*70 + "\n")
