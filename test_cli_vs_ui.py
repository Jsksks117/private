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

# Load tables
files = [f for f in os.listdir(DATADIR) if f.endswith('.csv')]
real_tables = {}
for f in files:
    path = os.path.join(DATADIR, f)
    name = _normalize_table_name(f)
    real_tables[name] = pd.read_csv(path)

print("Loaded tables:", list(real_tables.keys()))

# Run synthesis with same params as UI
epsilon = 10
num_rows = 300
focus_table = "Baggage"

schema_info = parse_schema(real_tables)
alloc = allocate_privacy(schema_info["schema_def"], epsilon)

print(f"\n[CLI RUN] epsilon={epsilon}, num_rows={num_rows}, focus_table={focus_table}, seed=42")

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

metrics = evaluate_synthetic_quality(
    real_tables=real_tables,
    synth_tables=synth_tables,
    relationships=schema_info["relationships"],
)

print("\n=== CLI RESULTS ===")
avg_ks = metrics["summary"].get("average_ks_score", 0.0)
tstr_acc = metrics["tstr"].get("TSTR_Accuracy") if metrics["tstr"] else None
tstr_auc = metrics["tstr"].get("TSTR_AUC") if metrics["tstr"] else None
avg_card_error = metrics["summary"].get("average_cardinality_error", 0.0)

print(f"Average KS: {avg_ks:.3f}")
print(f"TSTR Accuracy: {tstr_acc * 100:.1f}%" if tstr_acc is not None else "TSTR Accuracy: N/A")
print(f"TSTR AUC: {tstr_auc:.3f}" if tstr_auc is not None else "TSTR AUC: N/A")
print(f"Cardinality Error: {avg_card_error:.3f}")
print(f"Target table: {metrics.get('target_table')}, Target col: {metrics.get('target_col')}")

print("\n=== EXPECTED FROM UI ===")
print("Average KS: 0.274")
print("TSTR Accuracy: 59.9%")
print("TSTR AUC: 0.509")
print("Cardinality Error: 0.000")
print("Target table: Baggage, Target col: is_priority")

print("\n=== MATCH? ===")
ks_match = abs(avg_ks - 0.274) < 0.01
acc_match = abs((tstr_acc or 0) - 0.599) < 0.01 if tstr_acc else False
auc_match = abs((tstr_auc or 0) - 0.509) < 0.01 if tstr_auc else False
card_match = abs(avg_card_error - 0.0) < 0.01

print(f"KS match: {ks_match} (diff: {abs(avg_ks - 0.274):.3f})")
print(f"Acc match: {acc_match} (diff: {abs((tstr_acc or 0) - 0.599):.3f})")
print(f"AUC match: {auc_match} (diff: {abs((tstr_auc or 0) - 0.509):.3f})")
print(f"Card match: {card_match} (diff: {abs(avg_card_error - 0.0):.3f})")
