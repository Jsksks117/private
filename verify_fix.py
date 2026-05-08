import os
import pandas as pd
from modules.module1_schema_parser import parse_schema
from modules.module2_privacy_allocator import allocate_privacy
from modules.module3_synthesis_engine import DPMultiTableSynthesizer
from modules.module5_evaluation_engine import evaluate_synthetic_quality
import sys

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

# Run synthesis
epsilon = 10
num_rows = 300
focus_table = "Baggage"

schema_info = parse_schema(real_tables)
alloc = allocate_privacy(schema_info["schema_def"], epsilon)

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

avg_ks = metrics["summary"].get("average_ks_score", 0.0)
tstr_acc = metrics["tstr"].get("TSTR_Accuracy") if metrics["tstr"] else None
tstr_auc = metrics["tstr"].get("TSTR_AUC") if metrics["tstr"] else None
avg_card_error = metrics["summary"].get("average_cardinality_error", 0.0)

print("\n" + "="*60)
print("CLI RESULTS (epsilon=10, num_rows=300, focus_table=Baggage)")
print("="*60)
print(f"Average KS:       {avg_ks:.3f}")
print(f"TSTR Accuracy:    {(tstr_acc * 100):.1f}%" if tstr_acc else "TSTR Accuracy:    N/A")
print(f"TSTR AUC:         {tstr_auc:.3f}" if tstr_auc else "TSTR AUC:         N/A")
print(f"Cardinality Err:  {avg_card_error:.3f}")
print("="*60)
print("\nEXPECTED FROM UI:")
print("="*60)
print(f"Average KS:       0.274")
print(f"TSTR Accuracy:    59.9%")
print(f"TSTR AUC:         0.509")
print(f"Cardinality Err:  0.000")
print("="*60)

print("\nMATCH STATUS:")
print("="*60)
ks_close = abs(avg_ks - 0.274) < 0.02
acc_close = abs((tstr_acc or 0) - 0.599) < 0.02 if tstr_acc else False
auc_close = abs((tstr_auc or 0) - 0.509) < 0.02 if tstr_auc else False
card_close = abs(avg_card_error - 0.0) < 0.02

print(f"✓ KS:       {ks_close} (CLI: {avg_ks:.3f}, UI: 0.274, diff: {abs(avg_ks - 0.274):.3f})")
print(f"✓ Accuracy: {acc_close} (CLI: {(tstr_acc * 100):.1f}%, UI: 59.9%, diff: {abs((tstr_acc or 0) * 100 - 59.9):.1f}%)")
print(f"✓ AUC:      {auc_close} (CLI: {tstr_auc:.3f}, UI: 0.509, diff: {abs((tstr_auc or 0) - 0.509):.3f})")
print(f"✓ Cardinality: {card_close} (CLI: {avg_card_error:.3f}, UI: 0.000, diff: {abs(avg_card_error):.3f})")
print("="*60)

if ks_close and acc_close and auc_close and card_close:
    print("\n✅ FIXED! CLI and UI results match within tolerance (±0.02)!")
else:
    print("\n⚠️  Some metrics differ. Check the diffs above.")
print("="*60)
