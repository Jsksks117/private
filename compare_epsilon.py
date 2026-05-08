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

focus_table = "Baggage"
num_rows = 300

def run_with_epsilon(eps):
    schema_info = parse_schema(real_tables)
    alloc = allocate_privacy(schema_info["schema_def"], eps)
    
    synthesizer = DPMultiTableSynthesizer(epsilon=eps, seed=42, focus_runtime_threshold_seconds=90.0)
    
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
    
    metrics = evaluate_synthetic_quality(real_tables=real_tables, synth_tables=synth_tables, relationships=schema_info["relationships"])
    
    return {
        "epsilon": eps,
        "ks": metrics["summary"].get("average_ks_score", 0.0),
        "acc": (metrics["tstr"].get("TSTR_Accuracy", 0) * 100) if metrics["tstr"] else 0,
        "auc": metrics["tstr"].get("TSTR_AUC", 0) if metrics["tstr"] else 0,
    }

print("\n" + "="*80)
print("COMPARISON: Low vs High Epsilon (with fixed training)")
print("="*80)

print("\nRunning epsilon=2.0 (strict privacy)...")
low = run_with_epsilon(2.0)

print("\nRunning epsilon=20.0 (relaxed privacy)...")
high = run_with_epsilon(20.0)

print("\n" + "="*80)
print("RESULTS:")
print("="*80)
print(f"\nepsilon=2.0:   KS={low['ks']:.3f}, Accuracy={low['acc']:.1f}%, AUC={low['auc']:.3f}")
print(f"epsilon=20.0:  KS={high['ks']:.3f}, Accuracy={high['acc']:.1f}%, AUC={high['auc']:.3f}")
print("\n" + "="*80)

diff_ks = abs(high['ks'] - low['ks'])
diff_acc = abs(high['acc'] - low['acc'])
diff_auc = abs(high['auc'] - low['auc'])

print("DIFFERENCES (high - low):")
print("="*80)
print(f"KS difference:        {diff_ks:.3f} (should be > 0.05)")
print(f"Accuracy difference:  {diff_acc:.1f}% (should be > 1%)")
print(f"AUC difference:       {diff_auc:.3f} (should be > 0.01)")
print("="*80)

if diff_ks > 0.05 or diff_acc > 1 or diff_auc > 0.01:
    print("\n✅ SUCCESS! Different epsilon values produce DIFFERENT results!")
else:
    print("\n⚠️  Metrics are still too similar - more tuning needed")
print("="*80 + "\n")
