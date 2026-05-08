import os
import pandas as pd
from modules.repro_benchmark import run_repro_benchmark

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

print('Loaded tables:', list(real_tables.keys()))

res = run_repro_benchmark(real_tables, num_rows=120, low_eps=2.0, high_eps=20.0, seed=42)
print('--- LOW EPS ---')
print(res['low'])
print('--- HIGH EPS ---')
print(res['high'])
