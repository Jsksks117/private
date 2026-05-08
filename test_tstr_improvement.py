import pandas as pd
import time
from modules.module1_schema_parser import parse_schema
from modules.module2_privacy_allocator import allocate_privacy
from modules.module3_synthesis_engine import DPMultiTableSynthesizer
from modules.module5_evaluation_engine import evaluate_synthetic_quality

t = time.time()

files = ['confidential_flights.csv', 'confidential_passengers.csv', 'confidential_tickets.csv', 'confidential_baggage.csv']
tables = {
    f.split('.')[0].replace('confidential_', '').title(): pd.read_csv(f'sample_data_complex_fast/{f}')
    for f in files
}

s = parse_schema(tables)
a = allocate_privacy(s['schema_def'], 6.0)

syn = DPMultiTableSynthesizer(6.0)
out = syn.generate(
    real_tables=tables,
    schema_def=s['schema_def'],
    primary_keys=s['primary_keys'],
    relationships=s['relationships'],
    num_rows=80,
    epsilon_allocation=a['epsilon_allocation']
)

m = evaluate_synthetic_quality(tables, out, s['relationships'])

print(f'Runtime: {round(time.time()-t, 1)}s')
print(f'TSTR Accuracy: {m["tstr"].get("TSTR_Accuracy", 0):.3f}')
print(f'TSTR AUC: {m["tstr"].get("TSTR_AUC", 0):.3f}')
