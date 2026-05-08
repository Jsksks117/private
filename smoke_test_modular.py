import pandas as pd

from modules.module1_schema_parser import parse_schema
from modules.module2_privacy_allocator import allocate_privacy
from modules.module3_synthesis_engine import DPMultiTableSynthesizer
from modules.module5_evaluation_engine import evaluate_synthetic_quality


def main() -> None:
    files = [
        "confidential_flights.csv",
        "confidential_passengers.csv",
        "confidential_tickets.csv",
        "confidential_baggage.csv",
    ]
    tables = {
        f.split(".")[0].replace("confidential_", "").title(): pd.read_csv(f"sample_data/{f}")
        for f in files
    }

    schema_info = parse_schema(tables)
    allocation = allocate_privacy(schema_info["schema_def"], 6.0)

    synthesizer = DPMultiTableSynthesizer(6.0)
    synth_tables = synthesizer.generate(
        real_tables=tables,
        schema_def=schema_info["schema_def"],
        primary_keys=schema_info["primary_keys"],
        relationships=schema_info["relationships"],
        num_rows=80,
    )

    metrics = evaluate_synthetic_quality(tables, synth_tables, schema_info["relationships"])

    print("tables:", list(tables.keys()))
    print("relationships:", schema_info["relationships"])
    print("generation_order:", allocation["generation_order"])
    print("avg_fk_violation:", metrics["summary"]["average_fk_violation_rate"])


if __name__ == "__main__":
    main()
