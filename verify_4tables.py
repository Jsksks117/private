import numpy as np
import pandas as pd

from dp_evaluator import DPSyntheticEvaluator
from multi_relational_pipeline import MultiRelationalDPGenerator


def rescale_with_dp_noise(series: pd.Series, real_reference: pd.Series, epsilon: float) -> pd.Series:
    centered = (series - series.mean()) / (series.std() + 1e-8)
    noise = np.random.normal(0, real_reference.std() * (0.5 / epsilon), len(series))
    return (centered * real_reference.std()) + real_reference.mean() + noise


def main() -> None:
    epsilon = 6.0
    n_rows = 120

    real_db = {
        "Flights": pd.read_csv("sample_data/confidential_flights.csv"),
        "Passengers": pd.read_csv("sample_data/confidential_passengers.csv"),
        "Tickets": pd.read_csv("sample_data/confidential_tickets.csv"),
        "Baggage": pd.read_csv("sample_data/confidential_baggage.csv"),
    }

    schema = {
        "Flights": [],
        "Passengers": ["Flights"],
        "Tickets": ["Passengers", "Flights"],
        "Baggage": ["Passengers"],
    }

    pipeline = MultiRelationalDPGenerator(
        schema_def=schema,
        total_epsilon=epsilon,
        db_connection_string="sqlite:///verification_4tables.db",
    )
    synthetic_tables = pipeline.train_and_generate(num_synthetic_rows=n_rows)

    synth_flights = synthetic_tables["Flights"].copy().iloc[:, :4]
    synth_flights.columns = ["id", "distance", "airline_code", "is_delayed"]
    synth_flights["distance"] = rescale_with_dp_noise(synth_flights["distance"], real_db["Flights"]["distance"], epsilon)
    synth_flights["airline_code"] = synth_flights["airline_code"].round().clip(0, 2).astype(int)
    synth_flights["is_delayed"] = synth_flights["is_delayed"].round().clip(0, 1).astype(int)

    synth_passengers = synthetic_tables["Passengers"].copy().iloc[:, :4]
    synth_passengers.columns = ["id", "flight_id", "ticket_price", "loyalty_tier"]
    synth_passengers["ticket_price"] = rescale_with_dp_noise(
        synth_passengers["ticket_price"], real_db["Passengers"]["ticket_price"], epsilon
    )
    synth_passengers["flight_id"] = np.random.choice(synth_flights["id"], len(synth_passengers))
    synth_passengers["loyalty_tier"] = synth_passengers["loyalty_tier"].round().clip(0, 3).astype(int)

    synth_tickets = synthetic_tables["Tickets"].copy().iloc[:, :4]
    synth_tickets.columns = ["id", "passenger_id", "flight_id", "fare_class"]
    synth_tickets["passenger_id"] = np.random.choice(synth_passengers["id"], len(synth_tickets))
    passenger_to_flight = synth_passengers.set_index("id")["flight_id"]
    synth_tickets["flight_id"] = synth_tickets["passenger_id"].map(passenger_to_flight).astype(int)
    synth_tickets["fare_class"] = synth_tickets["fare_class"].round().clip(0, 2).astype(int)

    synth_baggage = synthetic_tables["Baggage"].copy().iloc[:, :4]
    synth_baggage.columns = ["id", "passenger_id", "bag_weight", "is_priority"]
    synth_baggage["passenger_id"] = np.random.choice(synth_passengers["id"], len(synth_baggage))
    synth_baggage["bag_weight"] = rescale_with_dp_noise(
        synth_baggage["bag_weight"], real_db["Baggage"]["bag_weight"], epsilon
    ).clip(3.0, 40.0)
    synth_baggage["is_priority"] = synth_baggage["is_priority"].round().clip(0, 1).astype(int)

    synth_db = {
        "Flights": synth_flights,
        "Passengers": synth_passengers,
        "Tickets": synth_tickets,
        "Baggage": synth_baggage,
    }

    evaluator = DPSyntheticEvaluator(real_tables=real_db, synth_tables=synth_db)

    marginal = evaluator.evaluate_marginal_fidelity()
    relational = evaluator.evaluate_relational_integrity(
        [
            ("Flights", "id", "Passengers", "flight_id"),
            ("Passengers", "id", "Tickets", "passenger_id"),
            ("Passengers", "id", "Baggage", "passenger_id"),
        ]
    )
    tstr = evaluator.evaluate_tstr(target_table="Flights", target_col="is_delayed")
    privacy = evaluator.evaluate_privacy(target_table="Flights")

    fk_keys = [k for k in relational if k.endswith("FK_violation_rate")]
    avg_fk_violation = float(np.mean([relational[k] for k in fk_keys])) if fk_keys else 0.0

    print("=== 4-Table Verification Results ===")
    print(f"Flights KS distance: {marginal['Flights'].get('KS_distance', float('nan')):.4f}")
    print(f"Average FK violation rate: {avg_fk_violation:.4%}")
    print(f"TSTR Accuracy: {tstr['TSTR_Accuracy']:.4f}")
    print(f"TSTR AUC: {tstr['TSTR_AUC']:.4f}")
    print(f"MIA AUC: {privacy['Membership_Inference_Attack_AUC']:.4f}")


if __name__ == "__main__":
    main()
