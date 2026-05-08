import os

import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_complex_samples(
    seed: int = 123,
    out_dir: str = "sample_data_complex",
    num_airports: int = 18,
    num_flights: int = 1200,
    num_passengers: int = 4200,
    num_tickets: int = 5600,
    num_baggage: int = 6100,
    num_claims: int = 1400,
) -> None:
    rng = np.random.default_rng(seed)

    os.makedirs(out_dir, exist_ok=True)

    # Airports: mild regional structure + capacity differences
    airports = pd.DataFrame(
        {
            "id": np.arange(1, num_airports + 1),
            "region_code": rng.choice([0, 1, 2, 3], size=num_airports, p=[0.35, 0.3, 0.2, 0.15]),
            "runway_length_m": rng.normal(3200, 500, size=num_airports).clip(1800, 5200).round(0).astype(int),
            "hub_score": rng.beta(2.0, 4.0, size=num_airports),
        }
    )

    # Flights: multimodal distance, schedule effects, and delay propensity
    dep_peak = rng.choice([0, 1], size=num_flights, p=[0.65, 0.35])
    dep_hour = np.where(dep_peak == 1, rng.normal(18, 3, size=num_flights), rng.normal(9, 4, size=num_flights))
    dep_hour = np.clip(dep_hour, 0, 23).round(0).astype(int)

    mode = rng.choice([0, 1, 2], size=num_flights, p=[0.5, 0.35, 0.15])
    distance = np.where(
        mode == 0,
        rng.normal(550, 130, size=num_flights),
        np.where(mode == 1, rng.normal(1400, 220, size=num_flights), rng.normal(3200, 420, size=num_flights)),
    )
    distance = np.clip(distance, 120, 5200)

    origin_airport_id = rng.integers(1, num_airports + 1, size=num_flights)
    destination_airport_id = rng.integers(1, num_airports + 1, size=num_flights)
    same_airport = origin_airport_id == destination_airport_id
    destination_airport_id[same_airport] = ((destination_airport_id[same_airport]) % num_airports) + 1

    airline_code = rng.choice([0, 1, 2, 3, 4, 5], size=num_flights, p=[0.22, 0.2, 0.18, 0.16, 0.14, 0.1])

    delay_logit = (
        -1.6
        + 0.0013 * distance
        + 0.08 * (dep_hour >= 17)
        + 0.11 * (dep_hour <= 6)
        + 0.06 * (airline_code == 5)
        + rng.normal(0, 0.55, size=num_flights)
    )
    is_delayed = rng.binomial(1, _sigmoid(delay_logit), size=num_flights)

    flights = pd.DataFrame(
        {
            "id": np.arange(1, num_flights + 1),
            "origin_airport_id": origin_airport_id,
            "destination_airport_id": destination_airport_id,
            "distance": distance.round(1),
            "departure_hour": dep_hour,
            "airline_code": airline_code,
            "is_delayed": is_delayed,
        }
    )

    # Passengers: skewed spend + demographics linked to loyalty and route patterns
    passenger_flight_id = rng.integers(1, num_flights + 1, size=num_passengers)
    loyalty_tier = rng.choice([0, 1, 2, 3], size=num_passengers, p=[0.44, 0.31, 0.18, 0.07])
    age = np.clip(rng.normal(38, 13, size=num_passengers), 18, 85).round(0).astype(int)
    group_size = rng.choice([1, 2, 3, 4, 5], size=num_passengers, p=[0.52, 0.25, 0.13, 0.07, 0.03])

    income_base = rng.lognormal(mean=10.7, sigma=0.42, size=num_passengers)
    income_adjust = 1.0 + 0.12 * loyalty_tier + 0.03 * (age > 50)
    annual_income = (income_base * income_adjust).clip(18000, 420000)

    passengers = pd.DataFrame(
        {
            "id": np.arange(1, num_passengers + 1),
            "flight_id": passenger_flight_id,
            "age": age,
            "annual_income": annual_income.round(0),
            "group_size": group_size,
            "loyalty_tier": loyalty_tier,
        }
    )

    # Tickets: depends on route distance, loyalty, and fare class mix
    ticket_passenger_id = rng.integers(1, num_passengers + 1, size=num_tickets)
    ticket_flight_id = passengers.iloc[ticket_passenger_id - 1]["flight_id"].to_numpy()
    flight_distance_for_ticket = flights.iloc[ticket_flight_id - 1]["distance"].to_numpy()

    fare_class = rng.choice([0, 1, 2], size=num_tickets, p=[0.56, 0.31, 0.13])
    purchase_channel = rng.choice([0, 1, 2, 3], size=num_tickets, p=[0.4, 0.22, 0.25, 0.13])

    passenger_loyalty_for_ticket = passengers.iloc[ticket_passenger_id - 1]["loyalty_tier"].to_numpy()
    base_price = 65 + 0.19 * flight_distance_for_ticket
    class_multiplier = np.where(fare_class == 0, 1.0, np.where(fare_class == 1, 1.55, 2.4))
    loyalty_discount = 1.0 - 0.03 * passenger_loyalty_for_ticket
    channel_markup = np.where(purchase_channel == 3, 1.08, 1.0)
    ticket_price = (
        base_price * class_multiplier * loyalty_discount * channel_markup + rng.normal(0, 48, size=num_tickets)
    ).clip(45, 4200)

    ancillaries_spend = (
        rng.gamma(shape=1.9, scale=18.0, size=num_tickets)
        + 4.0 * (fare_class == 2)
        + 2.0 * (fare_class == 1)
        + rng.normal(0, 4, size=num_tickets)
    ).clip(0, 420)

    tickets = pd.DataFrame(
        {
            "id": np.arange(1, num_tickets + 1),
            "passenger_id": ticket_passenger_id,
            "flight_id": ticket_flight_id,
            "fare_class": fare_class,
            "purchase_channel": purchase_channel,
            "ticket_price": ticket_price.round(2),
            "ancillaries_spend": ancillaries_spend.round(2),
        }
    )

    # Baggage: correlated with fare class, group size and route distance
    baggage_ticket_id = rng.integers(1, num_tickets + 1, size=num_baggage)
    baggage_passenger_id = tickets.iloc[baggage_ticket_id - 1]["passenger_id"].to_numpy()

    baggage_fare = tickets.iloc[baggage_ticket_id - 1]["fare_class"].to_numpy()
    baggage_distance = flights.iloc[tickets.iloc[baggage_ticket_id - 1]["flight_id"].to_numpy() - 1]["distance"].to_numpy()
    baggage_group = passengers.iloc[baggage_passenger_id - 1]["group_size"].to_numpy()

    bag_count = rng.choice([1, 2, 3], size=num_baggage, p=[0.67, 0.25, 0.08]) + (baggage_group >= 4).astype(int)
    bag_count = np.clip(bag_count, 1, 4)

    bag_weight = (
        rng.normal(11.5, 3.4, size=num_baggage)
        + 1.8 * bag_count
        + 1.4 * (baggage_fare == 2)
        + 0.8 * (baggage_distance > 2200)
    )
    bag_weight = np.clip(bag_weight, 3.5, 55)

    priority_logit = -1.3 + 0.95 * (baggage_fare == 2) + 0.35 * (baggage_fare == 1) + 0.2 * (bag_count >= 3)
    is_priority = rng.binomial(1, _sigmoid(priority_logit), size=num_baggage)
    oversize_flag = (bag_weight > 30).astype(int)

    baggage = pd.DataFrame(
        {
            "id": np.arange(1, num_baggage + 1),
            "ticket_id": baggage_ticket_id,
            "passenger_id": baggage_passenger_id,
            "bag_count": bag_count,
            "bag_weight": bag_weight.round(2),
            "is_priority": is_priority,
            "oversize_flag": oversize_flag,
        }
    )

    # Claims: rare event table with heavy-tailed amounts
    claim_ticket_id = rng.integers(1, num_tickets + 1, size=num_claims)
    claim_fare = tickets.iloc[claim_ticket_id - 1]["fare_class"].to_numpy()
    claim_bag_heavy = baggage.iloc[rng.integers(0, num_baggage, size=num_claims)]["oversize_flag"].to_numpy()

    claim_type = rng.choice([0, 1, 2, 3], size=num_claims, p=[0.5, 0.22, 0.19, 0.09])
    claim_base = rng.lognormal(mean=4.7, sigma=0.85, size=num_claims)
    claim_amount = claim_base * (1 + 0.2 * claim_type + 0.18 * claim_bag_heavy + 0.1 * (claim_fare == 2))
    claim_amount = np.clip(claim_amount, 30, 9500)

    resolved_days = np.clip(rng.normal(7, 3.2, size=num_claims) + 4 * (claim_type == 3), 1, 45).round(0).astype(int)
    fraud_logit = -2.7 + 0.0002 * claim_amount + 0.2 * (claim_type == 3) + 0.12 * (resolved_days > 12)
    fraud_flag = rng.binomial(1, _sigmoid(fraud_logit), size=num_claims)

    claims = pd.DataFrame(
        {
            "id": np.arange(1, num_claims + 1),
            "ticket_id": claim_ticket_id,
            "claim_type": claim_type,
            "claim_amount": claim_amount.round(2),
            "resolved_days": resolved_days,
            "fraud_flag": fraud_flag,
        }
    )

    # Save with same naming style used by the app's normalizer
    airports.to_csv(os.path.join(out_dir, "confidential_airports.csv"), index=False)
    flights.to_csv(os.path.join(out_dir, "confidential_flights.csv"), index=False)
    passengers.to_csv(os.path.join(out_dir, "confidential_passengers.csv"), index=False)
    tickets.to_csv(os.path.join(out_dir, "confidential_tickets.csv"), index=False)
    baggage.to_csv(os.path.join(out_dir, "confidential_baggage.csv"), index=False)
    claims.to_csv(os.path.join(out_dir, "confidential_claims.csv"), index=False)

    print(f"Generated complex sample dataset in: {out_dir}")
    print("Files:")
    for name in [
        "confidential_airports.csv",
        "confidential_flights.csv",
        "confidential_passengers.csv",
        "confidential_tickets.csv",
        "confidential_baggage.csv",
        "confidential_claims.csv",
    ]:
        print(f"  - {os.path.join(out_dir, name)}")


if __name__ == "__main__":
    generate_complex_samples()
