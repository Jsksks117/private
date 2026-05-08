import os

import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_extreme_samples(
    seed: int = 321,
    out_dir: str = "sample_data_extreme",
    num_airports: int = 12,
    num_flights: int = 720,
    num_passengers: int = 1800,
    num_tickets: int = 3000,
    num_baggage: int = 3800,
    num_claims: int = 420,
) -> None:
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    # Airports: clustered hubs with extreme throughput differences.
    airport_tier = rng.choice([0, 1, 2], size=num_airports, p=[0.55, 0.30, 0.15])
    hub_score = np.where(
        airport_tier == 2,
        rng.uniform(0.70, 0.98, size=num_airports),
        np.where(airport_tier == 1, rng.uniform(0.30, 0.75, size=num_airports), rng.uniform(0.02, 0.35, size=num_airports)),
    )
    runway_length_m = np.where(
        airport_tier == 2,
        rng.normal(4200, 180, size=num_airports),
        np.where(airport_tier == 1, rng.normal(3200, 250, size=num_airports), rng.normal(2200, 180, size=num_airports)),
    ).clip(1400, 5200)
    region_code = np.where(airport_tier == 2, rng.choice([1, 2], size=num_airports), rng.choice([0, 1, 2, 3], size=num_airports))
    airports = pd.DataFrame(
        {
            "id": np.arange(1, num_airports + 1),
            "region_code": region_code,
            "airport_tier": airport_tier,
            "runway_length_m": runway_length_m.round(0).astype(int),
            "hub_score": hub_score.round(3),
            "traffic_class": np.where(hub_score > 0.75, 3, np.where(hub_score > 0.45, 2, np.where(hub_score > 0.2, 1, 0))),
        }
    )

    # Flights: multimodal routes with a sharp nonlinear delay signal.
    dep_peak = rng.choice([0, 1], size=num_flights, p=[0.72, 0.28])
    departure_hour = np.where(dep_peak == 1, rng.normal(19, 2.0, size=num_flights), rng.normal(10, 3.5, size=num_flights))
    departure_hour = np.clip(departure_hour, 0, 23).round(0).astype(int)

    route_mode = rng.choice([0, 1, 2, 3], size=num_flights, p=[0.38, 0.28, 0.22, 0.12])
    distance = np.select(
        [route_mode == 0, route_mode == 1, route_mode == 2, route_mode == 3],
        [rng.normal(320, 70, size=num_flights), rng.normal(1100, 160, size=num_flights), rng.normal(2400, 260, size=num_flights), rng.normal(4100, 340, size=num_flights)],
    )
    distance = np.clip(distance, 150, 6500)

    origin_airport_probs = np.array([0.30, 0.18, 0.14, 0.12, 0.10, 0.08, 0.04, 0.03, 0.02, 0.006, 0.004, 0.004], dtype=float)
    origin_airport_probs = origin_airport_probs / origin_airport_probs.sum()
    origin_airport_id = rng.choice(airports["id"], size=num_flights, replace=True, p=origin_airport_probs)
    destination_airport_probs = np.array([0.18, 0.16, 0.12, 0.12, 0.10, 0.09, 0.08, 0.06, 0.04, 0.03, 0.02, 0.00], dtype=float)
    destination_airport_probs = destination_airport_probs / destination_airport_probs.sum()
    destination_airport_id = rng.choice(airports["id"], size=num_flights, replace=True, p=destination_airport_probs)
    same_airport = origin_airport_id == destination_airport_id
    destination_airport_id[same_airport] = ((destination_airport_id[same_airport]) % num_airports) + 1

    origin_hub_score = airports.iloc[origin_airport_id - 1]["hub_score"].to_numpy()
    destination_hub_score = airports.iloc[destination_airport_id - 1]["hub_score"].to_numpy()
    origin_tier = airports.iloc[origin_airport_id - 1]["airport_tier"].to_numpy()
    destination_tier = airports.iloc[destination_airport_id - 1]["airport_tier"].to_numpy()

    airline_code = rng.choice([0, 1, 2, 3, 4, 5, 6, 7], size=num_flights, p=[0.30, 0.22, 0.16, 0.12, 0.08, 0.05, 0.04, 0.03])
    congestion = rng.beta(2.0, 6.0, size=num_flights)
    weather_risk = rng.beta(1.5, 4.5, size=num_flights)

    severe_route = ((distance > 2200) & (congestion > 0.50) & (weather_risk > 0.40)).astype(float)
    late_storm_window = (((departure_hour >= 19) | (departure_hour <= 5)) & (weather_risk > 0.45)).astype(float)
    carrier_penalty = np.isin(airline_code, [5, 6, 7]).astype(float)
    hub_pair_pressure = ((origin_hub_score < 0.25) | (destination_hub_score < 0.25)).astype(float)
    tier_mismatch = (origin_tier != destination_tier).astype(float)
    hub_chain = ((origin_tier == 0) & (destination_tier == 2)).astype(float)
    delay_logit = (
        -4.8
        + 2.8 * severe_route
        + 2.0 * late_storm_window
        + 1.4 * carrier_penalty
        + 1.1 * hub_pair_pressure
        + 0.9 * tier_mismatch
        + 1.2 * hub_chain
        + 0.55 * (departure_hour >= 20)
        + 0.50 * (departure_hour <= 4)
        + 0.30 * congestion
        + 0.25 * weather_risk
        + rng.normal(0, 0.18, size=num_flights)
    )
    is_delayed = rng.binomial(1, _sigmoid(delay_logit), size=num_flights)

    flights = pd.DataFrame(
        {
            "id": np.arange(1, num_flights + 1),
            "origin_airport_id": origin_airport_id,
            "destination_airport_id": destination_airport_id,
            "distance": distance.round(1),
            "departure_hour": departure_hour,
            "airline_code": airline_code,
            "congestion": congestion.round(3),
            "weather_risk": weather_risk.round(3),
            "is_delayed": is_delayed,
        }
    )

    # Passengers: heavy income skew + loyalty + booking behavior.
    passenger_flight_id = rng.integers(1, num_flights + 1, size=num_passengers)
    loyalty_tier = rng.choice([0, 1, 2, 3, 4], size=num_passengers, p=[0.35, 0.28, 0.18, 0.12, 0.07])
    age = np.clip(rng.normal(36, 15, size=num_passengers), 18, 88).round(0).astype(int)
    family_size = rng.choice([1, 2, 3, 4, 5, 6], size=num_passengers, p=[0.45, 0.23, 0.14, 0.09, 0.06, 0.03])
    booking_lead_days = np.clip(rng.gamma(shape=2.1, scale=7.5, size=num_passengers), 0, 120).round(0).astype(int)

    income_base = rng.lognormal(mean=10.55, sigma=0.75, size=num_passengers)
    income_adjust = 1.0 + 0.18 * loyalty_tier + 0.05 * (family_size >= 4) + 0.06 * (age > 45)
    annual_income = (income_base * income_adjust).clip(12000, 750000)

    travel_risk = _sigmoid(-1.6 + 0.03 * (family_size >= 4) + 0.02 * (booking_lead_days < 5) - 0.08 * loyalty_tier + rng.normal(0, 0.55, size=num_passengers))
    frequent_flyer = rng.binomial(1, _sigmoid(-0.9 + 0.55 * loyalty_tier + 0.015 * booking_lead_days), size=num_passengers)

    passengers = pd.DataFrame(
        {
            "id": np.arange(1, num_passengers + 1),
            "flight_id": passenger_flight_id,
            "age": age,
            "annual_income": annual_income.round(0),
            "family_size": family_size,
            "booking_lead_days": booking_lead_days,
            "loyalty_tier": loyalty_tier,
            "travel_risk": travel_risk.round(3),
            "frequent_flyer": frequent_flyer,
        }
    )

    # Tickets: nonlinear price with strong class/channel effects.
    ticket_passenger_id = rng.integers(1, num_passengers + 1, size=num_tickets)
    ticket_flight_id = passengers.iloc[ticket_passenger_id - 1]["flight_id"].to_numpy()
    flight_distance_for_ticket = flights.iloc[ticket_flight_id - 1]["distance"].to_numpy()
    flight_delay_for_ticket = flights.iloc[ticket_flight_id - 1]["is_delayed"].to_numpy()
    flight_congestion_for_ticket = flights.iloc[ticket_flight_id - 1]["congestion"].to_numpy()

    fare_class = rng.choice([0, 1, 2, 3], size=num_tickets, p=[0.46, 0.28, 0.18, 0.08])
    purchase_channel = rng.choice([0, 1, 2, 3, 4], size=num_tickets, p=[0.34, 0.22, 0.16, 0.18, 0.10])
    seat_zone = rng.choice([0, 1, 2], size=num_tickets, p=[0.55, 0.30, 0.15])

    passenger_loyalty_for_ticket = passengers.iloc[ticket_passenger_id - 1]["loyalty_tier"].to_numpy()
    passenger_income_for_ticket = passengers.iloc[ticket_passenger_id - 1]["annual_income"].to_numpy()
    booking_lead_for_ticket = passengers.iloc[ticket_passenger_id - 1]["booking_lead_days"].to_numpy()
    frequent_flyer_for_ticket = passengers.iloc[ticket_passenger_id - 1]["frequent_flyer"].to_numpy()

    base_price = 42 + 0.22 * flight_distance_for_ticket + 26 * flight_congestion_for_ticket + 11 * flight_delay_for_ticket
    class_multiplier = np.select(
        [fare_class == 0, fare_class == 1, fare_class == 2, fare_class == 3],
        [1.0, 1.45, 2.1, 3.0],
    )
    loyalty_discount = 1.0 - 0.025 * passenger_loyalty_for_ticket - 0.012 * frequent_flyer_for_ticket
    channel_markup = np.where(purchase_channel == 4, 1.12, np.where(purchase_channel == 3, 1.06, 1.0))
    seat_markup = np.where(seat_zone == 2, 1.18, np.where(seat_zone == 1, 1.05, 0.96))
    income_markup = 1.0 + 0.00000055 * np.maximum(passenger_income_for_ticket - 60000, 0)
    booking_markup = 1.0 - 0.0015 * np.minimum(booking_lead_for_ticket, 40)

    ticket_price = (
        base_price * class_multiplier * loyalty_discount * channel_markup * seat_markup * income_markup * booking_markup
        + rng.normal(0, 28, size=num_tickets)
    ).clip(35, 9500)

    ancillary_bundle = rng.gamma(shape=1.5, scale=24.0, size=num_tickets)
    ancillaries_spend = (
        ancillary_bundle
        + 9.0 * (fare_class >= 2)
        + 3.5 * frequent_flyer_for_ticket
        + rng.normal(0, 6, size=num_tickets)
    ).clip(0, 650)

    tickets = pd.DataFrame(
        {
            "id": np.arange(1, num_tickets + 1),
            "passenger_id": ticket_passenger_id,
            "flight_id": ticket_flight_id,
            "fare_class": fare_class,
            "purchase_channel": purchase_channel,
            "seat_zone": seat_zone,
            "ticket_price": ticket_price.round(2),
            "ancillaries_spend": ancillaries_spend.round(2),
        }
    )

    # Baggage: more complex, mixed discrete/continuous distribution with rare oversized bags.
    baggage_ticket_id = rng.integers(1, num_tickets + 1, size=num_baggage)
    baggage_passenger_id = tickets.iloc[baggage_ticket_id - 1]["passenger_id"].to_numpy()
    baggage_fare = tickets.iloc[baggage_ticket_id - 1]["fare_class"].to_numpy()
    baggage_distance = flights.iloc[tickets.iloc[baggage_ticket_id - 1]["flight_id"].to_numpy() - 1]["distance"].to_numpy()
    baggage_delay = flights.iloc[tickets.iloc[baggage_ticket_id - 1]["flight_id"].to_numpy() - 1]["is_delayed"].to_numpy()
    baggage_family = passengers.iloc[baggage_passenger_id - 1]["family_size"].to_numpy()
    baggage_risk = passengers.iloc[baggage_passenger_id - 1]["travel_risk"].to_numpy()

    bag_count = (
        rng.choice([1, 2, 3, 4], size=num_baggage, p=[0.44, 0.31, 0.17, 0.08])
        + (baggage_family >= 4).astype(int)
        + (baggage_fare >= 2).astype(int)
    )
    bag_count = np.clip(bag_count, 1, 6)

    bag_weight = (
        rng.normal(10.0, 2.8, size=num_baggage)
        + 2.4 * bag_count
        + 1.2 * (baggage_fare >= 2)
        + 1.3 * (baggage_distance > 2500)
        + 0.9 * baggage_delay
        + 1.1 * baggage_risk
        + rng.gamma(shape=1.2, scale=2.0, size=num_baggage)
    )
    bag_weight = np.clip(bag_weight, 2.0, 72.0)

    oversize_flag = (bag_weight > 28).astype(int)
    priority_logit = -1.8 + 0.85 * (baggage_fare >= 2) + 0.45 * oversize_flag + 0.25 * (bag_count >= 4) + rng.normal(0, 0.2, size=num_baggage)
    is_priority = rng.binomial(1, _sigmoid(priority_logit), size=num_baggage)
    fragile_flag = rng.binomial(1, _sigmoid(-1.4 + 0.55 * baggage_risk + 0.4 * (baggage_fare == 3)), size=num_baggage)

    baggage = pd.DataFrame(
        {
            "id": np.arange(1, num_baggage + 1),
            "ticket_id": baggage_ticket_id,
            "passenger_id": baggage_passenger_id,
            "bag_count": bag_count,
            "bag_weight": bag_weight.round(2),
            "is_priority": is_priority,
            "oversize_flag": oversize_flag,
            "fragile_flag": fragile_flag,
        }
    )

    # Claims: rare, heavy-tailed fraud-like outcomes.
    claim_ticket_id = rng.integers(1, num_tickets + 1, size=num_claims)
    claim_fare = tickets.iloc[claim_ticket_id - 1]["fare_class"].to_numpy()
    claim_delay = flights.iloc[tickets.iloc[claim_ticket_id - 1]["flight_id"].to_numpy() - 1]["is_delayed"].to_numpy()
    claim_oversize = baggage.iloc[rng.integers(0, num_baggage, size=num_claims)]["oversize_flag"].to_numpy()
    claim_priority = baggage.iloc[rng.integers(0, num_baggage, size=num_claims)]["is_priority"].to_numpy()

    claim_type = rng.choice([0, 1, 2, 3, 4], size=num_claims, p=[0.42, 0.22, 0.16, 0.12, 0.08])
    claim_base = rng.lognormal(mean=4.9, sigma=1.0, size=num_claims)
    claim_amount = claim_base * (1 + 0.18 * claim_type + 0.22 * claim_oversize + 0.12 * claim_priority + 0.10 * claim_delay + 0.08 * (claim_fare >= 2))
    claim_amount = np.clip(claim_amount, 25, 18000)

    resolved_days = np.clip(rng.normal(8, 3.5, size=num_claims) + 5 * (claim_type == 4) + 3 * claim_oversize, 1, 60).round(0).astype(int)
    fraud_logit = -3.1 + 0.00018 * claim_amount + 0.35 * (claim_type == 4) + 0.18 * (resolved_days > 14) + 0.14 * claim_oversize + rng.normal(0, 0.25, size=num_claims)
    fraud_flag = rng.binomial(1, _sigmoid(fraud_logit), size=num_claims)
    dispute_flag = rng.binomial(1, _sigmoid(-2.4 + 0.28 * fraud_flag + 0.08 * (claim_amount > 2500)), size=num_claims)

    claims = pd.DataFrame(
        {
            "id": np.arange(1, num_claims + 1),
            "ticket_id": claim_ticket_id,
            "claim_type": claim_type,
            "claim_amount": claim_amount.round(2),
            "resolved_days": resolved_days,
            "fraud_flag": fraud_flag,
            "dispute_flag": dispute_flag,
        }
    )

    airports.to_csv(os.path.join(out_dir, "confidential_airports.csv"), index=False)
    flights.to_csv(os.path.join(out_dir, "confidential_flights.csv"), index=False)
    passengers.to_csv(os.path.join(out_dir, "confidential_passengers.csv"), index=False)
    tickets.to_csv(os.path.join(out_dir, "confidential_tickets.csv"), index=False)
    baggage.to_csv(os.path.join(out_dir, "confidential_baggage.csv"), index=False)
    claims.to_csv(os.path.join(out_dir, "confidential_claims.csv"), index=False)

    print(f"Generated extreme sample dataset in: {out_dir}")
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
    generate_extreme_samples()
