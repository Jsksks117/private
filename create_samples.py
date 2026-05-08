import os

import numpy as np
import pandas as pd

os.makedirs("sample_data", exist_ok=True)
np.random.seed(42)

num_flights = 400
num_passengers = 900
num_tickets = 1200
num_baggage = 1100

flights = pd.DataFrame(
    {
        "id": range(1, num_flights + 1),
        "distance": np.random.normal(500, 150, num_flights),
        "airline_code": np.random.choice([0, 1, 2], num_flights),
        "is_delayed": np.random.choice([0, 1], num_flights),
    }
)
flights.to_csv("sample_data/confidential_flights.csv", index=False)

passengers = pd.DataFrame(
    {
        "id": range(1, num_passengers + 1),
        "flight_id": np.random.choice(range(1, num_flights + 1), num_passengers),
        "ticket_price": np.random.uniform(100, 5000, num_passengers),
        "loyalty_tier": np.random.choice([0, 1, 2, 3], num_passengers, p=[0.4, 0.3, 0.2, 0.1]),
    }
)
passengers.to_csv("sample_data/confidential_passengers.csv", index=False)

tickets = pd.DataFrame(
    {
        "id": range(1, num_tickets + 1),
        "passenger_id": np.random.choice(passengers["id"], num_tickets),
        "flight_id": np.random.choice(flights["id"], num_tickets),
        "fare_class": np.random.choice([0, 1, 2], num_tickets, p=[0.55, 0.3, 0.15]),
    }
)
tickets.to_csv("sample_data/confidential_tickets.csv", index=False)

baggage = pd.DataFrame(
    {
        "id": range(1, num_baggage + 1),
        "passenger_id": np.random.choice(passengers["id"], num_baggage),
        "bag_weight": np.random.uniform(5, 35, num_baggage),
        "is_priority": np.random.choice([0, 1], num_baggage, p=[0.75, 0.25]),
    }
)
baggage.to_csv("sample_data/confidential_baggage.csv", index=False)

print("4-table samples generated in sample_data/")
