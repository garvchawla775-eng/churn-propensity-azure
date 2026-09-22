"""
Generates a synthetic customer churn / propensity dataset structured like a
telecom or retail-banking customer base: tenure, product usage, support
interactions, contract type, and a churn label driven by a realistic (if
synthetic) combination of those features.

Synthetic data — built to mirror the structure of the classic Telco Customer
Churn dataset since live internet access to Kaggle wasn't available in this
environment, but with the same feature shape and a comparable ~26% churn rate.
"""
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customer_churn.csv"

np.random.seed(7)
N = 5000

tenure_months = np.random.exponential(24, N).clip(0, 72).round().astype(int)
monthly_charge = np.random.normal(70, 25, N).clip(15, 150).round(2)
contract_type = np.random.choice(["month-to-month", "one_year", "two_year"], N, p=[0.55, 0.25, 0.20])
num_support_calls = np.random.poisson(1.5, N)
has_autopay = np.random.choice([0, 1], N, p=[0.4, 0.6])
num_products = np.random.choice([1, 2, 3, 4], N, p=[0.35, 0.30, 0.20, 0.15])
satisfaction_score = np.random.randint(1, 6, N)  # 1-5 survey score

# churn probability: higher with month-to-month, short tenure, many support
# calls, low satisfaction, no autopay -- classic churn drivers
logit = (
    -1.2
    + 1.3 * (contract_type == "month-to-month")
    - 0.55 * (contract_type == "two_year")
    - 0.035 * tenure_months
    + 0.28 * num_support_calls
    - 0.5 * has_autopay
    - 0.35 * (satisfaction_score - 3)
    - 0.12 * num_products
    + 0.006 * (monthly_charge - 70)
)
churn_prob = 1 / (1 + np.exp(-logit))
churned = np.random.binomial(1, churn_prob)

df = pd.DataFrame({
    "customer_id": [f"C{100000+i}" for i in range(N)],
    "tenure_months": tenure_months,
    "monthly_charge": monthly_charge,
    "contract_type": contract_type,
    "num_support_calls": num_support_calls,
    "has_autopay": has_autopay,
    "num_products": num_products,
    "satisfaction_score": satisfaction_score,
    "churned": churned,
})

DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(DATA_PATH, index=False)
print(f"Generated {len(df)} rows -> data/customer_churn.csv")
print(f"Churn rate: {df['churned'].mean():.1%}")
print(df.head())
