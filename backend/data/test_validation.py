from ingestion import fetch_market_data
from validation import validate_market_data


data = fetch_market_data(
    symbol="^NSEI",
    quantos_symbol="NIFTY50",
    period="max",
    interval="1d"
)

result = validate_market_data(data)

print("\nValidation Result:")
print(result)