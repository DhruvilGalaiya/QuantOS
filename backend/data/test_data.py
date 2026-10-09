from ingestion import fetch_market_data


data = fetch_market_data(
    symbol="^NSEI",
    quantos_symbol="NIFTY50",
    period="1mo",
    interval="1d"
)

print("\nData shape:")
print(data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nData types:")
print(data.dtypes)

print("\nFirst 5 rows:")
print(data.head())

print("\nLast 5 rows:")
print(data.tail())