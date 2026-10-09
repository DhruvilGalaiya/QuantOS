from assets import ASSET_UNIVERSE
from ingestion import fetch_market_data


print("Testing QuantOS multi-asset ingestion")
print("=" * 50)

successful = 0
failed = 0

for key, asset in ASSET_UNIVERSE.items():

    try:

        data = fetch_market_data(
            symbol=asset["symbol"],
            quantos_symbol=key,
            period="1mo",
            interval="1d"
        )

        print(
            f"✓ {key:15} "
            f"{len(data):5} rows"
        )

        successful += 1

    except Exception as e:

        print(
            f"✗ {key:15} FAILED"
        )

        print(
            f"  Error: {e}"
        )

        failed += 1


print("=" * 50)

print(f"Successful: {successful}")
print(f"Failed:     {failed}")
print(f"Total:      {len(ASSET_UNIVERSE)}")