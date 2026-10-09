from src.upstox_client import get
from src.config import (
    REGIME_INSTRUMENTS,
    VALIDATION_INSTRUMENTS,
)


all_instruments = {
    **REGIME_INSTRUMENTS,
    **VALIDATION_INSTRUMENTS,
}


print("\nQuantOS Instrument Configuration")
print("=" * 70)

for name, instrument_key in all_instruments.items():

    data = get(
        "/v2/instruments/search",
        params={
            "query": name.replace("_", " "),
            "exchanges": "NSE,BSE",
            "segments": "INDEX",
            "records": 20,
        }
    )

    matches = data.get("data", [])

    print(f"\n{name}")
    print(f"Configured key: {instrument_key}")

    exact_match = [
        x for x in matches
        if x.get("instrument_key") == instrument_key
    ]

    if exact_match:
        print("✓ VERIFIED")
        print(
            f"  Name: {exact_match[0].get('name')}"
        )
        print(
            f"  Key:  {exact_match[0].get('instrument_key')}"
        )
    else:
        print("✗ NOT VERIFIED")
        print("Available matches:")

        for x in matches[:10]:
            print(
                f"  {x.get('name')} | "
                f"{x.get('instrument_key')}"
            )