from assets import ASSET_UNIVERSE


print("QuantOS Asset Universe")
print("=" * 40)

for key, asset in ASSET_UNIVERSE.items():
    print(
        f"{key:15} | "
        f"{asset['symbol']:15} | "
        f"{asset['asset_class']:20} | "
        f"{asset['region']}"
    )

print("=" * 40)
print(f"Total assets: {len(ASSET_UNIVERSE)}")
