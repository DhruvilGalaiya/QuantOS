from asset_universe import list_quantos_assets


assets = list_quantos_assets()

print("=" * 60)
print("QUANTOS PORTFOLIO UNIVERSE")
print("=" * 60)

print(f"Total assets: {len(assets)}")

for asset in assets:
    print(
        f"{asset.symbol:20} "
        f"{asset.market:8} "
        f"{asset.currency:5} "
        f"{asset.asset_class}"
    )

print("=" * 60)