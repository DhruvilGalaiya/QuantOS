from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Asset:
    symbol: str
    display_name: str
    asset_class: str
    market: str
    currency: str
    data_source: str
    instrument_key: Optional[str] = None


QUANTOS_ASSETS: Dict[str, Asset] = {

    # ==========================================================
    # INDIA
    # ==========================================================

    "NIFTY_50": Asset(
        symbol="NIFTY_50",
        display_name="NIFTY 50",
        asset_class="INDEX",
        market="INDIA",
        currency="INR",
        data_source="UPSTOX",
        instrument_key="NSE_INDEX|Nifty 50",
    ),

    "NIFTY_BANK": Asset(
        symbol="NIFTY_BANK",
        display_name="NIFTY Bank",
        asset_class="INDEX",
        market="INDIA",
        currency="INR",
        data_source="UPSTOX",
        instrument_key="NSE_INDEX|Nifty Bank",
    ),

    "NIFTY_MIDCAP_100": Asset(
    symbol="NIFTY_MIDCAP_100",
    display_name="NIFTY Midcap 100",
    asset_class="INDEX",
    market="INDIA",
    currency="INR",
    data_source="UPSTOX",
    instrument_key=None,
),

"NIFTY_NEXT_50": Asset(
    symbol="NIFTY_NEXT_50",
    display_name="NIFTY Next 50",
    asset_class="INDEX",
    market="INDIA",
    currency="INR",
    data_source="UPSTOX",
    instrument_key=None,
),

    # ==========================================================
    # UNITED STATES
    # ==========================================================

    "NVDA": Asset(
        symbol="NVDA",
        display_name="NVIDIA",
        asset_class="EQUITY",
        market="US",
        currency="USD",
        data_source="YFINANCE",
    ),

    "AAPL": Asset(
        symbol="AAPL",
        display_name="Apple",
        asset_class="EQUITY",
        market="US",
        currency="USD",
        data_source="YFINANCE",
    ),

    "MSFT": Asset(
        symbol="MSFT",
        display_name="Microsoft",
        asset_class="EQUITY",
        market="US",
        currency="USD",
        data_source="YFINANCE",
    ),

    "GOOGL": Asset(
        symbol="GOOGL",
        display_name="Alphabet",
        asset_class="EQUITY",
        market="US",
        currency="USD",
        data_source="YFINANCE",
    ),

    "AMZN": Asset(
        symbol="AMZN",
        display_name="Amazon",
        asset_class="EQUITY",
        market="US",
        currency="USD",
        data_source="YFINANCE",
    ),

    "META": Asset(
        symbol="META",
        display_name="Meta Platforms",
        asset_class="EQUITY",
        market="US",
        currency="USD",
        data_source="YFINANCE",
    ),

    "AVGO": Asset(
        symbol="AVGO",
        display_name="Broadcom",
        asset_class="EQUITY",
        market="US",
        currency="USD",
        data_source="YFINANCE",
    ),

    "TSLA": Asset(
        symbol="TSLA",
        display_name="Tesla",
        asset_class="EQUITY",
        market="US",
        currency="USD",
        data_source="YFINANCE",
    ),

    "PLTR": Asset(
        symbol="PLTR",
        display_name="Palantir",
        asset_class="EQUITY",
        market="US",
        currency="USD",
        data_source="YFINANCE",
    ),

    "NFLX": Asset(
        symbol="NFLX",
        display_name="Netflix",
        asset_class="EQUITY",
        market="US",
        currency="USD",
        data_source="YFINANCE",
    ),
}


def get_asset(symbol: str) -> Asset:
    symbol = symbol.upper().strip()

    if symbol not in QUANTOS_ASSETS:
        raise KeyError(f"Unknown QuantOS asset: {symbol}")

    return QUANTOS_ASSETS[symbol]


def list_quantos_assets():
    return list(QUANTOS_ASSETS.values())


def register_custom_asset(
    symbol: str,
    display_name: str,
    asset_class: str,
    market: str,
    currency: str,
    data_source: str = "USER",
    instrument_key: Optional[str] = None,
) -> Asset:

    asset = Asset(
        symbol=symbol.upper().strip(),
        display_name=display_name,
        asset_class=asset_class,
        market=market,
        currency=currency,
        data_source=data_source,
        instrument_key=instrument_key,
    )

    QUANTOS_ASSETS[asset.symbol] = asset

    return asset