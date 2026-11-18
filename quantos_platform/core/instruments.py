"""QuantOS instrument universe (Upstox instrument keys)."""

REGIME_INSTRUMENTS = {
    "NIFTY_50": "NSE_INDEX|Nifty 50",
    "NIFTY_BANK": "NSE_INDEX|Nifty Bank",
    "NIFTY_IT": "NSE_INDEX|Nifty IT",
    "NIFTY_PHARMA": "NSE_INDEX|Nifty Pharma",
    "NIFTY_AUTO": "NSE_INDEX|Nifty Auto",
    "NIFTY_FIN_SERVICE": "NSE_INDEX|Nifty Fin Service",
    "INDIA_VIX": "NSE_INDEX|India VIX",
    "NIFTY_MIDCAP_100": "NSE_INDEX|NIFTY MIDCAP 100",
    "NIFTY_NEXT_50": "NSE_INDEX|Nifty Next 50",
}

VALIDATION_INSTRUMENTS = {
    "SENSEX": "BSE_INDEX|SENSEX",
}

# Extra NSE sector / breadth indices used by the V11 next-day model.
# Kept separate so the legacy V10 scripts are unaffected.
BREADTH_INSTRUMENTS = {
    "NIFTY_METAL": "NSE_INDEX|Nifty Metal",
    "NIFTY_FMCG": "NSE_INDEX|Nifty FMCG",
    "NIFTY_REALTY": "NSE_INDEX|Nifty Realty",
    "NIFTY_ENERGY": "NSE_INDEX|Nifty Energy",
    "NIFTY_PSU_BANK": "NSE_INDEX|Nifty PSU Bank",
    "NIFTY_MEDIA": "NSE_INDEX|Nifty Media",
    "NIFTY_500": "NSE_INDEX|Nifty 500",
    "NIFTY_SMALLCAP_100": "NSE_INDEX|NIFTY SMLCAP 100",
}

ALL_INDEX_INSTRUMENTS = {**REGIME_INSTRUMENTS, **VALIDATION_INSTRUMENTS, **BREADTH_INSTRUMENTS}
