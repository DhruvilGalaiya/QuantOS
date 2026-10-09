from src.config import REGIME_INSTRUMENTS, VALIDATION_INSTRUMENTS
from src.historical import fetch_daily_data, save_daily_data


ALL_INSTRUMENTS = {
    **REGIME_INSTRUMENTS,
    **VALIDATION_INSTRUMENTS,
}


FROM_DATE = "2022-01-01"
TO_DATE = "2026-09-04"


for name, instrument_key in ALL_INSTRUMENTS.items():

    df = fetch_daily_data(
        instrument_name=name,
        instrument_key=instrument_key,
        from_date=FROM_DATE,
        to_date=TO_DATE,
    )

    save_daily_data(
        df=df,
        instrument_name=name,
    )

    print(
        f"{name}: "
        f"{len(df)} rows | "
        f"{df['timestamp'].min().date()} → "
        f"{df['timestamp'].max().date()}"
    )