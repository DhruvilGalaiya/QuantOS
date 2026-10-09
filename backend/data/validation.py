import pandas as pd


def validate_market_data(data: pd.DataFrame) -> dict:

    issues = []

    # Required columns
    required_columns = [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "source"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        issues.append(
            f"Missing columns: {missing_columns}"
        )

    if issues:
        return {
            "valid": False,
            "issues": issues
        }

    # Missing values
    null_counts = data[
        [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close"
        ]
    ].isnull().sum()

    if null_counts.any():
        issues.append(
            f"Missing required values: "
            f"{null_counts[null_counts > 0].to_dict()}"
        )

    # Duplicate observations
    duplicates = data.duplicated(
        subset=["symbol", "timestamp"]
    ).sum()

    if duplicates > 0:
        issues.append(
            f"Found {duplicates} duplicate observations"
        )

    # Negative prices
    price_columns = [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close"
    ]

    for column in price_columns:

        if (data[column] < 0).any():
            issues.append(
                f"Negative values found in {column}"
            )

    # OHLC consistency
    invalid_high_low = (
        data["high"] < data["low"]
    ).sum()

    if invalid_high_low > 0:
        issues.append(
            f"{invalid_high_low} rows have high < low"
        )

    invalid_open = (
        (data["open"] > data["high"]) |
        (data["open"] < data["low"])
    ).sum()

    if invalid_open > 0:
        issues.append(
            f"{invalid_open} rows have invalid open prices"
        )

    invalid_close = (
        (data["close"] > data["high"]) |
        (data["close"] < data["low"])
    ).sum()

    if invalid_close > 0:
        issues.append(
            f"{invalid_close} rows have invalid close prices"
        )

    # Timestamp ordering
    if not data["timestamp"].is_monotonic_increasing:
        issues.append(
            "Timestamps are not sorted chronologically"
        )

    # Volume validation
    # Negative volume is invalid.
    # Zero volume can occur for index-level data such as NIFTY 50,
    # so it is treated as a warning rather than a validation failure.

    negative_volume = (
        data["volume"] < 0
    ).sum()

    zero_volume = (
        data["volume"] == 0
    ).sum()

    if negative_volume > 0:
        issues.append(
            f"{negative_volume} rows have negative volume"
        )

    if zero_volume > 0:
        print(
            f"Warning: {zero_volume} rows have zero volume. "
            "This may be expected for index-level data."
        )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "rows_checked": len(data)
    }