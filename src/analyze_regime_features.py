import pandas as pd
import numpy as np


INPUT_PATH = "data/regime/regime_features.parquet"


df = pd.read_parquet(INPUT_PATH)

feature_columns = [
    column
    for column in df.columns
    if column != "date"
]

X = df[feature_columns]


print("\nQuantOS Regime Feature Diagnostics")
print("=" * 75)

print("\nDataset")
print("-" * 75)

print("Rows:", len(df))
print("Features:", len(feature_columns))
print(
    "Date range:",
    df["date"].min().date(),
    "→",
    df["date"].max().date()
)


# ---------------------------------------------------------
# 1. Correlation matrix
# ---------------------------------------------------------

corr = X.corr()

print("\nHighly correlated feature pairs")
print("-" * 75)

found = False

for i in range(len(feature_columns)):

    for j in range(i + 1, len(feature_columns)):

        correlation = corr.iloc[i, j]

        if abs(correlation) >= 0.70:

            print(
                f"{feature_columns[i]:35s} "
                f"{feature_columns[j]:35s} "
                f"{correlation:+.3f}"
            )

            found = True


if not found:
    print("No feature pairs with |correlation| >= 0.70")


# ---------------------------------------------------------
# 2. Feature standard deviations
# ---------------------------------------------------------

print("\nFeature scales")
print("-" * 75)

scale_summary = (
    X.std()
    .sort_values(ascending=False)
)

for feature, std in scale_summary.items():

    print(
        f"{feature:35s} "
        f"std={std:.6f}"
    )


# ---------------------------------------------------------
# 3. Quantiles
# ---------------------------------------------------------

print("\nFeature quantiles")
print("-" * 75)

summary = X.describe(
    percentiles=[
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99,
    ]
).T

print(
    summary[
        [
            "mean",
            "std",
            "1%",
            "5%",
            "25%",
            "50%",
            "75%",
            "95%",
            "99%",
        ]
    ].round(5)
)