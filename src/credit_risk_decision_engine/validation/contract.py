import pandera.pandas as pa
from pandera import Column, Check


def home_credit_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            # Identifiers
            "SK_ID_CURR": Column(
                int,
                Check.greater_than(0),
                nullable=False,
                unique=True,
            ),

            # Target
            "TARGET": Column(
                int,
                Check.isin([0, 1]),
                nullable=False,
            ),

            # Core numeric features
            "AMT_INCOME_TOTAL": Column(
                float,
                Check.in_range(0, 200_000_000),
                nullable=True,
            ),
            "AMT_CREDIT": Column(
                float,
                Check.in_range(0, 5_000_000),
                nullable=True,
            ),
            "DAYS_BIRTH": Column(
                int,
                Check.in_range(-30_000, -5_000),
                nullable=True,
            ),
            "DAYS_EMPLOYED": Column(
                int,
                Check.greater_than(-50_000),
                nullable=True,
            ),

            # External scores (allow missing, but bounded)
            "EXT_SOURCE_1": Column(
                float,
                Check.in_range(0, 1),
                nullable=True,
            ),
            "EXT_SOURCE_2": Column(
                float,
                Check.in_range(0, 1),
                nullable=True,
            ),
            "EXT_SOURCE_3": Column(
                float,
                Check.in_range(0, 1),
                nullable=True,
            ),

            # Binary flags
            "FLAG_OWN_CAR": Column(
                object,
                Check.isin(["Y", "N"]),
                nullable=True,
            ),
            "FLAG_OWN_REALTY": Column(
                object,
                Check.isin(["Y", "N"]),
                nullable=True,
            ),
        },

        # Allow extra columns (we have 122 total)
        strict=False,

        # Dataset-level checks
        checks=[
            # Target must not be entirely missing
            Check(
                lambda df: df["TARGET"].notna().mean() > 0.99,
                error="TARGET has excessive missing values",
            ),

            # Core income feature should be mostly present
            Check(
                lambda df: df["AMT_INCOME_TOTAL"].notna().mean() > 0.70,
                error="AMT_INCOME_TOTAL missingness too high",
            ),
        ],
    )
