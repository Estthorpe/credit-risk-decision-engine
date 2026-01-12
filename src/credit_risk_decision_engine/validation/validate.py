from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas  as pd
import pandera.pandas as pa

import credit_risk_decision_engine

from credit_risk_decision_engine.validation.contract import home_credit_schema

@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    error: Optional[str] = None 


def validate_training_df(df: pd.DataFrame) -> ValidationResult:
    """
    Validate a training dataframe against the Home Credit data contract.

    Returns:
        ValidationResult(passed=True) if ok,
        ValidationResult(passed=False, error=...) if not.
    """

    schema = home_credit_schema()
    try:
        schema.validate(df, lazy=True)
        return ValidationResult(passed=True, error=None)
    except pa.errors.SchemaErrors as e:
        # Pandera provides a failure_cases table for debugging
        return ValidationResult(passed=False, error=str(e))
    except Exception as e:
        return ValidationResult(passed=False, error=str(e))
    

def validate_or_raise(df: pd.DataFrame) -> None:
    """
    Convenience function for scripts/pipelines:
    raise immediately if contract fails
    """
    result = validate_training_df(df)
    if not result.passed:
        raise ValueError(f"Data contract validation failed: {result.error}")



