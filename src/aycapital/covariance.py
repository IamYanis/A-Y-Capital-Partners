"""Complete-case covariance estimators with consistent asset labels."""

import pandas as pd
from sklearn.covariance import LedoitWolf

from aycapital.validation import periods, returns


def sample_covariance(data: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    return returns(data).cov() * periods(periods_per_year)


def shrinkage_covariance(data: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    data = returns(data)
    estimate = LedoitWolf(assume_centered=False).fit(data.to_numpy()).covariance_
    return pd.DataFrame(estimate * periods(periods_per_year),
                        index=data.columns, columns=data.columns)
