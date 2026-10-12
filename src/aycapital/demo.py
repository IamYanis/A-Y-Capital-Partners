"""Run a research demonstration; this is not an investment recommendation."""

import argparse
import json
import tomllib
from pathlib import Path

import pandas as pd

from aycapital.covariance import sample_covariance, shrinkage_covariance
from aycapital.data import load_yahoo, synthetic_prices, to_gbp
from aycapital.portfolio import Constraints, optimize
from aycapital.statistics import price_returns, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/demo.toml"))
    parser.add_argument("--source", choices=["synthetic", "yahoo"])
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    data = config["data"]
    source = args.source or data["source"]
    if source not in {"synthetic", "yahoo"}:
        raise ValueError("Unsupported data source")
    if source == "synthetic":
        prices = synthetic_prices(data["tickers"])
        provenance = {"source": "synthetic", "seed": 2026, "currency": "GBP illustration"}
    else:
        prices = load_yahoo(data["tickers"], data["start"], data["end"], data["cache_dir"])
        provenance = dict(prices.attrs["source"])
        factors = data.get("gbp_per_quote_unit")
        if factors is None or len(factors) != len(prices.columns):
            raise ValueError("Provide verified GBP conversion for every example asset")
        rates = pd.DataFrame([factors] * len(prices), index=prices.index, columns=prices.columns)
        prices = to_gbp(prices, rates)
        provenance["gbp_per_quote_unit"] = factors
    returns = price_returns(prices)
    research = config["research"]
    annual = research["periods_per_year"]
    statistics = summary(returns, annual, research["risk_free_rate"])
    sample = sample_covariance(returns, annual)
    shrinkage = shrinkage_covariance(returns, annual)
    settings = config["portfolio"]
    groups = {name: (tuple(group["assets"]), group["max_weight"])
              for name, group in settings.get("exposure_caps", {}).items()}
    competition = config["competition"]
    eligible = competition.get("eligible_assets")
    if competition["rules_configured"] and not eligible:
        raise ValueError("Configured competition rules require eligible_assets")
    weights = optimize(shrinkage, Constraints(settings["max_weight"],
                       tuple(eligible) if eligible is not None else None, groups))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, result in {"returns": returns, "statistics": statistics,
                         "sample_covariance": sample, "shrinkage_covariance": shrinkage,
                         "weights": weights}.items():
        result.to_csv(args.output / f"{name}.csv")
    (args.output / "metadata.json").write_text(json.dumps({
        "provenance": provenance, "configuration": config,
        "interpretation": "Historical/synthetic research only; weights are not forecasts or trades"
    }, indent=2) + "\n")
    print(f"{source.upper()} RESEARCH DEMO — example universe, not competition eligibility")
    print("Competition rules configured:", competition["rules_configured"])
    print("Simple returns (first five rows):\n", returns.head())
    print("Historical/synthetic statistics:\n", statistics)
    print("Annualized sample covariance:\n", sample)
    print("Annualized Ledoit–Wolf covariance:\n", shrinkage)
    print("Illustrative minimum-variance weights:\n", weights)
    print("Artifacts:", args.output)


if __name__ == "__main__":
    main()
