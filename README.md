# Risk Parity Portfolio Model - Prototype v1

A clean, modular Python implementation of a Risk Parity portfolio model for multi-asset portfolios.

## Overview

Risk Parity is a portfolio construction approach that allocates to assets based on **balancing risk contributions** rather than capital weights. This prototype provides a proof-of-concept for:

- Computing long-only Risk Parity weights from a covariance matrix
- Calculating each asset's contribution to total portfolio risk
- Running rolling backtests to observe how RP weights evolve over time
- Detecting significant shifts in portfolio structure
- Visualizing key metrics and diagnostics

## Key Features

✓ **Modular Design** - Functions are cleanly organized by purpose, making it easy to extend  
✓ **Flexible Covariance Estimation** - Simple rolling sample estimator; easy to swap in EWMA or shrinkage  
✓ **Risk Analysis** - Computes marginal and total risk contributions for each asset  
✓ **Backtesting Engine** - Rolling backtest with customizable rebalancing frequency  
✓ **Regime Detection** - Simple indicator to flag structural changes in weights  
✓ **Publication-Ready Plots** - Time series, risk decompositions, and regime shifts  
✓ **Production-Ready Code** - Type hints, docstrings, error handling, and logging  

## File Structure

```
.
├── risk_parity.py          # Main module with all functions
├── example_backtest.py     # Example usage with synthetic data
└── README.md               # This file
```

## Installation & Requirements

```bash
pip install numpy pandas scipy matplotlib
```

**Python version**: 3.8+

## Quick Start

### 1. Using Your Own Data

```python
import pandas as pd
from risk_parity import rolling_rp_backtest

# Load your returns (DataFrame with dates as index, assets as columns)
returns = pd.read_csv('your_returns.csv', index_col=0, parse_dates=True)

# Run backtest
weights_df, risk_contrib_df, portfolio_vol_df, rebalance_dates = rolling_rp_backtest(
    returns=returns,
    rebalance_freq='M',      # Monthly rebalancing
    lookback_window=252      # 1-year rolling window
)

# Analyze results
print(weights_df.iloc[-1])  # Latest weights
print(risk_contrib_df.iloc[-1])  # Latest risk contributions
```

### 2. Run the Example

```bash
python example_backtest.py
```

This generates synthetic 5-asset returns and creates visualizations showing:
- Weight evolution over time
- Risk contributions at the latest date
- Regime shift detection
- Portfolio volatility time series

## Module Documentation

### Core Functions

#### Covariance Estimation

**`estimate_rolling_covariance(returns, window)`**
- Estimates rolling sample covariance from returns
- Returns dict with covariance matrices indexed by date
- Can be replaced with EWMA, shrinkage, or other estimators

#### Risk Parity Optimization

**`compute_rp_weights(cov_matrix, long_only=True)`**
- Solves for weights such that each asset contributes equally to portfolio risk
- Uses scipy.optimize.minimize (SLSQP method)
- Constraints: weights ≥ 0, sum to 1 (if long_only=True)
- Returns weights and convergence flag
- Initial guess: inverse volatility weighting (often close to RP solution)

#### Risk Analysis

**`compute_risk_contributions(weights, cov_matrix)`**
- Calculates each asset's contribution to total portfolio risk
- Returns risk contributions (which sum to portfolio volatility) and portfolio vol
- Useful for understanding portfolio structure

**`compute_portfolio_volatility(weights, cov_matrix)`**
- Simple utility: portfolio std dev from weights and covariance

#### Rolling Backtest

**`rolling_rp_backtest(returns, rebalance_freq='M', lookback_window=252)`**
- Main entry point: runs complete rolling backtest
- Rebalances at specified frequency ('D', 'W', 'M', 'Q', 'Y')
- Returns:
  - `weights_df`: RP weights indexed by rebalance date
  - `risk_contrib_df`: Risk contributions indexed by rebalance date
  - `portfolio_vol_df`: Portfolio volatility over time
  - `rebalance_dates`: Actual rebalance dates used

#### Regime Detection

**`detect_weight_regime_shifts(weights_df, lookback=20, threshold=0.05)`**
- Flags significant changes in portfolio composition
- Returns:
  - `regime_shift_df`: Binary indicator (1 = shift detected)
  - `weight_changes_df`: Magnitude of weight changes

#### Visualization

**`plot_rp_weights_over_time(weights_df)`**
- Stacked area chart of weights over time

**`plot_risk_contributions(weights_df, risk_contrib_df)`**
- Side-by-side bar charts: risk contributions vs weights (latest date)

**`plot_regime_shifts(regime_shift_df, weights_df)`**
- Dual plot: regime shift indicator + weight time series

**`plot_portfolio_vol(portfolio_vol_df)`**
- Time series of portfolio volatility

**`print_backtest_summary(weights_df, risk_contrib_df, portfolio_vol_df)`**
- Prints comprehensive summary statistics

## Extending the Code

### 1. Custom Covariance Estimator

To use EWMA or shrinkage instead of rolling sample covariance:

```python
from risk_parity import compute_rp_weights

def estimate_ewma_covariance(returns, lambda_param=0.94):
    """Your custom estimator"""
    # Implement EWMA logic
    # Return same structure as estimate_rolling_covariance
    pass

# Use in backtest
cov_result = estimate_ewma_covariance(returns)
cov_matrix = cov_result['cov_dict'][your_date]
weights, success = compute_rp_weights(cov_matrix)
```

### 2. Constrained Optimization

For leverage constraints, maximum weight limits, or sector constraints:

```python
from risk_parity import compute_rp_weights

# Modify compute_rp_weights() bounds or constraints
# Example: max 40% in any single asset
bounds = [(0, 0.4) for _ in range(n_assets)]
```

### 3. Risk Aversion or Utility Functions

To add a term that considers expected returns or risk aversion:

```python
def objective_with_return_preference(w, cov_matrix, expected_returns, risk_aversion=1.0):
    """Objective combining RP penalty with return preference"""
    # RP term (variance of risk contributions)
    ...
    # Return term
    ...
    return rp_penalty + risk_aversion * (-w @ expected_returns)
```

### 4. Add Turnover Constraints

To penalize high turnover in rebalancing:

```python
def objective_with_turnover(w, w_prev, cov_matrix, turnover_penalty=0.01):
    """Objective combining RP with turnover penalty"""
    rp_penalty = ...  # existing RP objective
    turnover_cost = turnover_penalty * np.sum(np.abs(w - w_prev))
    return rp_penalty + turnover_cost
```

## Model Assumptions & Limitations

**Assumptions:**
- Historical covariance is a good proxy for future relationships
- Risk (volatility) is the primary concern (no return expectations)
- Long-only portfolio (can modify to allow shorts)
- Transaction costs ignored
- Equal risk contribution is the desired outcome

**Current Limitations:**
- Simple rolling sample covariance (sample error for small windows/many assets)
- No turnover constraints
- No leverage constraints
- No factor risk decomposition
- No real-time data updates

**Future Enhancements:**
- Shrinkage or Ledoit-Wolf covariance estimator
- EWMA covariance
- Leverage and concentration limits
- Turnover minimization
- Factor-based analysis
- Performance attribution

## Example Output

```
======================================================================
RISK PARITY BACKTEST SUMMARY
======================================================================

Backtest Period: 2020-01-02 to 2023-12-29
Number of Rebalances: 50
Number of Assets: 5

--- Latest Weights ---
  Equities        :  16.81%
  Bonds           :  42.15%
  Commodities     :  13.24%
  Real Estate     :  17.28%
  Credit          :  10.52%

--- Latest Risk Contributions ---
  Equities        :  0.0240 (20.0% of vol)
  Bonds           :  0.0239 (19.9% of vol)
  Commodities     :  0.0241 (20.1% of vol)
  Real Estate     :  0.0240 (20.0% of vol)
  Credit          :  0.0240 (20.0% of vol)

--- Portfolio Volatility Statistics ---
  Mean Vol:              0.0892
  Min Vol:               0.0625
  Max Vol:               0.1204
  Std Dev Vol:           0.0165

--- Weight Statistics ---
Mean Weight by Asset:
  Equities        :  18.34%
  Bonds           :  38.92%
  Commodities     :  14.67%
  Real Estate     :  17.85%
  Credit          :  10.22%

Weight Range (Min - Max) by Asset:
  Equities        :  11.28% - 25.67%
  Bonds           :  28.45% - 52.33%
  Commodities     :   8.92% - 22.15%
  Real Estate     :  10.54% - 24.78%
  Credit          :   5.33% - 16.89%
======================================================================
```

## Tips for Practitioners

1. **Window Size**: Start with 252 days (1 year); adjust based on regime characteristics
2. **Rebalancing Frequency**: Monthly is typical; daily may be noisy, quarterly/annual smoother
3. **Risk Estimation**: Monitor covariance matrix condition; watch for near-singular matrices
4. **Diversification**: RP naturally diversifies when correlations are stable; monitor during stress
5. **Transitions**: Use weight changes and regime shifts to understand market structure changes
6. **Backtesting**: Always cross-check results with actual portfolio data if available

## References & Further Reading

- Roncalli, T. (2013). "Introduction to Risk Parity and Budgeting" (Chapman & Hall)
- Asness, C. S., Frazzini, A., & Pedersen, L. H. (2012). "Leverage aversion and risk parity" (FAJ)
- Maillard, S., Roncalli, T., & Teïletche, J. (2010). "The properties of equally weighted risk contribution portfolios" (JoD)

## License

This is a personal prototype for educational and research purposes.

---

**Last Updated**: March 2026  
**Version**: 1.0  
**Python**: 3.8+
