"""
Example Usage: Risk Parity Prototype

This script demonstrates how to:
1. Generate synthetic multi-asset returns
2. Run a rolling Risk Parity backtest
3. Analyze results with regime shift detection
4. Create visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the risk parity module
from Old_iterations.risk_parity import (
    rolling_rp_backtest,
    detect_weight_regime_shifts,
    plot_rp_weights_over_time,
    plot_risk_contributions,
    plot_regime_shifts,
    plot_portfolio_vol,
    print_backtest_summary
)


def generate_synthetic_returns(n_assets=5, n_days=1000, seed=42):
    """
    Generate synthetic multi-asset returns with realistic correlations.
    
    Parameters
    ----------
    n_assets : int
        Number of assets in the portfolio.
    n_days : int
        Number of trading days to simulate.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    returns_df : pd.DataFrame
        Synthetic returns indexed by date, columns as asset names.
    asset_names : list
        Names of the assets.
    """
    np.random.seed(seed)
    
    # Asset names
    asset_names = [
        'Equities',
        'Bonds',
        'Commodities',
        'Real Estate',
        'Credit'
    ][:n_assets]
    
    # Create a realistic correlation structure
    if n_assets == 5:
        # Correlation matrix for the 5 assets
        correlation_matrix = np.array([
            [1.00, -0.20, 0.10, 0.80, 0.70],    # Equities
            [-0.20, 1.00, -0.10, 0.10, -0.30],  # Bonds
            [0.10, -0.10, 1.00, 0.20, 0.30],    # Commodities
            [0.80, 0.10, 0.20, 1.00, 0.50],     # Real Estate
            [0.70, -0.30, 0.30, 0.50, 1.00],    # Credit
        ])
    else:
        # Identity matrix for other numbers of assets
        correlation_matrix = np.eye(n_assets)
    
    # Volatilities (annualized)
    if n_assets == 5:
        volatilities = np.array([0.15, 0.05, 0.20, 0.12, 0.08])
    else:
        volatilities = np.ones(n_assets) * 0.10
    
    # Convert correlation to covariance
    D = np.diag(volatilities)
    covariance_matrix = D @ correlation_matrix @ D
    
    # Ensure positive definiteness
    eigenvalues = np.linalg.eigvals(covariance_matrix)
    if np.any(eigenvalues <= 0):
        covariance_matrix = np.eye(n_assets) * 0.01  # Fallback
    
    # Generate returns (daily, so divide annual vol by sqrt(252))
    daily_returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=covariance_matrix / 252,
        size=n_days
    )
    
    # Create DataFrame
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')  # Business days
    returns_df = pd.DataFrame(
        daily_returns,
        index=dates,
        columns=asset_names
    )
    
    return returns_df, asset_names


def main():
    """Main example script."""
    
    print("\n" + "=" * 70)
    print("RISK PARITY PROTOTYPE - EXAMPLE BACKTEST")
    print("=" * 70)
    
    # ========================================================================
    # STEP 1: Generate Synthetic Returns
    # ========================================================================
    print("\n[1] Generating synthetic multi-asset returns...")
    returns_df, asset_names = generate_synthetic_returns(n_assets=5, n_days=1000)
    print(f"    Generated {len(returns_df)} daily returns for {len(asset_names)} assets")
    print(f"    Date range: {returns_df.index[0].strftime('%Y-%m-%d')} to {returns_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"\n    Sample returns (first 5 days):")
    print(returns_df.head())
    
    # ========================================================================
    # STEP 2: Run Rolling Risk Parity Backtest
    # ========================================================================
    print("\n[2] Running rolling Risk Parity backtest...")
    print("    Parameters: rebalance_freq='M' (monthly), lookback_window=252 (1 year)")
    
    weights_df, risk_contrib_df, portfolio_vol_df, rebalance_dates = rolling_rp_backtest(
        returns=returns_df,
        rebalance_freq='M',          # Monthly rebalancing
        lookback_window=252           # 1-year rolling window
    )
    
    print(f"    Completed {len(weights_df)} rebalances")
    print(f"\n    Latest Risk Parity Weights ({rebalance_dates[-1].strftime('%Y-%m-%d')}):")
    latest_weights = weights_df.iloc[-1]
    for asset, weight in latest_weights.items():
        print(f"      {asset:15s}: {weight:7.2%}")
    
    # ========================================================================
    # STEP 3: Compute Risk Contributions
    # ========================================================================
    print("\n[3] Risk Contribution Analysis...")
    latest_contrib = risk_contrib_df.iloc[-1]
    latest_cov = np.cov(returns_df.iloc[-252:].values.T)
    
    print(f"    Latest Risk Contributions ({rebalance_dates[-1].strftime('%Y-%m-%d')}):")
    total_contrib = latest_contrib.sum()
    for asset, contrib in latest_contrib.items():
        pct_of_total = contrib / total_contrib * 100 if total_contrib > 0 else 0
        print(f"      {asset:15s}: {contrib:7.4f} ({pct_of_total:5.1f}% of portfolio risk)")
    
    print(f"\n    Portfolio Volatility: {portfolio_vol_df.iloc[-1].values[0]:7.4f} (annualized)")
    
    # ========================================================================
    # STEP 4: Detect Regime Shifts
    # ========================================================================
    print("\n[4] Detecting Risk Parity regime shifts...")
    regime_shift_df, weight_changes_df = detect_weight_regime_shifts(
        weights_df=weights_df,
        lookback=20,
        threshold=0.05
    )
    
    num_shifts = regime_shift_df['regime_shift'].sum()
    print(f"    Number of regime shifts detected: {int(num_shifts)} out of {len(weights_df)}")
    print(f"    Shift frequency: {num_shifts / len(weights_df) * 100:.1f}%")
    
    # ========================================================================
    # STEP 5: Print Summary Statistics
    # ========================================================================
    print("\n[5] Backtest Summary Statistics")
    print_backtest_summary(weights_df, risk_contrib_df, portfolio_vol_df)
    
    # ========================================================================
    # STEP 6: Create Visualizations
    # ========================================================================
    print("\n[6] Creating visualizations...")
    
    # Plot 1: Weights over time
    fig1, ax1 = plot_rp_weights_over_time(
        weights_df,
        title='Risk Parity Weights Over Time (Monthly Rebalancing)'
    )
    plt.savefig('rp_weights_over_time.png', dpi=300, bbox_inches='tight')
    print("    ✓ Saved: rp_weights_over_time.png")
    
    # Plot 2: Risk contributions
    fig2, ax2 = plot_risk_contributions(
        weights_df,
        risk_contrib_df,
        title='Risk Contributions vs Portfolio Weights'
    )
    plt.savefig('rp_risk_contributions.png', dpi=300, bbox_inches='tight')
    print("    ✓ Saved: rp_risk_contributions.png")
    
    # Plot 3: Regime shifts
    fig3, ax3 = plot_regime_shifts(
        regime_shift_df,
        weights_df,
        figsize=(12, 8)
    )
    plt.savefig('rp_regime_shifts.png', dpi=300, bbox_inches='tight')
    print("    ✓ Saved: rp_regime_shifts.png")
    
    # Plot 4: Portfolio volatility
    fig4, ax4 = plot_portfolio_vol(portfolio_vol_df)
    plt.savefig('rp_portfolio_vol.png', dpi=300, bbox_inches='tight')
    print("    ✓ Saved: rp_portfolio_vol.png")
    
    print("\n" + "=" * 70)
    print("EXAMPLE BACKTEST COMPLETE")
    print("=" * 70)
    print("\nYou can now:")
    print("  • Swap in your own returns data (pandas DataFrame)")
    print("  • Modify rebalancing frequency and lookback window")
    print("  • Extend with EWMA or shrinkage covariance estimators")
    print("  • Add your own analysis and visualizations")
    print("\nKey output files (DataFrames):")
    print("  • weights_df: Time series of Risk Parity weights")
    print("  • risk_contrib_df: Time series of risk contributions")
    print("  • portfolio_vol_df: Portfolio volatility over time")
    print("=" * 70 + "\n")
    
    # Return objects for interactive exploration
    return {
        'returns_df': returns_df,
        'weights_df': weights_df,
        'risk_contrib_df': risk_contrib_df,
        'portfolio_vol_df': portfolio_vol_df,
        'regime_shift_df': regime_shift_df,
        'rebalance_dates': rebalance_dates,
    }


if __name__ == '__main__':
    results = main()
