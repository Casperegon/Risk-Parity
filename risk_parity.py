"""
Risk Parity Portfolio Model - Prototype v1
A modular implementation for risk-balanced portfolio construction and monitoring.

This module provides tools to:
- Estimate covariance matrices from historical returns
- Compute long-only Risk Parity weights
- Calculate risk contributions
- Run rolling backtests
- Detect risk regime shifts
- Visualize results

Usage:
    import pandas as pd
    from risk_parity import rolling_rp_backtest
    
    returns = pd.read_csv('returns.csv', index_col=0, parse_dates=True)
    weights_df, risk_contrib_df = rolling_rp_backtest(
        returns=returns,
        rebalance_freq='M',
        lookback_window=252
    )
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# SECTION 1: COVARIANCE ESTIMATION
# ============================================================================

def estimate_rolling_covariance(returns, window):
    """
    Estimate rolling sample covariance matrix from returns.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Historical asset returns with shape (T, N) where T is time periods
        and N is number of assets. Rows are dates, columns are assets.
    window : int
        Rolling window size in number of observations.
    
    Returns
    -------
    dict with keys:
        'dates': pd.DatetimeIndex - Dates at which covariance is estimated
        'cov_matrices': list of np.ndarray - Covariance matrices indexed by date
        'cov_dict': dict with dates as keys, cov matrices as values (for easy access)
    
    Notes
    -----
    Uses simple rolling sample estimator: cov = X'X / (T-1) where X is
    demeaned returns over the window.
    
    Can be replaced with EWMA, shrinkage, or other estimators by following
    the same output structure.
    """
    if window > len(returns):
        raise ValueError(f"Window size ({window}) cannot exceed data length ({len(returns)})")
    
    dates = []
    cov_matrices = []
    
    # Rolling window calculation
    for i in range(window, len(returns) + 1):
        window_returns = returns.iloc[i - window:i].values
        cov_matrix = np.cov(window_returns.T)
        
        dates.append(returns.index[i - 1])
        cov_matrices.append(cov_matrix)
    
    # Create dictionary for easy lookup by date
    cov_dict = {date: cov for date, cov in zip(dates, cov_matrices)}
    
    return {
        'dates': pd.DatetimeIndex(dates),
        'cov_matrices': cov_matrices,
        'cov_dict': cov_dict
    }


# ============================================================================
# SECTION 2: RISK PARITY OPTIMIZATION
# ============================================================================

def compute_rp_weights(cov_matrix, long_only=True, verbose=False):
    """
    Compute long-only Risk Parity weights from a covariance matrix.
    
    Risk Parity aims to find weights such that each asset contributes equally
    to total portfolio risk. The optimization minimizes the sum of squared
    differences between risk contributions and target (equal) contribution.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix of asset returns with shape (N, N).
    long_only : bool, default True
        If True, weights are constrained to be non-negative (sum to 1).
        If False, weights can be negative (short selling allowed).
    verbose : bool, default False
        If True, print optimization warnings/info.
    
    Returns
    -------
    weights : np.ndarray
        Risk Parity weights with shape (N,) that sum to 1.
    success : bool
        Whether optimization converged successfully.
    
    Notes
    -----
    Uses scipy.optimize.minimize with SLSQP method.
    Initial guess is inverse volatility (proportional to 1/std).
    """
    n = cov_matrix.shape[0]

    def objective(w):
        portfolio_var = w @ cov_matrix @ w  # Porteføljens varians: w' * Σ * w
        if portfolio_var < 1e-10:  # Undgå division med nul hvis varians er negligibel
            return 1e10

        mrc = cov_matrix @ w  # Marginal risk contribution: Σ * w (gradient af varians mht. w)
        rc = w * mrc / portfolio_var  # Relativ RC: w_i * (Σw)_i / (w'Σw) — summer til 1 per konstruktion
        target = 1.0 / n  # Målværdi: hvert asset skal bidrage ligeligt = 1/n
        return np.sum((rc - target) ** 2)  # Minimer summen af kvadrerede afvigelser fra målet: Σ(RC_i - 1/n)²
    
    # Starting point: inverse volatility weighting (often close to RP)
    diag_cov = np.sqrt(np.diag(cov_matrix))
    w0 = 1.0 / diag_cov
    w0 = w0 / np.sum(w0)  # Normalize to sum to 1
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Sum to 1
    
    if long_only:
        bounds = [(0, 1) for _ in range(n)]  # Non-negative, at most 100%
    else:
        bounds = [(None, None) for _ in range(n)]  # Unrestricted
    
    # Optimize
    with warnings.catch_warnings():
        if not verbose:
            warnings.filterwarnings('ignore')
        
        result = minimize(
            objective,
            x0=w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
    return result.x, result.success


# ============================================================================
# SECTION 3: RISK ANALYSIS
# ============================================================================

def compute_risk_contributions(weights, cov_matrix):
    """
    Compute the risk contribution of each asset to total portfolio risk.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights with shape (N,).
    cov_matrix : np.ndarray
        Covariance matrix with shape (N, N).
    
    Returns
    -------
    risk_contrib : np.ndarray
        Risk contribution of each asset, shape (N,).
    portfolio_vol : float
        Total portfolio volatility.
    
    Notes
    -----
    Risk contribution = weight_i * (Sigma @ weights)_i / portfolio_vol
    
    where (Sigma @ weights)_i is the marginal risk contribution.
    These sum to total portfolio volatility.
    """
    portfolio_vol = compute_portfolio_volatility(weights, cov_matrix)
    portfolio_var = portfolio_vol ** 2

    if portfolio_vol < 1e-10:
        return np.zeros_like(weights), portfolio_vol
    
    # Marginal risk contributions
    mrc = cov_matrix @ weights
    
    # Risk contributions
    risk_contrib = weights * mrc / portfolio_var
    
    return risk_contrib, portfolio_vol


def compute_portfolio_volatility(weights, cov_matrix):
    """
    Compute portfolio volatility (standard deviation).
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights with shape (N,).
    cov_matrix : np.ndarray
        Covariance matrix with shape (N, N).
    
    Returns
    -------
    volatility : float
        Portfolio standard deviation (annualized if cov_matrix is annualized).
    """
    variance = weights @ cov_matrix @ weights
    return np.sqrt(max(variance, 0))


# ============================================================================
# SECTION 4: ROLLING BACKTEST ENGINE
# ============================================================================

def rolling_rp_backtest(returns, rebalance_freq='M', lookback_window=252):
    """
    Run a rolling Risk Parity backtest on historical returns.
    
    This is the main backtest function that:
    1. Identifies rebalance dates based on frequency
    2. Estimates covariance at each rebalance date
    3. Computes Risk Parity weights
    4. Calculates risk contributions
    5. Stores results in time series DataFrames
    
    Parameters
    ----------
    returns : pd.DataFrame
        Historical asset returns with shape (T, N).
        Rows = dates (in order), Columns = asset classes.
        Index must be a DatetimeIndex.
    rebalance_freq : str, default 'M'
        Rebalancing frequency for pandas date_range.
        'D' = daily, 'W' = weekly, 'M' = monthly, 'Q' = quarterly, 'Y' = yearly
    lookback_window : int, default 252
        Number of trading days for rolling covariance estimation.
    
    Returns
    -------
    weights_df : pd.DataFrame
        Risk Parity weights over time. Index = rebalance dates, columns = assets.
    risk_contrib_df : pd.DataFrame
        Risk contributions over time. Index = rebalance dates, columns = assets.
    portfolio_vol_df : pd.DataFrame
        Portfolio volatility over time. Index = rebalance dates.
    rebalance_dates : pd.DatetimeIndex
        The actual rebalance dates used (aligned with available data).
    
    Raises
    ------
    ValueError
        If returns DataFrame is empty or lookback_window is too small.
    """
    if returns.empty or len(returns) < lookback_window:
        raise ValueError(f"Returns must have at least {lookback_window} observations")
    
    # Generate candidate rebalance dates
    candidate_dates = pd.date_range(
        start=returns.index[lookback_window - 1],
        end=returns.index[-1],
        freq=rebalance_freq
    )
    
    # Find actual rebalance dates (dates that exist in returns)
    rebalance_dates = [d for d in candidate_dates if d in returns.index]
    
    if not rebalance_dates:
        raise ValueError(f"No valid rebalance dates found for frequency '{rebalance_freq}'")
    
    # Initialize storage
    weights_list = []
    risk_contrib_list = []
    portfolio_vol_list = []
    
    asset_names = returns.columns.tolist()
    
    # Loop through rebalance dates
    for rebal_date in rebalance_dates:
        # Get data up to rebalance date
        data_idx = returns.index.get_loc(rebal_date)
        start_idx = max(0, data_idx - lookback_window + 1)
        
        window_returns = returns.iloc[start_idx:data_idx + 1]
        
        # Estimate covariance (simple rolling sample)
        cov_matrix = np.cov(window_returns.T)*252
        
        # Compute RP weights
        weights, success = compute_rp_weights(cov_matrix, long_only=True)
        
        if not success:
            warnings.warn(
                f"RP optimization did not converge at {rebal_date}. "
                f"Using inverse volatility as fallback."
            )
            # Fallback: inverse volatility
            diag_cov = np.sqrt(np.diag(cov_matrix))
            weights = 1.0 / diag_cov
            weights = weights / np.sum(weights)
        
        # Compute risk contributions and portfolio vol
        risk_contrib, portfolio_vol = compute_risk_contributions(weights, cov_matrix)
        
        weights_list.append(weights)
        risk_contrib_list.append(risk_contrib)
        portfolio_vol_list.append(portfolio_vol)
    
    # Build DataFrames
    weights_df = pd.DataFrame(
        weights_list,
        index=pd.DatetimeIndex(rebalance_dates),
        columns=asset_names
    )
    
    risk_contrib_df = pd.DataFrame(
        risk_contrib_list,
        index=pd.DatetimeIndex(rebalance_dates),
        columns=asset_names
    )
    
    portfolio_vol_df = pd.DataFrame(
        portfolio_vol_list,
        index=pd.DatetimeIndex(rebalance_dates),
        columns=['Portfolio Vol']
    )
    
    return weights_df, risk_contrib_df, portfolio_vol_df, pd.DatetimeIndex(rebalance_dates)


# ============================================================================
# SECTION 5: REGIME DETECTION
# ============================================================================

def detect_weight_regime_shifts(weights_df, lookback=20, threshold=0.05):
    """
    Detect significant changes in Risk Parity weights over time.
    
    This is a simple regime shift indicator based on the magnitude of
    changes in portfolio weights. Can indicate structural changes in
    risk relationships (e.g., correlations shifting).
    
    Parameters
    ----------
    weights_df : pd.DataFrame
        Risk Parity weights over time from rolling_rp_backtest.
    lookback : int, default 20
        Number of periods to look back for change detection.
    threshold : float, default 0.05
        Threshold for flagging a significant regime shift (as a fraction
        of total weight change).
    
    Returns
    -------
    regime_shift_df : pd.DataFrame
        Binary indicator (0 or 1) for regime shifts. Index = dates.
        Columns = ['regime_shift'] (1 = shift detected, 0 = no shift).
    weight_change_df : pd.DataFrame
        Magnitude of weight changes. Index = dates, columns = assets.
    
    Notes
    -----
    Regime shift = mean absolute change in weights > threshold.
    """
    # Compute absolute changes in weights
    weight_changes = weights_df.diff().abs()
    
    # Rolling mean of absolute changes
    mean_change = weight_changes.rolling(window=lookback).mean()
    
    # Compute magnitude of regime shift indicator
    # Sum across assets to get total change per period
    total_change = weight_changes.sum(axis=1)
    
    # Simple indicator: flag if total change > threshold
    regime_shift = (total_change > threshold).astype(int)
    
    regime_shift_df = pd.DataFrame(
        regime_shift,
        columns=['regime_shift']
    )
    
    return regime_shift_df, weight_changes


# ============================================================================
# SECTION 6: VISUALIZATION
# ============================================================================

def plot_rp_weights_over_time(weights_df, figsize=(12, 6), title='Risk Parity Weights Over Time'):
    """
    Plot Risk Parity weights as a time series (stacked area chart).
    
    Parameters
    ----------
    weights_df : pd.DataFrame
        Risk Parity weights from rolling_rp_backtest, indexed by date.
    figsize : tuple, default (12, 6)
        Figure size for matplotlib.
    title : str
        Title for the plot.
    
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    weights_df.plot(
        kind='area',
        stacked=True,
        ax=ax,
        alpha=0.7
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weight')
    ax.set_ylim([0, 1])
    ax.legend(title='Assets', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax


def plot_risk_contributions(weights_df, risk_contrib_df, figsize=(10, 6), 
                           title='Risk Contributions (Latest Date)'):
    """
    Plot risk contributions as a bar chart (latest date).
    
    Parameters
    ----------
    weights_df : pd.DataFrame
        Risk Parity weights (for reference).
    risk_contrib_df : pd.DataFrame
        Risk contributions from rolling_rp_backtest.
    figsize : tuple, default (10, 6)
        Figure size.
    title : str
        Title for the plot.
    
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    # Get latest row
    latest_contrib = risk_contrib_df.iloc[-1]
    latest_weights = weights_df.iloc[-1]
    latest_date = risk_contrib_df.index[-1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Risk contributions
    latest_contrib.plot(kind='bar', ax=ax1, color='steelblue', alpha=0.7)
    ax1.set_title(f'Risk Contributions as of {latest_date.strftime("%Y-%m-%d")}', 
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel('Risk Contribution')
    ax1.set_xlabel('Asset')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=latest_contrib.mean(), color='red', linestyle='--', 
                label=f'Mean: {latest_contrib.mean():.4f}', linewidth=2)
    ax1.legend()
    
    # Weights
    latest_weights.plot(kind='bar', ax=ax2, color='coral', alpha=0.7)
    ax2.set_title(f'Portfolio Weights as of {latest_date.strftime("%Y-%m-%d")}', 
                  fontsize=12, fontweight='bold')
    ax2.set_ylabel('Weight')
    ax2.set_xlabel('Asset')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, latest_weights.max() * 1.1])
    
    plt.tight_layout()
    
    return fig, (ax1, ax2)


def plot_regime_shifts(regime_shift_df, weights_df, figsize=(12, 8)):
    """
    Plot regime shift indicator as a time series with weights overlay.
    
    Parameters
    ----------
    regime_shift_df : pd.DataFrame
        Regime shift indicator from detect_weight_regime_shifts.
    weights_df : pd.DataFrame
        Risk Parity weights for reference.
    figsize : tuple, default (12, 8)
        Figure size.
    
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot regime shifts
    ax1.fill_between(regime_shift_df.index, 0, regime_shift_df['regime_shift'],
                     alpha=0.3, color='red', label='Regime Shift Detected')
    ax1.plot(regime_shift_df.index, regime_shift_df['regime_shift'],
             color='red', linewidth=2, drawstyle='steps-post')
    ax1.set_ylabel('Regime Shift Indicator')
    ax1.set_title('Risk Parity Regime Shift Detection', fontsize=14, fontweight='bold')
    ax1.set_ylim([-0.1, 1.1])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot weights
    weights_df.plot(ax=ax2, alpha=0.7)
    ax2.set_ylabel('Weight')
    ax2.set_xlabel('Date')
    ax2.set_title('Portfolio Weights Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(title='Assets', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    return fig, (ax1, ax2)


def plot_portfolio_vol(portfolio_vol_df, figsize=(12, 5)):
    """
    Plot portfolio volatility over time.
    
    Parameters
    ----------
    portfolio_vol_df : pd.DataFrame
        Portfolio volatility from rolling_rp_backtest.
    figsize : tuple, default (12, 5)
        Figure size.
    
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    portfolio_vol_df.plot(ax=ax, linewidth=2, legend=False, color='darkblue')
    
    ax.set_title('Risk Parity Portfolio Volatility Over Time', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Volatility (Annualized)')
    ax.grid(True, alpha=0.3)
    ax.fill_between(portfolio_vol_df.index, portfolio_vol_df['Portfolio Vol'],
                    alpha=0.2, color='darkblue')
    
    plt.tight_layout()
    
    return fig, ax


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_backtest_summary(weights_df, risk_contrib_df, portfolio_vol_df):
    """
    Print a summary of backtest results.
    
    Parameters
    ----------
    weights_df : pd.DataFrame
        Weights from rolling_rp_backtest.
    risk_contrib_df : pd.DataFrame
        Risk contributions from rolling_rp_backtest.
    portfolio_vol_df : pd.DataFrame
        Portfolio volatility from rolling_rp_backtest.
    """
    print("=" * 70)
    print("RISK PARITY BACKTEST SUMMARY")
    print("=" * 70)
    print(f"\nBacktest Period: {weights_df.index[0].strftime('%Y-%m-%d')} to {weights_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Number of Rebalances: {len(weights_df)}")
    print(f"Number of Assets: {len(weights_df.columns)}")
    
    print("\n--- Latest Weights ---")
    latest_weights = weights_df.iloc[-1]
    for asset, weight in latest_weights.items():
        print(f"  {asset:15s}: {weight:7.2%}")
    
    print("\n--- Latest Risk Contributions ---")
    latest_contrib = risk_contrib_df.iloc[-1]
    for asset, contrib in latest_contrib.items():
        pct_of_vol = contrib / risk_contrib_df.iloc[-1].sum() * 100
        print(f"  {asset:15s}: {contrib:7.4f} ({pct_of_vol:5.1f}% of vol)")
    
    print("\n--- Portfolio Volatility Statistics ---")
    vol_series = portfolio_vol_df['Portfolio Vol']
    print(f"  Mean Vol:       {vol_series.mean():7.4f}")
    print(f"  Min Vol:        {vol_series.min():7.4f}")
    print(f"  Max Vol:        {vol_series.max():7.4f}")
    print(f"  Std Dev Vol:    {vol_series.std():7.4f}")
    
    print("\n--- Weight Statistics ---")
    print("Mean Weight by Asset:")
    for asset, mean_weight in weights_df.mean().items():
        print(f"  {asset:15s}: {mean_weight:7.2%}")
    
    print("\nWeight Range (Min - Max) by Asset:")
    for asset in weights_df.columns:
        min_w = weights_df[asset].min()
        max_w = weights_df[asset].max()
        print(f"  {asset:15s}: {min_w:7.2%} - {max_w:7.2%}")
    
    print("\n" + "=" * 70)
