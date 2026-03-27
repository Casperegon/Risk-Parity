"""
Risk Parity Pipeline - Automated Data Collection, Cleaning & Optimization
===========================================================================

This script automates the complete Risk Parity workflow:
1. Connect to LSEG Datastream and fetch raw price data
2. Clean data and calculate returns
3. Run Risk Parity optimization
4. Save results to CSV files for validation

Runs on two datasets: Futures and GICS Sectors
No plots, no user input - fully automated batch execution.
"""

import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import keyring
import eikon as ek
from pydatastream import Datastream
from functools import partial

# Import Risk Parity functions
from risk_parity_New import (
    rolling_rp_backtest,
    sample_covariance,
    ewma_covariance,
    ledoit_wolf_covariance,
)

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_dir='logs'):
    """Set up logging with timestamps to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'rp_pipeline_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("RISK PARITY PIPELINE STARTED")
    logger.info("=" * 80)
    
    return logger, log_file


# ============================================================================
# DATASTREAM CONNECTION
# ============================================================================

def connect_to_datastream(logger):
    """Connect to LSEG Datastream."""
    try:
        logger.info("Connecting to LSEG Datastream...")
        
        # Set Eikon app key
        ek.set_app_key('035d2f1682244553ba7f239ce9e8d281142013dd')
        
        # Get credentials from Windows credential manager
        service = "Datastream"
        DS_USERNAME = "ZNYK749"
        DS_PASSWORD = keyring.get_password(service, DS_USERNAME)
        
        if not DS_PASSWORD:
            logger.error("ERROR: Could not retrieve Datastream password from credential manager")
            logger.error(f"Please ensure credentials for user '{DS_USERNAME}' are stored in Windows credential manager")
            raise Exception("Datastream credentials not found")
        
        DS = Datastream(username=DS_USERNAME, password=DS_PASSWORD)
        logger.info(f"✓ Connected to Datastream as user {DS_USERNAME}")
        
        return DS
    
    except Exception as e:
        logger.error(f"✗ Failed to connect to Datastream: {e}")
        raise

# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_and_rename_data(DS, symbol, column_name, start_date, end_date, logger):
    """Fetch GICS sector data from Datastream."""
    try:
        data = DS.get_price([symbol], date_from=start_date, date_to=end_date)
        data = data.rename(columns={'P': column_name})
        logger.debug(f"  Fetched {column_name}: {len(data)} rows")
        return data
    except Exception as e:
        logger.warning(f"  Warning: Failed to fetch {column_name} ({symbol}): {e}")
        return None


def clean_price_series(data, column_name, jump_threshold=2.0, logger=None):
    """
    Detect and back-adjust contract switch errors in price series.
    
    jump_threshold: Factor where a price jump is considered an error (default: 2x = 200% change).
    """
    series = data[column_name].copy()
    
    # Calculate daily price ratios
    ratios = series / series.shift(1)
    
    # Find dates where price jumps more than threshold
    jump_dates = series.index[
        (ratios > jump_threshold) | (ratios < 1 / jump_threshold)
    ]
    
    if len(jump_dates) == 0:
        return data
    
    for jump_date in jump_dates:
        factor = ratios.loc[jump_date]
        if logger:
            logger.info(f"  Detected contract switch in '{column_name}' on {jump_date.date()}: "
                       f"price jump factor {factor:.2f}x - back-adjusting history")
        
        # Adjust all earlier prices
        series.loc[series.index < jump_date] *= factor
    
    data = data.copy()
    data[column_name] = series
    return data


def fetch_futures_data(DS, symbol, column_name, start_date, end_date, fields_to_try=None, logger=None):
    """Fetch futures data with fallback field logic."""
    if fields_to_try is None:
        fields_to_try = ['P', 'PS', 'P0']
    
    for field in fields_to_try:
        try:
            data = DS.fetch(
                symbol,
                fields=field,
                date_from=start_date,
                date_to=end_date,
                freq='D'
            )
            
            if data is not None and not data.empty and data.iloc[:, 0].notna().any():
                data = data.rename(columns={field: column_name})
                data = clean_price_series(data, column_name, logger=logger)
                if logger:
                    logger.debug(f"  Fetched {column_name} ({symbol}) with field '{field}'")
                return data
            else:
                if logger:
                    logger.debug(f"  {symbol} field '{field}' returned empty data, trying next...")
        
        except Exception as e:
            if logger:
                logger.debug(f"  {symbol} field '{field}' failed: {e}, trying next...")
    
    if logger:
        logger.warning(f"  Could not fetch {column_name} ({symbol}) - skipping this asset")
    return None


def fetch_gics_sectors(DS, start_date, end_date, logger):
    """Fetch all GICS sector data."""
    logger.info("Fetching GICS sector data...")
    
    gics_sectors = [
        ('M1AFCS$(MSNR)~DKK', 'Consumer Staples'),
        ('M1AFID$(MSNR)~DKK', 'Industrials'),
        ('M1AFHC$(MSNR)~DKK', 'Health Care'),
        ('M1AFM1$(MSNR)~DKK', 'Materials'),
        ('M1AFIT$(MSNR)~DKK', 'IT'),
        ('M1AFFN$(MSNR)~DKK', 'Financials'),
        ('M1AFCD$(MSNR)~DKK', 'Consumer Discretionary'),
        ('M1AFE1$(MSNR)~DKK', 'Energy'),
        ('M1AFR1$(MSNR)~DKK', 'Real Estate'),
        ('M1AFU1$(MSNR)~DKK', 'Utilities'),
        ('M1AFT1$(MSNR)~DKK', 'Communication Services'),
    ]
    
    gics_data = None
    
    for symbol, column_name in gics_sectors:
        series = fetch_and_rename_data(DS, symbol, column_name, start_date, end_date, logger)
        if series is not None:
            gics_data = series if gics_data is None else pd.merge(
                gics_data, series, left_index=True, right_index=True, how='outer'
            )
    
    if gics_data is None or gics_data.empty:
        logger.error("✗ No GICS data was fetched")
        raise ValueError("GICS data fetch failed")
    
    logger.info(f"✓ Fetched GICS data: {len(gics_data)} rows, {len(gics_data.columns)} sectors")
    return gics_data


def fetch_futures(DS, start_date, end_date, logger):
    """Fetch all futures data (equities and bonds)."""
    logger.info("Fetching futures data...")
    
    equity_futures = [
        ('ISMCS00', 'S&P 500'),
        ('CENCS00', 'NASDAQ 100'),
        ('GEXCS00', 'Euro Stoxx 50'),
        ('ONACS00', 'Nikkei 225'),
        ('SCNCS00', 'FTSE China A50'),
        ('GDXCS00', 'DAX'),
        ('HSICS00', 'Hang Seng', ['PS', 'P']),
        ('LSXCS00', 'FTSE 100'),
    ]
    
    bond_futures = [
        ('GGECS00', 'EURO BUND 10Y'),
        ('GBECS00', 'EURO BOBL 5Y'),
        ('GEBCS00', 'EURO SCHATZ 2Y'),
        ('CTTCS00', 'US T-Note 10Y'),
        ('CTFCS00', 'US T-Note 5Y'),
        ('CTECS00', 'US T-Note 2Y'),
    ]
    
    futures_data = None
    
    # Equity futures
    for item in equity_futures:
        symbol, column_name = item[0], item[1]
        fields = item[2] if len(item) > 2 else ['P', 'PS', 'PO']
        
        series = fetch_futures_data(DS, symbol, column_name, start_date, end_date, fields, logger)
        if series is not None:
            futures_data = series if futures_data is None else pd.merge(
                futures_data, series, left_index=True, right_index=True, how='outer'
            )
    
    # Bond futures
    for symbol, column_name in bond_futures:
        series = fetch_futures_data(DS, symbol, column_name, start_date, end_date, ['P'], logger)
        if series is not None:
            futures_data = series if futures_data is None else pd.merge(
                futures_data, series, left_index=True, right_index=True, how='outer'
            )
    
    if futures_data is None or futures_data.empty:
        logger.error("✗ No futures data was fetched")
        raise ValueError("Futures data fetch failed")
    
    logger.info(f"✓ Fetched futures data: {len(futures_data)} rows, {len(futures_data.columns)} assets")
    return futures_data


# ============================================================================
# DATA CLEANING & PROCESSING
# ============================================================================
#def clean_and_process_data(price_data, dataset_name, logger):
    """Clean price data and calculate returns."""
    logger.info(f"Processing {dataset_name}...")
    
    # Remove rows with all NaN
    price_data = price_data.dropna(how='all')
    
    # Forward fill then back fill to handle gaps
    price_data = price_data.fillna(method='ffill').fillna(method='bfill')

    # Remove columns with too many missing values
    missing_pct = price_data.isna().sum() / len(price_data) * 100
    cols_to_drop = missing_pct[missing_pct > 50].index.tolist()
    
    if cols_to_drop:
        logger.info(f"  Dropping {len(cols_to_drop)} columns with >50% missing data: {cols_to_drop}")
        price_data = price_data.drop(columns=cols_to_drop)
    
    # Calculate daily returns
    returns = price_data.pct_change().dropna()
    
    logger.info(f"  Cleaned data: {len(returns)} rows, {len(returns.columns)} assets")
    logger.info(f"  Date range: {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')}")
    
    return returns

def clean_and_process_data(price_data, dataset_name, logger):
    """Clean price data and calculate returns."""
    logger.info(f"Processing {dataset_name}...")

    # Remove rows where ALL columns are NaN
    price_data = price_data.dropna(how='all')

    # Remove columns with too many missing values
    missing_pct = price_data.isna().sum() / len(price_data) * 100
    cols_to_drop = missing_pct[missing_pct > 50].index.tolist()

    if cols_to_drop:
        logger.info(f"  Dropping {len(cols_to_drop)} columns with >50% missing data: {cols_to_drop}")
        price_data = price_data.drop(columns=cols_to_drop)

    # Calculate daily returns — behold NaN, kovariansestimatoren håndterer dem
    returns = price_data.pct_change()
    #Remove return rows with all NaN
    returns.dropna(how='all', inplace=True)

    logger.info(f"  Cleaned data: {len(returns)} rows, {len(returns.columns)} assets")
    logger.info(f"  Date range: {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')}")

    return returns


# ============================================================================
# RISK PARITY OPTIMIZATION
# ============================================================================

def run_risk_parity_backtest(returns, dataset_name, logger):
    """Run Risk Parity backtest with configured parameters."""
    logger.info(f"Running Risk Parity optimization for {dataset_name}...")
    
    # Configuration parameters
    REBALANCE_FREQ = 'W'          # 'ME' = monthly
    LOOKBACK_WINDOW = 90          # 252 days = 1 year of trading days
    TARGET_VOL = None              # No volatility scaling
    COV_ESTIMATOR = sample_covariance  # Can switch to ewma_covariance or ledoit_wolf_covariance
    
    logger.info(f"  Rebalance frequency: {REBALANCE_FREQ}")
    logger.info(f"  Lookback window: {LOOKBACK_WINDOW} days")
    logger.info(f"  Covariance estimator: {COV_ESTIMATOR.__name__}")
    
    try:
        weights_df, risk_contrib_df, portfolio_vol_df, rebalance_dates = rolling_rp_backtest(
            returns=returns,
            rebalance_freq=REBALANCE_FREQ,
            lookback_window=LOOKBACK_WINDOW,
            target_vol=TARGET_VOL,
            cov_estimator=COV_ESTIMATOR,
        )
        
        logger.info(f"✓ Optimization completed: {len(weights_df)} rebalances")
        logger.info(f"  Date range: {weights_df.index[0].strftime('%Y-%m-%d')} to {weights_df.index[-1].strftime('%Y-%m-%d')}")
        
        return weights_df, risk_contrib_df, portfolio_vol_df
    
    except Exception as e:
        logger.error(f"✗ Risk Parity optimization failed: {e}")
        raise


# ============================================================================
# OUTPUT & SAVING
# ============================================================================

def save_results(weights_df, risk_contrib_df, portfolio_vol_df, returns, dataset_name, output_dir, logger):
    """Save backtest results to CSV files."""
    logger.info(f"Saving results for {dataset_name}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths
    weights_file = os.path.join(output_dir, f'{dataset_name}_weights.csv')
    risk_contrib_file = os.path.join(output_dir, f'{dataset_name}_risk_contributions.csv')
    portfolio_vol_file = os.path.join(output_dir, f'{dataset_name}_portfolio_volatility.csv')
    returns_file = os.path.join(output_dir, f'{dataset_name}_returns.csv')
    
    # Save files
    weights_df.to_csv(weights_file)
    risk_contrib_df.to_csv(risk_contrib_file)
    portfolio_vol_df.to_csv(portfolio_vol_file)
    returns.to_csv(returns_file)
    
    logger.info(f"✓ Saved weights: {weights_file}")
    logger.info(f"✓ Saved risk contributions: {risk_contrib_file}")
    logger.info(f"✓ Saved portfolio volatility: {portfolio_vol_file}")
    logger.info(f"✓ Saved returns: {returns_file}")
    
    # Print summary statistics
    logger.info(f"\n--- Summary Statistics for {dataset_name} ---")
    logger.info(f"Latest weights (as of {weights_df.index[-1].strftime('%Y-%m-%d')}):")
    for asset, weight in weights_df.iloc[-1].items():
        logger.info(f"  {asset:20s}: {weight:7.2%}")
    
    logger.info(f"\nPortfolio volatility statistics:")
    vol = portfolio_vol_df['Portfolio Vol']
    logger.info(f"  Mean:  {vol.mean():.4f}")
    logger.info(f"  Min:   {vol.min():.4f}")
    logger.info(f"  Max:   {vol.max():.4f}")
    logger.info(f"  Std:   {vol.std():.4f}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute the complete Risk Parity pipeline."""
    
    # Setup
    logger, log_file = setup_logging()
    
    try:
        # Date range for data collection
        start_gics = '20160901' #Valgt den dato, hvor der er data fra alle aktiver
        end_gics = datetime.today().strftime('%Y%m%d')
        start_futures = '20060915' #Valgt den dato, hvor der er data fra alle aktiver
        end_futures = datetime.today().strftime('%Y%m%d')
        
        output_dir = 'output'
        
        logger.info(f"Data collection period:")
        logger.info(f"  GICS Sectors: {start_gics} to {end_gics}")
        logger.info(f"  Futures: {start_futures} to {end_futures}")
        logger.info(f"  Output directory: {output_dir}")
        
        # ===== DATASTREAM CONNECTION =====
        DS = connect_to_datastream(logger)
        
        # ===== FETCH DATA =====
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: DATA COLLECTION")
        logger.info("=" * 80)
        
        gics_data = fetch_gics_sectors(DS, start_gics, end_gics, logger)
        futures_data = fetch_futures(DS, start_futures, end_futures, logger)
        
        # ===== CLEAN & PROCESS DATA =====
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: DATA CLEANING & PROCESSING")
        logger.info("=" * 80)
        
        gics_returns = clean_and_process_data(gics_data, "GICS Sectors", logger)
        futures_returns = clean_and_process_data(futures_data, "Futures", logger)
        
        # ===== RISK PARITY OPTIMIZATION =====
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: RISK PARITY OPTIMIZATION")
        logger.info("=" * 80)
        
        gics_weights, gics_risk_contrib, gics_portfolio_vol = run_risk_parity_backtest(
            gics_returns, "GICS Sectors", logger
        )
        
        futures_weights, futures_risk_contrib, futures_portfolio_vol = run_risk_parity_backtest(
            futures_returns, "Futures", logger
        )
        
        # ===== SAVE RESULTS =====
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: SAVING RESULTS")
        logger.info("=" * 80)
        
        save_results(gics_weights, gics_risk_contrib, gics_portfolio_vol, gics_returns, 
                    "GICS", output_dir, logger)
        
        save_results(futures_weights, futures_risk_contrib, futures_portfolio_vol, futures_returns,
                    "Futures", output_dir, logger)
        
        # ===== COMPLETION =====
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Log file: {log_file}")
        logger.info(f"Output directory: {os.path.abspath(output_dir)}")
    
    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error("PIPELINE FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

