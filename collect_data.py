#!/usr/bin/env python3
"""Script to collect Bitcoin price data"""

import os
import sys
import argparse
import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    CRYPTO_SYMBOL, QUOTE_CURRENCY, TIME_INTERVAL,
    START_DATE, END_DATE, PATHS
)
from src.utils.helpers import setup_logger
from src.data.collector import collect_data

def parse_arguments():
    """Parse command-line arguments.
    
    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Collect Bitcoin price data')
    
    parser.add_argument('--symbol', type=str, default=CRYPTO_SYMBOL,
                        help='Cryptocurrency symbol')
    
    parser.add_argument('--quote', type=str, default=QUOTE_CURRENCY,
                        help='Quote currency')
    
    parser.add_argument('--interval', type=str, default=TIME_INTERVAL,
                        help='Time interval (1d, 1h, etc.)')
    
    parser.add_argument('--start_date', type=str, default=START_DATE,
                        help='Start date for data collection (YYYY-MM-DD)')
    
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date for data collection (YYYY-MM-DD)')
    
    parser.add_argument('--output', type=str, default=PATHS['RAW_DATA_PATH'],
                        help='Output file path')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    parser.add_argument('--source', type=str, default=None,
                        help='Specific data source to use (binance, ccxt, cryptocompare, yfinance)')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logger('data_collector', 'data_collection.log')
    
    # Set end date to today if not provided
    if args.end_date is None:
        args.end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Set up data sources
    sources = [args.source] if args.source else None
    
    print(f"Collecting {args.symbol}/{args.quote} price data from {args.start_date} to {args.end_date}...")
    
    # Collect data
    data = collect_data(
        symbol=args.symbol,
        quote=args.quote,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        output_file=args.output,
        sources=sources
    )
    
    if data is not None:
        print(f"Successfully collected {len(data)} records")
        print(f"Data saved to {args.output}")
        print("\nSample data:")
        print(data.head())
        return 0
    else:
        print("Data collection failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)