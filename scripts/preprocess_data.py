#!/usr/bin/env python3
"""Script to preprocess Bitcoin price data"""

import os
import sys
import argparse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import PATHS
from src.utils.helpers import setup_logger
from src.data.preprocessor import preprocess_data

def parse_arguments():
    """Parse command-line arguments.
    
    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Preprocess Bitcoin price data')
    
    parser.add_argument('--input', type=str, default=PATHS['RAW_DATA_PATH'],
                        help='Input data file path')
    
    parser.add_argument('--output', type=str, default=PATHS['PROCESSED_DATA_PATH'],
                        help='Output file path')
    
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize the data')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logger('data_preprocessor', 'data_preprocessing.log')
    
    print(f"Preprocessing Bitcoin price data from {args.input}...")
    
    # Preprocess data
    data = preprocess_data(
        input_file=args.input,
        output_file=args.output
    )
    
    if data is not None:
        print(f"Successfully preprocessed {len(data)} records")
        print(f"Data saved to {args.output}")
        print("\nSample data:")
        print(data.head())
        print("\nNew columns added:")
        original_cols = ['open', 'high', 'low', 'close', 'volume']
        new_cols = [col for col in data.columns if col not in original_cols]
        print(", ".join(new_cols[:10]) + (", ..." if len(new_cols) > 10 else ""))
        return 0
    else:
        print("Data preprocessing failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)