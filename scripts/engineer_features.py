#!/usr/bin/env python3
"""Script to engineer features for Bitcoin price prediction"""

import os
import sys
import argparse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import PATHS, TECHNICAL_INDICATORS
from src.utils.helpers import setup_logger
from src.data.engineer import engineer_features, FeatureEngineer

def parse_arguments():
    """Parse command-line arguments.
    
    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Engineer features for Bitcoin price prediction')
    
    parser.add_argument('--input', type=str, default=PATHS['PROCESSED_DATA_PATH'],
                        help='Input data file path')
    
    parser.add_argument('--output', type=str, default=PATHS['FEATURED_DATA_PATH'],
                        help='Output file path')
    
    parser.add_argument('--indicators', nargs='+', 
                        default=TECHNICAL_INDICATORS,
                        help='Technical indicators to include')
    
    parser.add_argument('--pca', action='store_true',
                        help='Apply PCA dimensionality reduction')
    
    parser.add_argument('--components', type=int, default=10,
                        help='Number of PCA components to keep')
    
    parser.add_argument('--anomaly', action='store_true',
                        help='Add anomaly detection features')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logger('feature_engineer', 'feature_engineering.log')
    
    print(f"Engineering features for Bitcoin price data from {args.input}...")
    
    # Create custom feature engineer if custom options specified
    if args.indicators != TECHNICAL_INDICATORS or args.pca or args.anomaly:
        print("Using custom feature engineering settings")
        engineer = FeatureEngineer(
            input_file=args.input,
            output_file=args.output
        )
        
        # Load data
        data = engineer.load_data()
        if data is None:
            print(f"Failed to load data from {args.input}")
            return 1
        
        # Add technical indicators
        engineer.add_technical_indicators(indicators=args.indicators)
        
        # Add price patterns
        engineer.add_price_patterns()
        
        # Add market regime features
        engineer.add_market_regime_features()
        
        # Add custom features
        engineer.add_custom_features()
        
        # Apply PCA if requested
        if args.pca:
            print(f"Applying PCA dimensionality reduction with {args.components} components")
            engineer.add_dimensionality_reduction(n_components=args.components, method='pca')
        
        # Add anomaly detection if requested
        if args.anomaly:
            print("Adding anomaly detection features")
            engineer.add_anomaly_detection_features()
        
        # Remove NaN values
        engineer.remove_nan_values()
        
        # Save data
        engineer.save_data()
        
        data = engineer.data
    else:
        # Use standard feature engineering
        data = engineer_features(
            input_file=args.input,
            output_file=args.output
        )
    
    if data is not None:
        print(f"Successfully engineered features for {len(data)} records")
        print(f"Data saved to {args.output}")
        print(f"\nTotal number of features: {len(data.columns)}")
        print("\nSample features:")
        # Print some technical indicator and custom feature names
        tech_indicators = [col for col in data.columns if any(indicator in col for indicator in ['RSI', 'MACD', 'Bollinger'])]
        custom_features = [col for col in data.columns if any(feature in col for feature in ['momentum', 'volatility', 'regime'])]
        
        print("Technical indicators: " + ", ".join(tech_indicators[:5]) + (", ..." if len(tech_indicators) > 5 else ""))
        print("Custom features: " + ", ".join(custom_features[:5]) + (", ..." if len(custom_features) > 5 else ""))
        
        print("\nSample data:")
        print(data.head())
        return 0
    else:
        print("Feature engineering failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)