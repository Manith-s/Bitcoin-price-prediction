#!/usr/bin/env python3
"""Script to visualize Bitcoin price data and model predictions"""

import os
import sys
import argparse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import PATHS, RESULTS_DIR
from src.utils.helpers import setup_logger
from src.evaluation.visualizer import create_visualizations, DataVisualizer

def parse_arguments():
    """Parse command-line arguments.
    
    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Visualize Bitcoin price data and model predictions')
    
    parser.add_argument('--input', type=str, default=PATHS['FEATURED_DATA_PATH'],
                        help='Input data file path')
    
    parser.add_argument('--output-dir', type=str, default=RESULTS_DIR,
                        help='Directory to save visualization results')
    
    parser.add_argument('--visualization-type', type=str, default='all',
                        choices=['all', 'basic', 'technical', 'correlation', 'distributions', 
                                 'returns', 'volatility', 'seasonality', 'regimes', 'dashboard'],
                        help='Type of visualization to generate')
    
    parser.add_argument('--candlestick-days', type=int, default=90,
                        help='Number of days to include in candlestick chart')
    
    parser.add_argument('--show-plots', action='store_true',
                        help='Show plots in addition to saving them')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logger('data_visualizer', 'data_visualization.log')
    
    print(f"Visualizing Bitcoin price data from {args.input}...")
    
    # Create visualizer
    visualizer = DataVisualizer(args.input, args.output_dir)
    
    # Load data
    data = visualizer.load_data()
    
    if data is None:
        print(f"Failed to load data from {args.input}")
        return 1
    
    print(f"Data loaded, shape: {data.shape}")
    
    # Generate visualizations based on type
    if args.visualization_type in ['all', 'basic']:
        print("\nGenerating basic price history and volume visualizations...")
        visualizer.plot_price_history(show_plot=args.show_plots)
        visualizer.plot_price_with_volume(show_plot=args.show_plots)
    
    if args.visualization_type in ['all', 'technical']:
        print("Generating technical analysis visualizations...")
        visualizer.plot_candlestick(days=args.candlestick_days, show_plot=args.show_plots)
        visualizer.plot_technical_indicators(show_plot=args.show_plots)
    
    if args.visualization_type in ['all', 'correlation']:
        print("Generating correlation matrix...")
        visualizer.plot_correlation_matrix(show_plot=args.show_plots)
    
    if args.visualization_type in ['all', 'distributions']:
        print("Generating feature distributions...")
        visualizer.plot_feature_distributions(show_plot=args.show_plots)
    
    if args.visualization_type in ['all', 'returns']:
        print("Generating returns analysis...")
        visualizer.plot_returns_analysis(show_plot=args.show_plots)
    
    if args.visualization_type in ['all', 'volatility']:
        print("Generating volatility analysis...")
        visualizer.plot_volatility_analysis(show_plot=args.show_plots)
    
    if args.visualization_type in ['all', 'seasonality']:
        print("Generating seasonality analysis...")
        visualizer.plot_seasonality_analysis(show_plot=args.show_plots)
    
    if args.visualization_type in ['all', 'regimes']:
        print("Generating market regimes analysis...")
        visualizer.plot_market_regimes(show_plot=args.show_plots)
    
    if args.visualization_type in ['all', 'dashboard']:
        print("Generating interactive dashboard...")
        visualizer.create_interactive_dashboard(show_dashboard=args.show_plots)
    
    if args.visualization_type == 'all':
        print("\nAll visualizations have been generated.")
    
    print(f"\nVisualizations saved to {args.output_dir}")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)