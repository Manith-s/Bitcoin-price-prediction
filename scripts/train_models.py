#!/usr/bin/env python3
"""Script to train models for Bitcoin price prediction"""

import os
import sys
import argparse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import PATHS, ENSEMBLE_MODELS
from src.utils.helpers import setup_logger
from src.models.traditional import train_traditional_models
from src.models.deep_learning import train_deep_learning_models
from src.models.ensemble import train_ensemble_models
from src.models.rl import train_rl_models

def parse_arguments():
    """Parse command-line arguments.
    
    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Train models for Bitcoin price prediction')
    
    parser.add_argument('--input', type=str, default=PATHS['FEATURED_DATA_PATH'],
                        help='Input data file path')
    
    parser.add_argument('--target', type=str, default='close',
                        help='Target column to predict')
    
    parser.add_argument('--model-type', type=str, default='all',
                        choices=['all', 'traditional', 'deep-learning', 'ensemble'],
                        help='Type of models to train')
    
    parser.add_argument('--traditional-models', nargs='+',
                        default=['RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost'],
                        help='Traditional models to train')
    
    parser.add_argument('--deep-learning-models', nargs='+',
                        default=['LSTM', 'GRU', 'BidirectionalLSTM'],
                        help='Deep learning models to train')
    
    parser.add_argument('--ensemble-models', nargs='+',
                        default=ENSEMBLE_MODELS,
                        help='Base models for ensemble')
    
    parser.add_argument('--sequence-length', type=int, default=10,
                        help='Sequence length for sequential models')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logger('model_trainer', 'model_training.log')
    
    print(f"Training models for Bitcoin price prediction using data from {args.input}...")
    
    # Train specified model types
    if args.model_type in ['all', 'traditional']:
        print("\nTraining traditional machine learning models...")
        traditional_models, traditional_results = train_traditional_models(
            data_file=args.input,
            target_col=args.target
        )
        
        if traditional_models:
            print("Traditional models trained successfully:")
            for model_name, model in traditional_models.items():
                test_metrics = traditional_results[model_name]['test_metrics']
                print(f"  - {model_name}: RMSE={test_metrics['RMSE']:.4f}, DirectionalAcc={test_metrics['Directional Accuracy']:.4f}")
        else:
            print("Failed to train traditional models")
    
    if args.model_type in ['all', 'deep-learning']:
        print("\nTraining deep learning models...")
        deep_learning_models, deep_learning_results = train_deep_learning_models(
            data_file=args.input,
            target_col=args.target,
            sequence_length=args.sequence_length
        )
        
        if deep_learning_models:
            print("Deep learning models trained successfully:")
            for model_name, model in deep_learning_models.items():
                test_metrics = deep_learning_results[model_name]['test_metrics']
                print(f"  - {model_name}: RMSE={test_metrics['RMSE']:.4f}, DirectionalAcc={test_metrics['Directional Accuracy']:.4f}")
        else:
            print("Failed to train deep learning models")
    
    if args.model_type in ['all', 'ensemble']:
        print("\nTraining ensemble models...")
        ensemble_models, ensemble_results = train_ensemble_models(
            data_file=args.input,
            target_col=args.target
        )
        
        if ensemble_models:
            print("Ensemble models trained successfully:")
            for model_name, model in ensemble_models.items():
                test_metrics = ensemble_results[model_name]['test_metrics']
                print(f"  - {model_name}: RMSE={test_metrics['RMSE']:.4f}, DirectionalAcc={test_metrics['Directional Accuracy']:.4f}")
        else:
            print("Failed to train ensemble models")
    
    if args.model_type in ['all', 'rl']:
        print("\nTraining reinforcement learning models...")
        rl_models, rl_results = train_rl_models(
            data_file=args.input,
            algorithms=args.rl_algorithms,
            window_size=args.sequence_length,
            total_timesteps=50000
        )
        
        if rl_models:
            print("Reinforcement learning models trained successfully:")
            for model_name, model in rl_models.items():
                metrics = rl_results[model_name]
                print(f"  - {model_name}: TotalReturn={metrics['total_return']:.4f}, SharpeRatio={metrics['sharpe_ratio']:.4f}")
        else:
            print("Failed to train reinforcement learning models")
    
    print("\nModel training completed. Check the results directory for evaluation metrics and visualizations.")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)