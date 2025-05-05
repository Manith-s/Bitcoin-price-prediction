"""Main module for Bitcoin price prediction project"""

import os
import argparse
import logging
import datetime

from config.config import (
    CRYPTO_SYMBOL, QUOTE_CURRENCY, TIME_INTERVAL,
    START_DATE, END_DATE, PATHS
)
from src.utils.helpers import logger, setup_logger
from src.data.collector import collect_data
from src.data.preprocessor import preprocess_data
from src.data.engineer import engineer_features
from src.models.traditional import train_traditional_models
from src.models.deep_learning import train_deep_learning_models
from src.models.ensemble import train_ensemble_models
from src.evaluation.evaluator import evaluate_models
from src.evaluation.visualizer import create_visualizations

def parse_arguments():
    """Parse command-line arguments.
    
    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Bitcoin Price Prediction')
    
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'collect', 'preprocess', 'engineer', 
                                 'train', 'evaluate', 'visualize', 'train-traditional',
                                 'train-deep-learning', 'train-ensemble'],
                        help='Mode to run the pipeline in')
    
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
    
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    parser.add_argument('--show_plots', action='store_true',
                        help='Show plots during visualization')
    
    return parser.parse_args()

def run_data_collection(args):
    """Run data collection pipeline.
    
    Args:
        args: Command-line arguments
        
    Returns:
        DataFrame of collected data or None if collection failed
    """
    logger.info("Running data collection pipeline...")
    
    # Set end date to today if not provided
    if args.end_date is None:
        args.end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Collect data
    data = collect_data(
        symbol=args.symbol,
        quote=args.quote,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        output_file=PATHS['RAW_DATA_PATH']
    )
    
    logger.info("Data collection completed")
    return data

def run_preprocessing(args):
    """Run data preprocessing pipeline.
    
    Args:
        args: Command-line arguments
        
    Returns:
        DataFrame of preprocessed data or None if preprocessing failed
    """
    logger.info("Running data preprocessing pipeline...")
    
    # Preprocess data
    data = preprocess_data(
        input_file=PATHS['RAW_DATA_PATH'],
        output_file=PATHS['PROCESSED_DATA_PATH']
    )
    
    logger.info("Data preprocessing completed")
    return data

def run_feature_engineering(args):
    """Run feature engineering pipeline.
    
    Args:
        args: Command-line arguments
        
    Returns:
        DataFrame of engineered data or None if engineering failed
    """
    logger.info("Running feature engineering pipeline...")
    
    # Engineer features
    data = engineer_features(
        input_file=PATHS['PROCESSED_DATA_PATH'],
        output_file=PATHS['FEATURED_DATA_PATH']
    )
    
    logger.info("Feature engineering completed")
    return data

def run_model_training(args):
    """Run model training pipeline.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary of models or None if training failed
    """
    logger.info("Running model training pipeline...")
    
    models = {}
    
    # Train traditional models
    if args.mode == 'full' or args.mode == 'train' or args.mode == 'train-traditional':
        traditional_models, traditional_results = train_traditional_models(
            data_file=PATHS['FEATURED_DATA_PATH']
        )
        models['traditional'] = traditional_models
    
    # Train deep learning models
    if args.mode == 'full' or args.mode == 'train' or args.mode == 'train-deep-learning':
        deep_learning_models, deep_learning_results = train_deep_learning_models(
            data_file=PATHS['FEATURED_DATA_PATH']
        )
        models['deep_learning'] = deep_learning_models
    
    # Train ensemble models
    if args.mode == 'full' or args.mode == 'train' or args.mode == 'train-ensemble':
        ensemble_models, ensemble_results = train_ensemble_models(
            data_file=PATHS['FEATURED_DATA_PATH']
        )
        models['ensemble'] = ensemble_models
    
    logger.info("Model training completed")
    return models

def run_evaluation(args):
    """Run model evaluation pipeline.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info("Running model evaluation pipeline...")
    
    # Evaluate models
    results = evaluate_models(model_type='all')
    
    logger.info("Model evaluation completed")
    return results

def run_visualization(args):
    """Run data visualization pipeline.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary of visualization figures
    """
    logger.info("Running data visualization pipeline...")
    
    # Create visualizations
    figures = create_visualizations(
        data_file=PATHS['FEATURED_DATA_PATH'],
        show_plots=args.show_plots
    )
    
    logger.info("Data visualization completed")
    return figures

def run_full_pipeline(args):
    """Run the full pipeline.
    
    Args:
        args: Command-line arguments
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Running full Bitcoin price prediction pipeline...")
    
    try:
        # Run each pipeline stage
        data = run_data_collection(args)
        if data is None:
            logger.error("Data collection failed")
            return False
            
        data = run_preprocessing(args)
        if data is None:
            logger.error("Data preprocessing failed")
            return False
            
        data = run_feature_engineering(args)
        if data is None:
            logger.error("Feature engineering failed")
            return False
            
        models = run_model_training(args)
        if not models:
            logger.error("Model training failed")
            return False
            
        results = run_evaluation(args)
        if not results:
            logger.error("Model evaluation failed")
            return False
            
        figures = run_visualization(args)
        if not figures:
            logger.error("Data visualization failed")
            return False
        
        logger.info("Full pipeline completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in full pipeline: {str(e)}")
        return False

def main():
    """Main function.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger('bitcoin_prediction', 'bitcoin_prediction.log', level=log_level)
    
    try:
        # Run specified mode
        if args.mode == 'full':
            success = run_full_pipeline(args)
        elif args.mode == 'collect':
            data = run_data_collection(args)
            success = data is not None
        elif args.mode == 'preprocess':
            data = run_preprocessing(args)
            success = data is not None
        elif args.mode == 'engineer':
            data = run_feature_engineering(args)
            success = data is not None
        elif args.mode == 'train' or args.mode.startswith('train-'):
            models = run_model_training(args)
            success = models is not None
        elif args.mode == 'evaluate':
            results = run_evaluation(args)
            success = results is not None
        elif args.mode == 'visualize':
            figures = run_visualization(args)
            success = figures is not None
        else:
            logger.error(f"Unknown mode: {args.mode}")
            return 1
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)