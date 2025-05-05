#!/usr/bin/env python3
"""Script to evaluate models for Bitcoin price prediction"""

import os
import sys
import argparse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import PATHS, RESULTS_DIR
from src.utils.helpers import setup_logger
from src.evaluation.evaluator import evaluate_models, ModelEvaluator

def parse_arguments():
    """Parse command-line arguments.
    
    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate models for Bitcoin price prediction')
    
    parser.add_argument('--model-type', type=str, default='all',
                        choices=['all', 'traditional', 'deep_learning', 'ensemble', 'rl'],
                        help='Type of models to evaluate')
    
    parser.add_argument('--metric', type=str, default='RMSE',
                        choices=['RMSE', 'MAE', 'R2', 'Directional Accuracy', 'F1 Score'],
                        help='Primary metric for comparison')
    
    parser.add_argument('--feature-importance', action='store_true',
                        help='Generate feature importance analysis')
    
    parser.add_argument('--directional-accuracy', action='store_true',
                        help='Generate directional accuracy analysis')
    
    parser.add_argument('--summary-report', action='store_true',
                        help='Generate summary report')
    
    parser.add_argument('--output-dir', type=str, default=RESULTS_DIR,
                        help='Directory to save evaluation results')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logger('model_evaluator', 'model_evaluation.log')
    
    print(f"Evaluating {args.model_type} models for Bitcoin price prediction...")
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Load results
    results = evaluator.load_results(model_type=args.model_type)
    
    if not results:
        print(f"No results found for model type: {args.model_type}")
        return 1
    
    # Compare models
    print(f"\nComparing models based on {args.metric}:")
    comparison = evaluator.compare_models(metric=args.metric, ascending=args.metric in ['RMSE', 'MAE'])
    
    if comparison:
        for model_name, value in comparison.items():
            print(f"  {model_name}: {value:.4f}")
        
        best_model = list(comparison.keys())[0]
        best_value = list(comparison.values())[0]
        print(f"\nBest model by {args.metric}: {best_model} ({args.metric}={best_value:.4f})")
    else:
        print(f"No models found for comparison")
    
    # Generate feature importance analysis if requested
    if args.feature_importance:
        print("\nGenerating feature importance analysis...")
        feature_importance = evaluator.evaluate_feature_importance()
        
        if feature_importance is not None:
            print("\nTop 10 features by importance:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                print(f"  {i+1}. {row['feature']}: {row['avg_importance']:.4f}")
        else:
            print("Feature importance analysis not available")
    
    # Generate directional accuracy analysis if requested
    if args.directional_accuracy:
        print("\nGenerating directional accuracy analysis...")
        directional_accuracy = evaluator.evaluate_directional_accuracy()
        
        if directional_accuracy:
            print("\nDirectional accuracy metrics:")
            for model_name, metrics in directional_accuracy.items():
                print(f"  {model_name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
        else:
            print("Directional accuracy analysis not available")
    
    # Generate summary report if requested
    if args.summary_report:
        print("\nGenerating summary report...")
        summary_report = evaluator.create_summary_report()
        
        if summary_report:
            print(f"Summary report saved to {os.path.join(args.output_dir, 'model_evaluation_summary.md')}")
        else:
            print("Failed to generate summary report")
    
    # Create visualizations
    print("\nGenerating evaluation visualizations...")
    evaluator.plot_model_comparison()
    evaluator.plot_prediction_comparison()
    
    print(f"\nEvaluation completed. Results saved to {args.output_dir}")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)