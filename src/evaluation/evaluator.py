"""Model evaluation module for Bitcoin price prediction"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import FEATURED_DATA_PATH, MODEL_DIR, RESULTS_DIR, CRYPTO_SYMBOL, QUOTE_CURRENCY
from src.utils.helpers import logger, load_model, calculate_metrics

class ModelEvaluator:
    """Class to evaluate and compare trained models."""
    
    def __init__(self, results_file='model_results.json', test_data=None):
        """Initialize ModelEvaluator with results file and test data.
        
        Args:
            results_file: Name of the JSON file containing model results
            test_data: Test data for evaluation
        """
        self.results_file = os.path.join(RESULTS_DIR, results_file)
        self.test_data = test_data
        self.results = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
    
    def load_results(self, model_type=None):
        """Load results from JSON file(s).
        
        Args:
            model_type: Type of models to load results for (traditional, deep_learning, ensemble, or None for all)
            
        Returns:
            Dictionary of model results
        """
        try:
            results = {}
            
            # Define file patterns based on model type
            if model_type is None or model_type == 'all':
                file_patterns = [
                    'traditional_models_results.json', 
                    'deep_learning_models_results.json', 
                    'ensemble_models_results.json'
                ]
            elif model_type == 'traditional':
                file_patterns = ['traditional_models_results.json']
            elif model_type == 'deep_learning':
                file_patterns = ['deep_learning_models_results.json']
            elif model_type == 'ensemble':
                file_patterns = ['ensemble_models_results.json']
            else:
                file_patterns = [f'{model_type}_results.json']
            
            # Load each file
            for file_pattern in file_patterns:
                file_path = os.path.join(RESULTS_DIR, file_pattern)
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        model_results = json.load(f)
                        results.update(model_results)
            
            if not results:
                logger.warning(f"No results found for model type: {model_type}")
                return None
                
            self.results = results
            logger.info(f"Results loaded for {len(results)} models")
            return self.results
            
        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            return None
    
    def load_models(self, model_names=None):
        """Load trained models from disk.
        
        Args:
            model_names: List of model names to load. If None, load all available models.
            
        Returns:
            Dictionary of loaded models
        """
        # Model name to file name mapping
        model_files = {
            'RandomForest': 'random_forest',
            'GradientBoosting': 'gradient_boosting',
            'XGBoost': 'xgboost',
            'LightGBM': 'lightgbm',
            'CatBoost': 'catboost',
            'LSTM': 'lstm_model',  # Note: This is an .h5 file, needs special handling
            'GRU': 'gru_model',    # Note: This is an .h5 file, needs special handling
            'BidirectionalLSTM': 'bidirectional_lstm_model',  # Note: This is an .h5 file, needs special handling
            'VotingEnsemble': 'voting_ensemble',
            'StackingEnsemble': 'stacking_ensemble',
            'BlendingEnsemble': 'blending_ensemble'
        }
        
        # If no specific models are requested, load all available
        if model_names is None:
            model_names = list(model_files.keys())
        
        for model_name in model_names:
            file_name = model_files.get(model_name)
            if file_name:
                try:
                    # Handle different model file types
                    if model_name in ['LSTM', 'GRU', 'BidirectionalLSTM']:
                        # For Keras models (.h5 files)
                        import tensorflow as tf
                        model_path = os.path.join(MODEL_DIR, f"{file_name}.h5")
                        if os.path.exists(model_path):
                            model = tf.keras.models.load_model(model_path)
                            self.models[model_name] = model
                            logger.info(f"Model {model_name} loaded from {model_path}")
                    else:
                        # For pickled models (.pkl files)
                        model = load_model(file_name)
                        if model is not None:
                            self.models[model_name] = model
                            logger.info(f"Model {model_name} loaded from {file_name}.pkl")
                except Exception as e:
                    logger.warning(f"Could not load model {model_name}: {str(e)}")
        
        return self.models
    
    def compare_models(self, metric='RMSE', ascending=True):
        """Compare models based on a specific metric.
        
        Args:
            metric: Metric to compare models by
            ascending: Whether lower values are better (True for error metrics)
            
        Returns:
            Dictionary of models sorted by metric
        """
        if self.results is None:
            logger.error("No results to compare")
            return None
            
        logger.info(f"Comparing models based on {metric}...")
        
        comparison = {}
        for model_name, model_results in self.results.items():
            if 'test_metrics' in model_results and metric in model_results['test_metrics']:
                comparison[model_name] = model_results['test_metrics'][metric]
        
        # Sort by metric
        comparison = dict(sorted(comparison.items(), key=lambda x: x[1], reverse=not ascending))
        
        # Print comparison
        logger.info(f"Model comparison based on {metric}:")
        for model_name, value in comparison.items():
            logger.info(f"{model_name}: {value:.4f}")
        
        # Identify best model
        if comparison:
            self.best_model_name = list(comparison.keys())[0]
            logger.info(f"Best model based on {metric}: {self.best_model_name}")
        
        return comparison
    
    def plot_model_comparison(self, metrics=None, save_path='model_comparison.png'):
        """Plot model comparison for multiple metrics.
        
        Args:
            metrics: List of metrics to compare. If None, use default metrics.
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if self.results is None:
            logger.error("No results to compare")
            return None
            
        if metrics is None:
            metrics = ['RMSE', 'MAE', 'Directional Accuracy', 'F1 Score']
        
        logger.info(f"Plotting model comparison for metrics: {metrics}")
        
        # Prepare data for plotting
        data = []
        for model_name, model_results in self.results.items():
            if 'test_metrics' in model_results:
                for metric in metrics:
                    if metric in model_results['test_metrics']:
                        data.append({
                            'Model': model_name,
                            'Metric': metric,
                            'Value': model_results['test_metrics'][metric]
                        })
        
        if not data:
            logger.warning("No data for comparison plot")
            return None
        
        df = pd.DataFrame(data)
        
        # Create subplots
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
        
        if len(metrics) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            metric_data = df[df['Metric'] == metric]
            
            if metric_data.empty:
                logger.warning(f"No data for metric {metric}")
                continue
            
            # Sort by value (lower is better for error metrics, higher for others)
            if metric in ['RMSE', 'MSE', 'MAE']:
                metric_data = metric_data.sort_values('Value')
            else:
                metric_data = metric_data.sort_values('Value', ascending=False)
            
            # Plot
            ax = axes[i]
            bars = ax.bar(metric_data['Model'], metric_data['Value'])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{height:.4f}',
                    ha='center', va='bottom', rotation=0
                )
            
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.set_xticklabels(metric_data['Model'], rotation=45, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(RESULTS_DIR, save_path)
        plt.savefig(save_path)
        logger.info(f"Comparison plot saved to {save_path}")
        
        plt.close()
        
        return fig
    
    def plot_prediction_comparison(self, models=None, save_path='prediction_comparison.png'):
        """Plot prediction comparison for multiple models.
        
        Args:
            models: List of model names to compare. If None, use top 3 models based on RMSE.
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if self.results is None:
            logger.error("No results to compare")
            return None
            
        logger.info("Plotting prediction comparison...")
        
        if models is None:
            # Use top 3 models based on RMSE
            comparison = self.compare_models(metric='RMSE')
            models = list(comparison.keys())[:3]
        
        # Prepare data for plotting
        data = []
        actual_data = None
        
        # Add actual values if available
        for model_name in models:
            if model_name in self.results and 'test_pred' in self.results[model_name]:
                # Get predictions
                test_pred = self.results[model_name]['test_pred']
                
                # Get actual values if available
                if 'test_actual' in self.results[model_name] and actual_data is None:
                    actual_data = self.results[model_name]['test_actual']
                
                # Add to data
                for i, pred in enumerate(test_pred):
                    data.append({
                        'Index': i,
                        'Model': model_name,
                        'Value': pred
                    })
        
        if not data:
            logger.warning("No prediction data available")
            return None
        
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot each model's predictions
        for model_name in models:
            model_data = df[df['Model'] == model_name]
            if not model_data.empty:
                plt.plot(model_data['Index'], model_data['Value'], label=model_name)
        
        # Plot actual values if available
        if actual_data:
            plt.plot(range(len(actual_data)), actual_data, label='Actual', linestyle='--', color='black')
        
        plt.title('Model Prediction Comparison')
        plt.xlabel('Time')
        plt.ylabel(f'{CRYPTO_SYMBOL}/{QUOTE_CURRENCY} Price')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        save_path = os.path.join(RESULTS_DIR, save_path)
        plt.savefig(save_path)
        logger.info(f"Prediction comparison plot saved to {save_path}")
        
        plt.close()
        
        return plt.gcf()
    
    def evaluate_feature_importance(self, top_n=20, save_path='feature_importance.png'):
        """Evaluate and plot feature importance across models.
        
        Args:
            top_n: Number of top features to display
            save_path: Path to save the plot
            
        Returns:
            DataFrame of aggregated feature importance
        """
        if self.results is None:
            logger.error("No results to evaluate")
            return None
            
        logger.info("Evaluating feature importance...")
        
        # Collect feature importance from all models
        feature_importance = {}
        
        for model_name, model_results in self.results.items():
            if 'feature_importance' in model_results:
                model_feature_importance = model_results['feature_importance']
                
                # Convert to DataFrame if it's a dictionary
                if isinstance(model_feature_importance, dict):
                    features = model_feature_importance.get('feature', {})
                    importances = model_feature_importance.get('importance', {})
                    
                    # Convert dictionaries to lists if necessary
                    if isinstance(features, dict):
                        features = list(features.values())
                    if isinstance(importances, dict):
                        importances = list(importances.values())
                    
                    fi_df = pd.DataFrame({
                        'feature': features,
                        'importance': importances
                    })
                else:
                    fi_df = pd.DataFrame(model_feature_importance)
                
                # Normalize importance
                if not fi_df.empty and 'importance' in fi_df.columns:
                    fi_df['importance'] = fi_df['importance'] / fi_df['importance'].sum()
                    feature_importance[model_name] = fi_df
        
        if not feature_importance:
            logger.warning("No feature importance data available")
            return None
        
        # Aggregate feature importance across models
        all_features = set()
        for model_name, fi_df in feature_importance.items():
            all_features.update(fi_df['feature'])
        
        # Create aggregated DataFrame
        agg_data = []
        
        for feature in all_features:
            feature_data = {
                'feature': feature,
                'avg_importance': 0,
                'count': 0
            }
            
            for model_name, fi_df in feature_importance.items():
                model_fi = fi_df[fi_df['feature'] == feature]
                if not model_fi.empty:
                    feature_data[model_name] = model_fi['importance'].values[0]
                    feature_data['avg_importance'] += model_fi['importance'].values[0]
                    feature_data['count'] += 1
            
            if feature_data['count'] > 0:
                feature_data['avg_importance'] /= feature_data['count']
                agg_data.append(feature_data)
        
        agg_df = pd.DataFrame(agg_data)
        
        # Sort by average importance
        agg_df = agg_df.sort_values('avg_importance', ascending=False)
        
        # Get top features
        top_features = agg_df.head(top_n)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot average importance
        plt.barh(top_features['feature'], top_features['avg_importance'])
        
        plt.title(f'Top {top_n} Features by Average Importance')
        plt.xlabel('Average Importance')
        plt.ylabel('Feature')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Save figure
        save_path = os.path.join(RESULTS_DIR, save_path)
        plt.savefig(save_path)
        logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.close()
        
        return agg_df
    
    def evaluate_directional_accuracy(self, save_path='directional_accuracy.png'):
        """Evaluate and plot directional accuracy for all models.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Dictionary of directional accuracy metrics for each model
        """
        if self.results is None:
            logger.error("No results to evaluate")
            return None
            
        logger.info("Evaluating directional accuracy...")
        
        # Collect directional accuracy data
        directional_data = {}
        
        for model_name, model_results in self.results.items():
            if 'test_metrics' in model_results and 'test_pred' in model_results:
                test_pred = model_results['test_pred']
                
                if 'test_actual' in model_results:
                    test_actual = model_results['test_actual']
                elif model_name == self.best_model_name:
                    # If this is the best model, save its actual values for other models
                    test_actual = model_results.get('test_actual', [])
                else:
                    # Skip if no actual values
                    continue
                
                # Calculate directional accuracy metrics
                if test_pred and test_actual:
                    y_true_direction = np.sign(np.diff(np.append([0], test_actual)))
                    y_pred_direction = np.sign(np.diff(np.append([0], test_pred)))
                    
                    acc = accuracy_score(y_true_direction, y_pred_direction)
                    precision = precision_score(y_true_direction, y_pred_direction, average='weighted', zero_division=0)
                    recall = recall_score(y_true_direction, y_pred_direction, average='weighted', zero_division=0)
                    f1 = f1_score(y_true_direction, y_pred_direction, average='weighted', zero_division=0)
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_true_direction, y_pred_direction)
                    
                    directional_data[model_name] = {
                        'accuracy': acc,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'confusion_matrix': cm.tolist()
                    }
        
        if not directional_data:
            logger.warning("No directional accuracy data available")
            return None
        
        # Create comparison plot
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        data = []
        for model_name, model_metrics in directional_data.items():
            for metric in metrics:
                data.append({
                    'Model': model_name,
                    'Metric': metric.capitalize(),
                    'Value': model_metrics[metric]
                })
        
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Create grouped bar chart
        ax = sns.barplot(x='Model', y='Value', hue='Metric', data=df)
        
        plt.title('Directional Accuracy Metrics by Model')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.legend(title='Metric')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(RESULTS_DIR, save_path)
        plt.savefig(save_path)
        logger.info(f"Directional accuracy plot saved to {save_path}")
        
        plt.close()
        
        # Create confusion matrix plots
        for model_name, model_metrics in directional_data.items():
            cm = np.array(model_metrics['confusion_matrix'])
            
            plt.figure(figsize=(8, 6))
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            
            plt.title(f'Confusion Matrix for {model_name}')
            plt.ylabel('True Direction')
            plt.xlabel('Predicted Direction')
            
            # Save figure
            cm_save_path = os.path.join(RESULTS_DIR, f'confusion_matrix_{model_name}.png')
            plt.savefig(cm_save_path)
            logger.info(f"Confusion matrix plot for {model_name} saved to {cm_save_path}")
            
            plt.close()
        
        return directional_data
    
    def create_summary_report(self, save_path='model_evaluation_summary.md'):
        """Create a summary report of model evaluation.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            String containing the summary report
        """
        if self.results is None:
            logger.error("No results to summarize")
            return None
            
        logger.info("Creating summary report...")
        
        # Initialize report
        report = f"# Bitcoin Price Prediction: Model Evaluation Summary\n\n"
        report += f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add basic statistics
        report += f"## Model Comparison\n\n"
        report += f"### Performance Metrics\n\n"
        
        # Create performance table
        metrics_table = "| Model | RMSE | MAE | R² | Directional Accuracy | F1 Score |\n"
        metrics_table += "| ----- | ---- | --- | -- | ------------------- | -------- |\n"
        
        for model_name, model_results in self.results.items():
            if 'test_metrics' in model_results:
                metrics = model_results['test_metrics']
                metrics_table += f"| {model_name} | "
                metrics_table += f"{metrics.get('RMSE', 'N/A'):.4f} | "
                metrics_table += f"{metrics.get('MAE', 'N/A'):.4f} | "
                metrics_table += f"{metrics.get('R2', 'N/A'):.4f} | "
                metrics_table += f"{metrics.get('Directional Accuracy', 'N/A'):.4f} | "
                metrics_table += f"{metrics.get('F1 Score', 'N/A'):.4f} |\n"
        
        report += metrics_table + "\n"
        
        # Add best model information
        best_model_rmse = self.compare_models(metric='RMSE', ascending=True)
        best_model_dir_acc = self.compare_models(metric='Directional Accuracy', ascending=False)
        
        report += f"### Best Models\n\n"
        report += f"- Best model by RMSE: **{list(best_model_rmse.keys())[0]}** (RMSE: {list(best_model_rmse.values())[0]:.4f})\n"
        report += f"- Best model by Directional Accuracy: **{list(best_model_dir_acc.keys())[0]}** (Accuracy: {list(best_model_dir_acc.values())[0]:.4f})\n\n"
        
        # Add feature importance section
        report += f"## Feature Importance\n\n"
        report += f"The top 10 most important features across all models:\n\n"
        
        # Aggregate feature importance
        agg_df = self.evaluate_feature_importance(top_n=10)
        
        if agg_df is not None and not agg_df.empty:
            feature_table = "| Feature | Average Importance |\n"
            feature_table += "| ------- | ----------------- |\n"
            
            for _, row in agg_df.head(10).iterrows():
                feature_table += f"| {row['feature']} | {row['avg_importance']:.4f} |\n"
            
            report += feature_table + "\n"
        else:
            report += "Feature importance analysis not available.\n\n"
        
        # Add conclusion and recommendations
        report += f"## Conclusion and Recommendations\n\n"
        
        # Determine best overall model
        best_models = list(best_model_rmse.keys())[:3]
        
        report += f"Based on the evaluation results, the following models performed best for Bitcoin price prediction:\n\n"
        
        for i, model in enumerate(best_models):
            if model in self.results and 'test_metrics' in self.results[model]:
                metrics = self.results[model]['test_metrics']
                report += f"{i+1}. **{model}**\n"
                report += f"   - RMSE: {metrics.get('RMSE', 'N/A'):.4f}\n"
                report += f"   - Directional Accuracy: {metrics.get('Directional Accuracy', 'N/A'):.4f}\n"
                report += f"   - F1 Score: {metrics.get('F1 Score', 'N/A'):.4f}\n\n"
        
        # Add final recommendations
        report += "### Recommendations\n\n"
        
        if best_models:
            best_model = best_models[0]
            report += f"1. The **{best_model}** model is recommended for production use based on overall performance.\n"
            report += f"2. Ensemble models generally performed better, suggesting that combining predictions from multiple models is beneficial.\n"
            report += f"3. Important features like technical indicators and price momentum should be monitored closely for trading decisions.\n"
            report += f"4. Models should be regularly retrained as new data becomes available to maintain accuracy.\n"
        
        # Save report
        save_path = os.path.join(RESULTS_DIR, save_path)
        with open(save_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Summary report saved to {save_path}")
        
        return report
    
    def run_evaluation(self, model_type=None):
        """Run all evaluation methods.
        
        Args:
            model_type: Type of models to evaluate (traditional, deep_learning, ensemble, or None for all)
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Running full model evaluation...")
        
        # Load results
        self.load_results(model_type)
        
        if self.results is None:
            logger.error("No results found for evaluation")
            return None
        
        # Compare models
        rmse_comparison = self.compare_models(metric='RMSE')
        dir_acc_comparison = self.compare_models(metric='Directional Accuracy', ascending=False)
        
        # Plot comparisons
        self.plot_model_comparison()
        self.plot_prediction_comparison()
        
        # Evaluate feature importance
        feature_importance = self.evaluate_feature_importance()
        
        # Evaluate directional accuracy
        directional_accuracy = self.evaluate_directional_accuracy()
        
        # Create summary report
        summary_report = self.create_summary_report()
        
        # Compile all evaluation results
        evaluation_results = {
            'rmse_comparison': rmse_comparison,
            'dir_acc_comparison': dir_acc_comparison,
            'feature_importance': feature_importance.to_dict() if feature_importance is not None else None,
            'directional_accuracy': directional_accuracy
        }
        
        logger.info("Evaluation completed")
        
        return evaluation_results


def evaluate_models(model_type=None):
    """Main function to evaluate Bitcoin price prediction models.
    
    Args:
        model_type: Type of models to evaluate (traditional, deep_learning, ensemble, or None for all)
        
    Returns:
        Dictionary containing evaluation results
    """
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Run evaluation
    results = evaluator.run_evaluation(model_type)
    
    return results


if __name__ == "__main__":
    # This allows running the module directly for testing
    print("Evaluating Bitcoin price prediction models...")
    results = evaluate_models()
    
    if results:
        print("Evaluation completed successfully")
        print("\nBest model by RMSE:")
        best_model = list(results['rmse_comparison'].keys())[0]
        best_rmse = list(results['rmse_comparison'].values())[0]
        print(f"{best_model}: {best_rmse:.4f}")
        
        print("\nBest model by Directional Accuracy:")
        best_model = list(results['dir_acc_comparison'].keys())[0]
        best_acc = list(results['dir_acc_comparison'].values())[0]
        print(f"{best_model}: {best_acc:.4f}")
    else:
        print("Failed to evaluate models")