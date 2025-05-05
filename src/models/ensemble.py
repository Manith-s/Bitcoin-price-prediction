"""Ensemble models for Bitcoin price prediction"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import VotingRegressor, StackingRegressor

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import FEATURED_DATA_PATH, ENSEMBLE_MODELS
from src.models.traditional import TraditionalModels
from src.utils.helpers import (
    logger, save_model, load_model, save_results, calculate_metrics,
    plot_price_prediction
)

class EnsembleModels:
    """Class for training ensemble models for Bitcoin price prediction."""
    
    def __init__(self, data_file=FEATURED_DATA_PATH, target_col='close'):
        """Initialize the ensemble model trainer.
        
        Args:
            data_file: Path to the featured data file
            target_col: Target column to predict
        """
        self.data_file = data_file
        self.target_col = target_col
        self.traditional_trainer = TraditionalModels(data_file, target_col)
        self.models = {}
        self.results = {}
    
    def load_base_models(self, models_to_ensemble=ENSEMBLE_MODELS):
        """Load or train base models for ensemble.
        
        Args:
            models_to_ensemble: List of models to include in the ensemble
            
        Returns:
            Dictionary of base models
        """
        logger.info("Loading or training base models for ensemble...")
        
        # Load data and prepare for traditional models
        self.traditional_trainer.load_data()
        self.traditional_trainer.prepare_data()
        
        base_models = {}
        
        # Train or load each model
        for model_name in models_to_ensemble:
            if model_name in self.traditional_trainer.models:
                # Model already trained
                base_models[model_name] = self.traditional_trainer.models[model_name]
            else:
                # Try to load model from disk
                model = load_model(model_name.lower())
                
                if model is not None:
                    base_models[model_name] = model
                    self.traditional_trainer.models[model_name] = model
                else:
                    # Train model
                    logger.info(f"Training {model_name} model...")
                    
                    if model_name == 'RandomForest':
                        result = self.traditional_trainer.train_random_forest()
                    elif model_name == 'GradientBoosting':
                        result = self.traditional_trainer.train_gradient_boosting()
                    elif model_name == 'XGBoost':
                        result = self.traditional_trainer.train_xgboost()
                    elif model_name == 'LightGBM':
                        result = self.traditional_trainer.train_lightgbm()
                    elif model_name == 'CatBoost':
                        result = self.traditional_trainer.train_catboost()
                    else:
                        logger.warning(f"Unknown model: {model_name}")
                        continue
                    
                    if result is not None:
                        model, _ = result
                        base_models[model_name] = model
        
        logger.info(f"Loaded {len(base_models)} base models for ensemble")
        
        return base_models
    
    def train_voting_ensemble(self, models_to_ensemble=ENSEMBLE_MODELS, weights=None):
        """Train a voting ensemble model.
        
        Args:
            models_to_ensemble: List of models to include in the ensemble
            weights: List of weights for each model (optional)
            
        Returns:
            Tuple of (model, results) or None if an error occurred
        """
        if not self.traditional_trainer.X_train is not None:
            self.traditional_trainer.load_data()
            self.traditional_trainer.prepare_data()
            
        logger.info("Training voting ensemble model...")
        
        try:
            # Load base models
            base_models = self.load_base_models(models_to_ensemble)
            
            if not base_models:
                logger.error("No base models available for ensemble")
                return None
            
            # Define estimators for voting
            estimators = [(name, model) for name, model in base_models.items()]
            
            # Create voting ensemble
            voting_model = VotingRegressor(estimators=estimators, weights=weights)
            
            # Fit the ensemble model
            voting_model.fit(self.traditional_trainer.X_train, self.traditional_trainer.y_train)
            
            # Make predictions
            train_pred = voting_model.predict(self.traditional_trainer.X_train)
            val_pred = voting_model.predict(self.traditional_trainer.X_val)
            test_pred = voting_model.predict(self.traditional_trainer.X_test)
            
            # Calculate metrics
            train_metrics = calculate_metrics(self.traditional_trainer.y_train, train_pred)
            val_metrics = calculate_metrics(self.traditional_trainer.y_val, val_pred)
            test_metrics = calculate_metrics(self.traditional_trainer.y_test, test_pred)
            
            # Store model and results
            self.models['VotingEnsemble'] = voting_model
            self.results['VotingEnsemble'] = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'train_pred': train_pred.tolist(),
                'val_pred': val_pred.tolist(),
                'test_pred': test_pred.tolist(),
                'test_actual': self.traditional_trainer.y_test.tolist(),
                'base_models': list(base_models.keys()),
                'weights': weights
            }
            
            # Save model
            save_model(voting_model, 'voting_ensemble')
            
            logger.info(f"Voting ensemble model trained, test RMSE: {test_metrics['RMSE']:.4f}")
            logger.info(f"Directional Accuracy: {test_metrics['Directional Accuracy']:.4f}")
            
            return voting_model, self.results['VotingEnsemble']
            
        except Exception as e:
            logger.error(f"Error training voting ensemble model: {str(e)}")
            return None
    
    def train_stacking_ensemble(self, models_to_ensemble=ENSEMBLE_MODELS, meta_model=None):
        """Train a stacking ensemble model.
        
        Args:
            models_to_ensemble: List of models to include in the ensemble
            meta_model: Model to use as meta-learner (optional)
            
        Returns:
            Tuple of (model, results) or None if an error occurred
        """
        if not self.traditional_trainer.X_train is not None:
            self.traditional_trainer.load_data()
            self.traditional_trainer.prepare_data()
            
        logger.info("Training stacking ensemble model...")
        
        try:
            # Load base models
            base_models = self.load_base_models(models_to_ensemble)
            
            if not base_models:
                logger.error("No base models available for ensemble")
                return None
            
            # Define estimators for stacking
            estimators = [(name, model) for name, model in base_models.items()]
            
            # Define meta-learner
            if meta_model is None:
                meta_model = Ridge(alpha=1.0)
            
            # Create stacking ensemble
            stacking_model = StackingRegressor(
                estimators=estimators,
                final_estimator=meta_model,
                cv=5
            )
            
            # Fit the ensemble model
            stacking_model.fit(self.traditional_trainer.X_train, self.traditional_trainer.y_train)
            
            # Make predictions
            train_pred = stacking_model.predict(self.traditional_trainer.X_train)
            val_pred = stacking_model.predict(self.traditional_trainer.X_val)
            test_pred = stacking_model.predict(self.traditional_trainer.X_test)
            
            # Calculate metrics
            train_metrics = calculate_metrics(self.traditional_trainer.y_train, train_pred)
            val_metrics = calculate_metrics(self.traditional_trainer.y_val, val_pred)
            test_metrics = calculate_metrics(self.traditional_trainer.y_test, test_pred)
            
            # Store model and results
            self.models['StackingEnsemble'] = stacking_model
            self.results['StackingEnsemble'] = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'train_pred': train_pred.tolist(),
                'val_pred': val_pred.tolist(),
                'test_pred': test_pred.tolist(),
                'test_actual': self.traditional_trainer.y_test.tolist(),
                'base_models': list(base_models.keys()),
                'meta_model': meta_model.__class__.__name__
            }
            
            # Save model
            save_model(stacking_model, 'stacking_ensemble')
            
            logger.info(f"Stacking ensemble model trained, test RMSE: {test_metrics['RMSE']:.4f}")
            logger.info(f"Directional Accuracy: {test_metrics['Directional Accuracy']:.4f}")
            
            return stacking_model, self.results['StackingEnsemble']
            
        except Exception as e:
            logger.error(f"Error training stacking ensemble model: {str(e)}")
            return None
    
    def train_blending_ensemble(self, models_to_ensemble=ENSEMBLE_MODELS):
        """Train a custom blending ensemble model.
        
        This creates a meta-model that combines the predictions of base models
        through a separate validation step (manual stacking).
        
        Args:
            models_to_ensemble: List of models to include in the ensemble
            
        Returns:
            Tuple of (model, results) or None if an error occurred
        """
        if not self.traditional_trainer.X_train is not None:
            self.traditional_trainer.load_data()
            self.traditional_trainer.prepare_data()
            
        logger.info("Training custom blending ensemble model...")
        
        try:
            # Load base models
            base_models = self.load_base_models(models_to_ensemble)
            
            if not base_models:
                logger.error("No base models available for ensemble")
                return None
            
            # Generate base model predictions for the validation set
            val_predictions = []
            model_names = []
            
            for name, model in base_models.items():
                val_pred = model.predict(self.traditional_trainer.X_val)
                val_predictions.append(val_pred.reshape(-1, 1))
                model_names.append(name)
            
            # Combine predictions
            X_meta = np.hstack(val_predictions)
            
            # Train a meta-model on validation predictions
            meta_model = LinearRegression()
            meta_model.fit(X_meta, self.traditional_trainer.y_val)
            
            # Generate base model predictions for each dataset
            train_predictions = []
            val_predictions = []
            test_predictions = []
            
            for name, model in base_models.items():
                train_pred = model.predict(self.traditional_trainer.X_train)
                val_pred = model.predict(self.traditional_trainer.X_val)
                test_pred = model.predict(self.traditional_trainer.X_test)
                
                train_predictions.append(train_pred.reshape(-1, 1))
                val_predictions.append(val_pred.reshape(-1, 1))
                test_predictions.append(test_pred.reshape(-1, 1))
            
            # Combine predictions
            X_train_meta = np.hstack(train_predictions)
            X_val_meta = np.hstack(val_predictions)
            X_test_meta = np.hstack(test_predictions)
            
            # Make final predictions
            train_pred = meta_model.predict(X_train_meta)
            val_pred = meta_model.predict(X_val_meta)
            test_pred = meta_model.predict(X_test_meta)
            
            # Calculate metrics
            train_metrics = calculate_metrics(self.traditional_trainer.y_train, train_pred)
            val_metrics = calculate_metrics(self.traditional_trainer.y_val, val_pred)
            test_metrics = calculate_metrics(self.traditional_trainer.y_test, test_pred)
            
            # Create a blending ensemble model object
            blending_ensemble = {
                'base_models': base_models,
                'meta_model': meta_model,
                'model_names': model_names
            }
            
            # Store model and results
            self.models['BlendingEnsemble'] = blending_ensemble
            self.results['BlendingEnsemble'] = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'train_pred': train_pred.tolist(),
                'val_pred': val_pred.tolist(),
                'test_pred': test_pred.tolist(),
                'test_actual': self.traditional_trainer.y_test.tolist(),
                'base_models': model_names,
                'model_coefficients': {
                    'intercept': float(meta_model.intercept_),
                    'coefficients': meta_model.coef_.tolist()
                }
            }
            
            # Save model
            save_model(blending_ensemble, 'blending_ensemble')
            
            logger.info(f"Blending ensemble model trained, test RMSE: {test_metrics['RMSE']:.4f}")
            logger.info(f"Directional Accuracy: {test_metrics['Directional Accuracy']:.4f}")
            
            return blending_ensemble, self.results['BlendingEnsemble']
            
        except Exception as e:
            logger.error(f"Error training blending ensemble model: {str(e)}")
            return None
    
    def train_all_models(self):
        """Train all ensemble models.
        
        Returns:
            Dictionary of models and results
        """
        logger.info("Training all ensemble models...")
        
        # Train each model
        self.train_voting_ensemble()
        self.train_stacking_ensemble()
        self.train_blending_ensemble()
        
        # Save results
        save_results(self.results, 'ensemble_models_results')
        
        logger.info("All ensemble models trained successfully")
        
        return self.models, self.results
    
    def generate_visualizations(self, output_dir='results'):
        """Generate visualizations for the trained models.
        
        Args:
            output_dir: Directory to save visualizations
        """
        if not self.models or not self.results:
            logger.error("No models or results to visualize")
            return
            
        logger.info("Generating visualizations for ensemble models...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations for each model
        for model_name, results in self.results.items():
            # Price prediction
            if 'test_pred' in results and 'test_actual' in results:
                plot_price_prediction(
                    results['test_actual'], results['test_pred'], 
                    title=f'{model_name} Price Prediction',
                    save_path=os.path.join(output_dir, f'price_prediction_{model_name}.png')
                )
            
            # Model coefficients for blending ensemble
            if model_name == 'BlendingEnsemble' and 'model_coefficients' in results:
                # Plot coefficients
                import matplotlib.pyplot as plt
                
                coefs = results['model_coefficients']['coefficients']
                model_names = results['base_models']
                
                plt.figure(figsize=(10, 6))
                plt.bar(model_names, coefs)
                plt.title('Model Weights in Blending Ensemble')
                plt.xlabel('Model')
                plt.ylabel('Weight')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'blending_ensemble_weights.png'))
                plt.close()
        
        logger.info("Visualizations generated successfully")


def train_ensemble_models(data_file=FEATURED_DATA_PATH, target_col='close'):
    """Main function to train ensemble models.
    
    Args:
        data_file: Path to the featured data file
        target_col: Target column to predict
        
    Returns:
        Dictionary of models and results
    """
    # Initialize ensemble trainer
    trainer = EnsembleModels(data_file, target_col)
    
    # Train all models
    models, results = trainer.train_all_models()
    
    # Generate visualizations
    trainer.generate_visualizations()
    
    return models, results


if __name__ == "__main__":
    # This allows running the module directly for testing
    import matplotlib.pyplot as plt
    
    print("Training ensemble models...")
    models, results = train_ensemble_models()
    
    if models:
        print("Models trained successfully")
        print("\nModel performance (Test RMSE):")
        for model_name, result in results.items():
            print(f"{model_name}: {result['test_metrics']['RMSE']:.4f}")
    else:
        print("Failed to train models")