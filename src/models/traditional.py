"""Traditional machine learning models for Bitcoin price prediction"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import FEATURED_DATA_PATH, RANDOM_STATE, TEST_SIZE, VAL_SIZE
from src.utils.helpers import (
    logger, save_model, save_results, calculate_metrics, 
    plot_feature_importance, plot_price_prediction, plot_confusion_matrix
)

class TraditionalModels:
    """Class for training traditional ML models for Bitcoin price prediction."""
    
    def __init__(self, data_file=FEATURED_DATA_PATH, target_col='close'):
        """Initialize the model trainer.
        
        Args:
            data_file: Path to the featured data file
            target_col: Target column to predict
        """
        self.data_file = data_file
        self.target_col = target_col
        self.data = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.models = {}
        self.results = {}
    
    def load_data(self):
        """Load the data from CSV file.
        
        Returns:
            DataFrame of loaded data or None if the file doesn't exist
        """
        try:
            self.data = pd.read_csv(self.data_file, index_col=0, parse_dates=True)
            logger.info(f"Data loaded from {self.data_file}, shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
    
    def prepare_data(self):
        """Prepare data for model training.
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test) or None if an error occurred
        """
        if self.data is None:
            logger.error("No data to prepare")
            return None
            
        logger.info("Preparing data for model training...")
        
        # Make sure target column exists
        if self.target_col not in self.data.columns:
            logger.error(f"Target column '{self.target_col}' not found in data")
            return None
        
        # Create target
        X = self.data.drop(self.target_col, axis=1)
        y = self.data[self.target_col]
        
        # Split data into training, validation, and test sets
        # Using time series split to maintain temporal order
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, shuffle=False
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=VAL_SIZE / (1 - TEST_SIZE), shuffle=False
        )
        
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Validation set shape: {X_val.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train_random_forest(self, params=None):
        """Train a Random Forest model.
        
        Args:
            params: Dictionary of model parameters
            
        Returns:
            Tuple of (model, results) or None if an error occurred
        """
        if self.X_train is None or self.y_train is None:
            logger.error("Data not prepared for training")
            return None
            
        logger.info("Training Random Forest model...")
        
        # Default parameters
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': RANDOM_STATE,
                'n_jobs': -1,
                'verbose': 1
            }
        
        # Create model
        model = RandomForestRegressor(**params)
        
        try:
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            train_pred = model.predict(self.X_train)
            val_pred = model.predict(self.X_val)
            test_pred = model.predict(self.X_test)
            
            # Calculate metrics
            train_metrics = calculate_metrics(self.y_train, train_pred)
            val_metrics = calculate_metrics(self.y_val, val_pred)
            test_metrics = calculate_metrics(self.y_test, test_pred)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Store model and results
            self.models['RandomForest'] = model
            self.results['RandomForest'] = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'feature_importance': feature_importance.to_dict(),
                'train_pred': train_pred.tolist(),
                'val_pred': val_pred.tolist(),
                'test_pred': test_pred.tolist(),
                'test_actual': self.y_test.tolist()
            }
            
            # Save model
            save_model(model, 'random_forest')
            
            logger.info(f"Random Forest model trained, test RMSE: {test_metrics['RMSE']:.4f}")
            logger.info(f"Directional Accuracy: {test_metrics['Directional Accuracy']:.4f}")
            
            return model, self.results['RandomForest']
            
        except Exception as e:
            logger.error(f"Error training Random Forest model: {str(e)}")
            return None
    
    def train_gradient_boosting(self, params=None):
        """Train a Gradient Boosting model.
        
        Args:
            params: Dictionary of model parameters
            
        Returns:
            Tuple of (model, results) or None if an error occurred
        """
        if self.X_train is None or self.y_train is None:
            logger.error("Data not prepared for training")
            return None
            
        logger.info("Training Gradient Boosting model...")
        
        # Default parameters
        if params is None:
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': RANDOM_STATE,
                'verbose': 1
            }
        
        # Create model
        model = GradientBoostingRegressor(**params)
        
        try:
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            train_pred = model.predict(self.X_train)
            val_pred = model.predict(self.X_val)
            test_pred = model.predict(self.X_test)
            
            # Calculate metrics
            train_metrics = calculate_metrics(self.y_train, train_pred)
            val_metrics = calculate_metrics(self.y_val, val_pred)
            test_metrics = calculate_metrics(self.y_test, test_pred)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Store model and results
            self.models['GradientBoosting'] = model
            self.results['GradientBoosting'] = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'feature_importance': feature_importance.to_dict(),
                'train_pred': train_pred.tolist(),
                'val_pred': val_pred.tolist(),
                'test_pred': test_pred.tolist(),
                'test_actual': self.y_test.tolist()
            }
            
            # Save model
            save_model(model, 'gradient_boosting')
            
            logger.info(f"Gradient Boosting model trained, test RMSE: {test_metrics['RMSE']:.4f}")
            logger.info(f"Directional Accuracy: {test_metrics['Directional Accuracy']:.4f}")
            
            return model, self.results['GradientBoosting']
            
        except Exception as e:
            logger.error(f"Error training Gradient Boosting model: {str(e)}")
            return None
    
    def train_xgboost(self, params=None):
        """Train an XGBoost model.
        
        Args:
            params: Dictionary of model parameters
            
        Returns:
            Tuple of (model, results) or None if an error occurred
        """
        if self.X_train is None or self.y_train is None:
            logger.error("Data not prepared for training")
            return None
            
        logger.info("Training XGBoost model...")
        
        # Default parameters
        if params is None:
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'gamma': 0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'random_state': RANDOM_STATE,
                'n_jobs': -1,
                'verbosity': 1
            }
        
        try:
            # Create model
            model = xgb.XGBRegressor(**params)
            
            # Train model
            model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                callbacks=[xgb.callback.EarlyStopping(rounds=20)],
                verbose=True
            )
            
            # Make predictions
            train_pred = model.predict(self.X_train)
            val_pred = model.predict(self.X_val)
            test_pred = model.predict(self.X_test)
            
            # Calculate metrics
            train_metrics = calculate_metrics(self.y_train, train_pred)
            val_metrics = calculate_metrics(self.y_val, val_pred)
            test_metrics = calculate_metrics(self.y_test, test_pred)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Store model and results
            self.models['XGBoost'] = model
            self.results['XGBoost'] = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'feature_importance': feature_importance.to_dict(),
                'train_pred': train_pred.tolist(),
                'val_pred': val_pred.tolist(),
                'test_pred': test_pred.tolist(),
                'test_actual': self.y_test.tolist(),
                'best_iteration': model.best_iteration
            }
            
            # Save model
            save_model(model, 'xgboost')
            
            logger.info(f"XGBoost model trained, test RMSE: {test_metrics['RMSE']:.4f}")
            logger.info(f"Directional Accuracy: {test_metrics['Directional Accuracy']:.4f}")
            
            return model, self.results['XGBoost']
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {str(e)}")
            return None
    
    def train_lightgbm(self, params=None):
        """Train a LightGBM model.
        
        Args:
            params: Dictionary of model parameters
            
        Returns:
            Tuple of (model, results) or None if an error occurred
        """
        if self.X_train is None or self.y_train is None:
            logger.error("Data not prepared for training")
            return None
            
        logger.info("Training LightGBM model...")
        
        # Default parameters
        if params is None:
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'regression',
                'random_state': RANDOM_STATE,
                'n_jobs': -1,
                'verbose': 1
            }
        
        try:
            # Create model
            model = lgb.LGBMRegressor(**params)
            
            # Train model
            model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(stopping_rounds=20)],
                verbose=True
            )

            # Make predictions
            train_pred = model.predict(self.X_train)
            val_pred = model.predict(self.X_val)
            test_pred = model.predict(self.X_test)
            
            # Calculate metrics
            train_metrics = calculate_metrics(self.y_train, train_pred)
            val_metrics = calculate_metrics(self.y_val, val_pred)
            test_metrics = calculate_metrics(self.y_test, test_pred)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Store model and results
            self.models['LightGBM'] = model
            self.results['LightGBM'] = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'feature_importance': feature_importance.to_dict(),
                'train_pred': train_pred.tolist(),
                'val_pred': val_pred.tolist(),
                'test_pred': test_pred.tolist(),
                'test_actual': self.y_test.tolist(),
                'best_iteration': model.best_iteration_
            }
            
            # Save model
            save_model(model, 'lightgbm')
            
            logger.info(f"LightGBM model trained, test RMSE: {test_metrics['RMSE']:.4f}")
            logger.info(f"Directional Accuracy: {test_metrics['Directional Accuracy']:.4f}")
            
            return model, self.results['LightGBM']
            
        except Exception as e:
            logger.error(f"Error training LightGBM model: {str(e)}")
            return None
    
    def train_catboost(self, params=None):
        """Train a CatBoost model.
        
        Args:
            params: Dictionary of model parameters
            
        Returns:
            Tuple of (model, results) or None if an error occurred
        """
        if self.X_train is None or self.y_train is None:
            logger.error("Data not prepared for training")
            return None
            
        logger.info("Training CatBoost model...")
        
        # Default parameters
        if params is None:
            params = {
                'iterations': 100,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3,
                'random_seed': RANDOM_STATE,
                'verbose': 20
            }
        
        try:
            # Create model
            model = cb.CatBoostRegressor(**params)
            
            # Train model
            model.fit(
                self.X_train, self.y_train,
                eval_set=(self.X_val, self.y_val),
                early_stopping_rounds=20,
                verbose=True
            )
            
            # Make predictions
            train_pred = model.predict(self.X_train)
            val_pred = model.predict(self.X_val)
            test_pred = model.predict(self.X_test)
            
            # Calculate metrics
            train_metrics = calculate_metrics(self.y_train, train_pred)
            val_metrics = calculate_metrics(self.y_val, val_pred)
            test_metrics = calculate_metrics(self.y_test, test_pred)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Store model and results
            self.models['CatBoost'] = model
            self.results['CatBoost'] = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'feature_importance': feature_importance.to_dict(),
                'train_pred': train_pred.tolist(),
                'val_pred': val_pred.tolist(),
                'test_pred': test_pred.tolist(),
                'test_actual': self.y_test.tolist(),
                'best_iteration': model.best_iteration_
            }
            
            # Save model
            save_model(model, 'catboost')
            
            logger.info(f"CatBoost model trained, test RMSE: {test_metrics['RMSE']:.4f}")
            logger.info(f"Directional Accuracy: {test_metrics['Directional Accuracy']:.4f}")
            
            return model, self.results['CatBoost']
            
        except Exception as e:
            logger.error(f"Error training CatBoost model: {str(e)}")
            return None
    
    def train_all_models(self):
        """Train all traditional machine learning models.
        
        Returns:
            Dictionary of models and results
        """
        logger.info("Training all traditional models...")
        
        # Load and prepare data
        self.load_data()
        self.prepare_data()
        
        # Train each model
        self.train_random_forest()
        self.train_gradient_boosting()
        self.train_xgboost()
        self.train_lightgbm()
        self.train_catboost()
        
        # Save results
        save_results(self.results, 'traditional_models_results')
        
        logger.info("All traditional models trained successfully")
        
        return self.models, self.results
    
    def generate_visualizations(self, output_dir='results'):
        """Generate visualizations for the trained models.
        
        Args:
            output_dir: Directory to save visualizations
        """
        if not self.models or not self.results:
            logger.error("No models or results to visualize")
            return
            
        logger.info("Generating visualizations for traditional models...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations for each model
        for model_name, model in self.models.items():
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                plot_feature_importance(
                    model, self.X_train.columns, model_name, 
                    save_path=os.path.join(output_dir, f'feature_importance_{model_name}.png')
                )
            
            # Price prediction
            results = self.results[model_name]
            if 'test_pred' in results and 'test_actual' in results:
                plot_price_prediction(
                    results['test_actual'], results['test_pred'], 
                    title=f'{model_name} Price Prediction',
                    save_path=os.path.join(output_dir, f'price_prediction_{model_name}.png')
                )
                
                # Confusion matrix for directional prediction
                plot_confusion_matrix(
                    results['test_actual'], results['test_pred'],
                    title=f'{model_name} Directional Prediction',
                    save_path=os.path.join(output_dir, f'confusion_matrix_{model_name}.png')
                )
        
        logger.info("Visualizations generated successfully")


def train_traditional_models(data_file=FEATURED_DATA_PATH, target_col='close'):
    """Main function to train traditional machine learning models.
    
    Args:
        data_file: Path to the featured data file
        target_col: Target column to predict
        
    Returns:
        Dictionary of models and results
    """
    # Initialize model trainer
    trainer = TraditionalModels(data_file, target_col)
    
    # Train all models
    models, results = trainer.train_all_models()
    
    # Generate visualizations
    trainer.generate_visualizations()
    
    return models, results


if __name__ == "__main__":
    # This allows running the module directly for testing
    print("Training traditional machine learning models...")
    models, results = train_traditional_models()
    
    if models:
        print("Models trained successfully")
        print("\nModel performance (Test RMSE):")
        for model_name, result in results.items():
            print(f"{model_name}: {result['test_metrics']['RMSE']:.4f}")
    else:
        print("Failed to train models")