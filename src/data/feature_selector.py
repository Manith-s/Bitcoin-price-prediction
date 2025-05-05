"""Feature selection module for Bitcoin price prediction"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor

class FeatureSelector:
    """Class for automated feature selection."""
    
    def __init__(self, data, target_col='close'):
        """Initialize feature selector.
        
        Args:
            data: DataFrame with features
            target_col: Target column to predict
        """
        self.data = data
        self.target_col = target_col
    
    def select_best_features(self, method='random_forest', n_features=20):
        """Select the best features.
        
        Args:
            method: Feature selection method
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        X = self.data.drop(self.target_col, axis=1)
        y = self.data[self.target_col]
        
        if method == 'f_regression':
            selector = SelectKBest(f_regression, k=n_features)
            selector.fit(X, y)
            mask = selector.get_support()
            return list(X.columns[mask])
            
        elif method == 'rfe':
            estimator = RandomForestRegressor(n_estimators=100)
            selector = RFE(estimator, n_features_to_select=n_features)
            selector.fit(X, y)
            mask = selector.get_support()
            return list(X.columns[mask])
            
        elif method == 'random_forest':
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X, y)
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1][:n_features]
            return list(X.columns[indices])
        
        else:
            raise ValueError(f"Unknown method: {method}")