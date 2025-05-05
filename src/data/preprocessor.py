"""Data preprocessing module for Bitcoin price prediction"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.utils.helpers import logger, save_dataframe, load_dataframe

class DataPreprocessor:
    """Class to preprocess cryptocurrency data."""
    
    def __init__(self, input_file=RAW_DATA_PATH, output_file=PROCESSED_DATA_PATH):
        """Initialize DataPreprocessor with file paths.
        
        Args:
            input_file: Path to the raw data file
            output_file: Path to save the processed data
        """
        self.input_file = input_file
        self.output_file = output_file
        self.data = None
        self.price_scaler = None
        self.feature_scaler = None
        self.original_prices = None
    
    def load_data(self):
        """Load data from CSV file.
        
        Returns:
            DataFrame of loaded data or None if the file doesn't exist
        """
        self.data = load_dataframe(self.input_file)
        return self.data
    
    def clean_data(self):
        """Clean the data by handling missing values and outliers.
        
        Returns:
            DataFrame of cleaned data or None if no data was loaded
        """
        if self.data is None:
            logger.error("No data to clean")
            return None
            
        logger.info("Cleaning data...")
        
        # Make a copy to avoid modifying the original
        df = self.data.copy()
        
        # Handle missing values
        logger.info(f"Missing values before imputation: {df.isnull().sum().sum()}")
        
        # For OHLCV data, forward fill is a reasonable approach
        df = df.ffill()
        
        # If there are still missing values (e.g., at the beginning), use backward fill
        df = df.bfill()
        
        # Check if there are still missing values
        if df.isnull().sum().sum() > 0:
            logger.info("Using median imputation for remaining missing values")
            # Use simple imputer for any remaining NaN values
            imputer = SimpleImputer(strategy='median')
            df_values = imputer.fit_transform(df)
            df = pd.DataFrame(df_values, index=df.index, columns=df.columns)
        
        logger.info(f"Missing values after imputation: {df.isnull().sum().sum()}")
        
        # Handle outliers using Interquartile Range (IQR) method
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define bounds
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # Identify outliers
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                
                if not outliers.empty:
                    logger.info(f"Found {len(outliers)} outliers in column {col}")
                    
                    # Cap outliers instead of removing them
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
        
        self.data = df
        logger.info("Data cleaning completed")
        return df
    
    def normalize_data(self, price_scaling='minmax', feature_scaling='standard'):
        """Normalize the data.
        
        Args:
            price_scaling: Scaling method for price columns ('minmax', 'standard', 'robust', or None)
            feature_scaling: Scaling method for other features ('minmax', 'standard', 'robust', or None)
            
        Returns:
            DataFrame of normalized data or None if no data was loaded
        """
        if self.data is None:
            logger.error("No data to normalize")
            return None
            
        logger.info("Normalizing data...")
        
        df = self.data.copy()
        
        # Separate price columns and other features
        price_cols = ['open', 'high', 'low', 'close']
        available_price_cols = [col for col in price_cols if col in df.columns]
        
        # Store original prices for later reverse transformation
        self.original_prices = df[available_price_cols].copy() if available_price_cols else None
        
        # Scale price columns
        if price_scaling and available_price_cols:
            price_data = df[available_price_cols].values
            
            if price_scaling == 'minmax':
                self.price_scaler = MinMaxScaler()
            elif price_scaling == 'standard':
                self.price_scaler = StandardScaler()
            elif price_scaling == 'robust':
                self.price_scaler = RobustScaler()
            
            scaled_prices = self.price_scaler.fit_transform(price_data)
            df[available_price_cols] = scaled_prices
            
            logger.info(f"Price columns scaled using {price_scaling} scaling")
        
        # Scale other features
        feature_cols = [col for col in df.columns if col not in available_price_cols]
        
        if feature_scaling and feature_cols:
            feature_data = df[feature_cols].select_dtypes(include=[np.number]).values
            
            if feature_data.size > 0:  # Only scale if we have numerical data
                if feature_scaling == 'minmax':
                    self.feature_scaler = MinMaxScaler()
                elif feature_scaling == 'standard':
                    self.feature_scaler = StandardScaler()
                elif feature_scaling == 'robust':
                    self.feature_scaler = RobustScaler()
                
                # Get numerical feature columns
                numerical_feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
                
                # Scale numerical features
                scaled_features = self.feature_scaler.fit_transform(feature_data)
                df[numerical_feature_cols] = scaled_features
                
                logger.info(f"Feature columns scaled using {feature_scaling} scaling")
            else:
                logger.warning("No numerical feature columns to scale")
        
        self.data = df
        logger.info("Data normalization completed")
        return df
    
    def add_date_features(self):
        """Add date-based features.
        
        Returns:
            DataFrame with added date features or None if no data was loaded
        """
        if self.data is None:
            logger.error("No data to add date features to")
            return None
            
        logger.info("Adding date features...")
        
        df = self.data.copy()
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                logger.error(f"Error converting index to datetime: {str(e)}")
                return df
        
        # Extract date features
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        
        # Cyclical encoding for periodic features
        df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        df['sin_day_of_month'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['cos_day_of_month'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
        
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        
        self.data = df
        logger.info("Date features added")
        return df
    
    def add_lagged_features(self, lag_periods=[1, 2, 3, 7, 14, 30], columns=None):
        """Add lagged features for specified columns.
        
        Args:
            lag_periods: List of periods to lag
            columns: List of column names to create lags for. If None, use price columns.
            
        Returns:
            DataFrame with added lagged features or None if no data was loaded
        """
        if self.data is None:
            logger.error("No data to add lagged features to")
            return None
            
        logger.info("Adding lagged features...")
        
        df = self.data.copy()
        
        # If columns not specified, use price columns
        if columns is None:
            columns = ['close', 'high', 'low', 'volume']
            columns = [col for col in columns if col in df.columns]
        
        # Create lagged features
        for col in columns:
            for lag in lag_periods:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        self.data = df
        logger.info(f"Added {len(columns) * len(lag_periods)} lagged features")
        return df
    
    def add_rolling_features(self, window_sizes=[7, 14, 30], columns=None):
        """Add rolling statistics features for specified columns.
        
        Args:
            window_sizes: List of window sizes for rolling statistics
            columns: List of column names to create features for. If None, use price columns.
            
        Returns:
            DataFrame with added rolling features or None if no data was loaded
        """
        if self.data is None:
            logger.error("No data to add rolling features to")
            return None
            
        logger.info("Adding rolling features...")
        
        df = self.data.copy()
        
        # If columns not specified, use price columns
        if columns is None:
            columns = ['close', 'high', 'low', 'volume']
            columns = [col for col in columns if col in df.columns]
        
        # Calculate rolling statistics
        for col in columns:
            for window in window_sizes:
                # Rolling mean
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                
                # Rolling standard deviation
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                
                # Rolling min and max
                df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
                
                # Rolling median
                df[f'{col}_rolling_median_{window}'] = df[col].rolling(window=window).median()
        
        self.data = df
        logger.info(f"Added {len(columns) * len(window_sizes) * 5} rolling features")
        return df
    
    def add_returns(self, periods=[1, 2, 3, 5, 7, 14, 30]):
        """Add price returns over different periods.
        
        Args:
            periods: List of periods to calculate returns for
            
        Returns:
            DataFrame with added return features or None if no data was loaded
        """
        if self.data is None:
            logger.error("No data to add returns to")
            return None
            
        logger.info("Adding return features...")
        
        df = self.data.copy()
        
        # Check if 'close' column exists
        if 'close' not in df.columns:
            logger.warning("No 'close' column found, skipping returns calculation")
            return df
        
        # Calculate simple returns
        for period in periods:
            df[f'return_{period}d'] = df['close'].pct_change(period)
            
            # Log returns are often more suitable for financial modeling
            df[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))
        
        # Calculate volatility (standard deviation of returns)
        for period in periods:
            df[f'volatility_{period}d'] = df[f'return_{period}d'].rolling(window=period).std()
        
        self.data = df
        logger.info(f"Added {len(periods) * 3} return features")
        return df
    
    def remove_nan_values(self):
        """Remove rows with NaN values in essential columns only."""
        if self.data is None:
            logger.error("No data to remove NaN values from")
            return None
            
        logger.info("Handling rows with NaN values...")
        
        # Get original data length
        original_length = len(self.data)
        
        # List of essential columns that shouldn't have NaN values
        essential_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Only remove rows with NaN in essential columns
        self.data.dropna(subset=essential_columns, inplace=True)
        
        # For other columns, fill NaN with appropriate values
        # Fill numeric columns with their median
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col not in essential_columns:
                self.data[col] = self.data[col].fillna(self.data[col].median())
        
        # Get new data length
        new_length = len(self.data)
        
        # Log number of dropped rows
        logger.info(f"Removed {original_length - new_length} rows with NaN values in essential columns")
        
        return self.data

    def preprocess(self):
        """Perform all preprocessing steps.
        
        Returns:
            DataFrame of preprocessed data or None if an error occurred
        """
        logger.info("Starting data preprocessing...")
        
        try:
            # Load the data
            if self.load_data() is None:
                return None
            
            # Clean the data
            self.clean_data()
            
            # Add date features
            self.add_date_features()
            
            # Add lagged features
            self.add_lagged_features()
            
            # Add rolling features
            self.add_rolling_features()
            
            # Add returns
            self.add_returns()
            
            # Remove NaN values
            self.remove_nan_values()
            
            # Save the processed data
            self.save_data()
            
            logger.info("Data preprocessing completed")
            return self.data
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return None
    
    def save_data(self):
        """Save processed data to CSV file."""
        if self.data is not None:
            save_dataframe(self.data, self.output_file)
        else:
            logger.warning("No processed data to save")
    
    def inverse_transform_prices(self, scaled_prices):
        """Transform scaled prices back to original scale.
        
        Args:
            scaled_prices: Array of scaled prices
            
        Returns:
            Array of prices in original scale
        """
        if self.price_scaler is None:
            logger.warning("No price scaler found, returning input unchanged")
            return scaled_prices
            
        return self.price_scaler.inverse_transform(scaled_prices)


def preprocess_data(input_file=RAW_DATA_PATH, output_file=PROCESSED_DATA_PATH):
    """Main function to preprocess Bitcoin price data.
    
    Args:
        input_file: Path to the raw data file
        output_file: Path to save the processed data
        
    Returns:
        DataFrame of preprocessed data or None if an error occurred
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor(input_file, output_file)
    
    # Run preprocessing
    data = preprocessor.preprocess()
    
    return data


if __name__ == "__main__":
    # This allows running the module directly for testing
    print("Preprocessing Bitcoin price data...")
    data = preprocess_data()
    
    if data is not None:
        print(f"Successfully preprocessed {len(data)} records")
        print("\nSample data:")
        print(data.head())
    else:
        print("Failed to preprocess data")