"""Feature engineering module for Bitcoin price prediction"""

import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import PROCESSED_DATA_PATH, FEATURED_DATA_PATH, TECHNICAL_INDICATORS
from src.utils.helpers import logger, save_dataframe, load_dataframe

class FeatureEngineer:
    """Class to engineer features for cryptocurrency price prediction."""
    
    def __init__(self, input_file=PROCESSED_DATA_PATH, output_file=FEATURED_DATA_PATH):
        """Initialize FeatureEngineer with file paths.
        
        Args:
            input_file: Path to the processed data file
            output_file: Path to save the featured data
        """
        self.input_file = input_file
        self.output_file = output_file
        self.data = None
    
    def load_data(self):
        """Load data from CSV file.
        
        Returns:
            DataFrame of loaded data or None if the file doesn't exist
        """
        self.data = load_dataframe(self.input_file)
        return self.data
    
    def add_technical_indicators(self, indicators=TECHNICAL_INDICATORS):
        """Add technical indicators using pandas-ta.
        
        Args:
            indicators: List of indicators to add
            
        Returns:
            DataFrame with added technical indicators or None if no data was loaded
        """
        if self.data is None or self.data.empty:
            logger.error("No data (or empty DataFrame) to add technical indicators to")
            return None
            
        logger.info(f"Adding technical indicators... (DataFrame shape: {self.data.shape})")

        df = self.data.copy()
        
        # Check if OHLCV columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for technical indicators: {missing_cols}")
            logger.warning("Some indicators may not be calculated")
        
        # Add indicators based on availability
        # RSI
        if 'RSI' in indicators and 'close' in df.columns:
            try:
                # Relative Strength Index
                for period in [7, 14, 21]:
                    df[f'RSI_{period}'] = ta.rsi(df['close'], length=period)
                logger.info("Added RSI indicators")
            except Exception as e:
                logger.error(f"Error adding RSI indicators: {str(e)}")
        
        # MACD
        if 'MACD' in indicators and 'close' in df.columns:
            try:
                # Moving Average Convergence Divergence
                macd_result = ta.macd(df['close'], fast=12, slow=26, signal=9)
                
                try:
                    if isinstance(macd_result, pd.DataFrame):
                        df['MACD'] = macd_result['MACD_12_26_9']
                        df['MACD_signal'] = macd_result['MACDs_12_26_9']
                        df['MACD_hist'] = macd_result['MACDh_12_26_9']
                    elif isinstance(macd_result, (tuple, list)):
                        if len(macd_result) >= 3:
                            df['MACD'] = macd_result[0]
                            df['MACD_signal'] = macd_result[1]
                            df['MACD_hist'] = macd_result[2]
                        else:
                            logger.warning(f"MACD result tuple/list has insufficient length: {len(macd_result)}")
                    else:
                        logger.warning(f"Unexpected MACD result type: {type(macd_result)}")
                    logger.info("Added MACD indicators")
                except Exception as e:
                    logger.error(f"Error processing MACD result: {str(e)}, result type: {type(macd_result)}")
            except Exception as e:
                logger.error(f"Error calculating MACD: {str(e)}")
        
        # Bollinger Bands
        if 'Bollinger' in indicators and 'close' in df.columns:
            try:
                # Bollinger Bands
                for period in [20, 50]:
                    try:
                        bb_result = ta.bbands(df['close'], length=period, std=2)
                        
                        if isinstance(bb_result, pd.DataFrame):
                            df[f'Bollinger_upper_{period}'] = bb_result[f'BBU_{period}_2.0']
                            df[f'Bollinger_middle_{period}'] = bb_result[f'BBM_{period}_2.0']
                            df[f'Bollinger_lower_{period}'] = bb_result[f'BBL_{period}_2.0']
                        elif isinstance(bb_result, (tuple, list)):
                            if len(bb_result) >= 3:
                                df[f'Bollinger_upper_{period}'] = bb_result[0]
                                df[f'Bollinger_middle_{period}'] = bb_result[1]
                                df[f'Bollinger_lower_{period}'] = bb_result[2]
                            else:
                                logger.warning(f"Bollinger result has insufficient length: {len(bb_result)}")
                                continue
                        else:
                            logger.warning(f"Unexpected Bollinger result type: {type(bb_result)}")
                            continue
                        
                        # Bollinger Band width
                        df[f'Bollinger_width_{period}'] = (df[f'Bollinger_upper_{period}'] - df[f'Bollinger_lower_{period}']) / df[f'Bollinger_middle_{period}']
                        
                        # Bollinger %B (position within bands)
                        df[f'Bollinger_pctB_{period}'] = (df['close'] - df[f'Bollinger_lower_{period}']) / (df[f'Bollinger_upper_{period}'] - df[f'Bollinger_lower_{period}'])
                    except Exception as e:
                        logger.error(f"Error processing Bollinger Bands for period {period}: {str(e)}")
                logger.info("Added Bollinger Band indicators")
            except Exception as e:
                logger.error(f"Error adding Bollinger Band indicators: {str(e)}")
        
        # ATR
        if 'ATR' in indicators and all(col in df.columns for col in ['high', 'low', 'close']):
            try:
                # Average True Range
                for period in [7, 14, 21]:
                    atr_result = ta.atr(df['high'], df['low'], df['close'], length=period)
                    if isinstance(atr_result, pd.Series) or isinstance(atr_result, np.ndarray):
                        df[f'ATR_{period}'] = atr_result
                    elif isinstance(atr_result, (tuple, list)) and len(atr_result) > 0:
                        df[f'ATR_{period}'] = atr_result[0]
                    else:
                        logger.warning(f"Unexpected ATR result type: {type(atr_result)}")
                logger.info("Added ATR indicators")
            except Exception as e:
                logger.error(f"Error adding ATR indicators: {str(e)}")
        
        # OBV
        if 'OBV' in indicators and all(col in df.columns for col in ['close', 'volume']):
            try:
                # On-Balance Volume
                obv_result = ta.obv(df['close'], df['volume'])
                if isinstance(obv_result, pd.Series) or isinstance(obv_result, np.ndarray):
                    df['OBV'] = obv_result
                elif isinstance(obv_result, (tuple, list)) and len(obv_result) > 0:
                    df['OBV'] = obv_result[0]
                else:
                    logger.warning(f"Unexpected OBV result type: {type(obv_result)}")
                logger.info("Added OBV indicator")
            except Exception as e:
                logger.error(f"Error adding OBV indicator: {str(e)}")
        
        # Ichimoku Cloud
        if 'Ichimoku' in indicators and all(col in df.columns for col in ['high', 'low', 'close']):
            try:
                # Ichimoku Cloud
                ichimoku = ta.ichimoku(df['high'], df['low'], df['close'], tenkan=9, kijun=26, senkou=52)
                
                try:
                    # Add the ichimoku components
                    if isinstance(ichimoku, pd.DataFrame):
                        df['Ichimoku_tenkan_sen'] = ichimoku['ITS_9']
                        df['Ichimoku_kijun_sen'] = ichimoku['IKS_26']
                        df['Ichimoku_senkou_span_a'] = ichimoku['ISA_9']
                        df['Ichimoku_senkou_span_b'] = ichimoku['ISB_26']
                        df['Ichimoku_chikou_span'] = ichimoku['ICS_26']
                    elif isinstance(ichimoku, (tuple, list)):
                        if len(ichimoku) >= 5:
                            df['Ichimoku_tenkan_sen'] = ichimoku[0]
                            df['Ichimoku_kijun_sen'] = ichimoku[1]
                            df['Ichimoku_senkou_span_a'] = ichimoku[2]
                            df['Ichimoku_senkou_span_b'] = ichimoku[3]
                            df['Ichimoku_chikou_span'] = ichimoku[4]
                        else:
                            logger.warning(f"Ichimoku result has insufficient length: {len(ichimoku)}")
                    else:
                        logger.warning(f"Unexpected Ichimoku result type: {type(ichimoku)}")
                    logger.info("Added Ichimoku Cloud indicators")
                except Exception as e:
                    logger.error(f"Error processing Ichimoku result: {str(e)}")
            except Exception as e:
                logger.error(f"Error calculating Ichimoku Cloud: {str(e)}")
        
        # ROC
        if 'ROC' in indicators and 'close' in df.columns:
            try:
                # Rate of Change
                for period in [5, 10, 20, 50]:
                    roc_result = ta.roc(df['close'], length=period)
                    if isinstance(roc_result, pd.Series) or isinstance(roc_result, np.ndarray):
                        df[f'ROC_{period}'] = roc_result
                    elif isinstance(roc_result, (tuple, list)) and len(roc_result) > 0:
                        df[f'ROC_{period}'] = roc_result[0]
                    else:
                        logger.warning(f"Unexpected ROC result type: {type(roc_result)}")
                logger.info("Added ROC indicators")
            except Exception as e:
                logger.error(f"Error adding ROC indicators: {str(e)}")
        
        # Williams %R
        if 'Williams_R' in indicators and all(col in df.columns for col in ['high', 'low', 'close']):
            try:
                # Williams %R
                for period in [14, 28]:
                    willr_result = ta.willr(df['high'], df['low'], df['close'], length=period)
                    if isinstance(willr_result, pd.Series) or isinstance(willr_result, np.ndarray):
                        df[f'Williams_R_{period}'] = willr_result
                    elif isinstance(willr_result, (tuple, list)) and len(willr_result) > 0:
                        df[f'Williams_R_{period}'] = willr_result[0]
                    else:
                        logger.warning(f"Unexpected Williams_R result type: {type(willr_result)}")
                logger.info("Added Williams %R indicators")
            except Exception as e:
                logger.error(f"Error adding Williams %R indicators: {str(e)}")
        
        # MFI
        if 'MFI' in indicators and all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
            try:
                # Money Flow Index
                for period in [14, 28]:
                    mfi_result = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=period)
                    if isinstance(mfi_result, pd.Series) or isinstance(mfi_result, np.ndarray):
                        # Convert to float before assignment to avoid dtype warning
                        df[f'MFI_{period}'] = mfi_result.astype(float)
                    elif isinstance(mfi_result, (tuple, list)) and len(mfi_result) > 0:
                        df[f'MFI_{period}'] = mfi_result[0].astype(float)
                    else:
                        logger.warning(f"Unexpected MFI result type: {type(mfi_result)}")
                logger.info("Added MFI indicators")
            except Exception as e:
                logger.error(f"Error adding MFI indicators: {str(e)}")
        
        # CCI
        if 'CCI' in indicators and all(col in df.columns for col in ['high', 'low', 'close']):
            try:
                # Commodity Channel Index
                for period in [14, 28]:
                    cci_result = ta.cci(df['high'], df['low'], df['close'], length=period)
                    if isinstance(cci_result, pd.Series) or isinstance(cci_result, np.ndarray):
                        df[f'CCI_{period}'] = cci_result
                    elif isinstance(cci_result, (tuple, list)) and len(cci_result) > 0:
                        df[f'CCI_{period}'] = cci_result[0]
                    else:
                        logger.warning(f"Unexpected CCI result type: {type(cci_result)}")
                logger.info("Added CCI indicators")
            except Exception as e:
                logger.error(f"Error adding CCI indicators: {str(e)}")
        
        self.data = df
        logger.info(f"Added technical indicators, new shape: {df.shape}")
        return df
    
    def add_price_patterns(self):
        """Add candlestick pattern recognition features.
        
        Returns:
            DataFrame with added price pattern features or None if no data was loaded
        """
        if self.data is None:
            logger.error("No data to add price patterns to")
            return None
                
        logger.info("Adding price pattern features...")
        
        df = self.data.copy()
        
        # Check if OHLC columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for price patterns: {missing_cols}")
            return df
        
        try:
            # Use pandas-ta's candle pattern detection
            # Create a clean OHLC dataframe
            ohlc_df = df[['open', 'high', 'low', 'close']].copy()
            
            # Try to use pandas-ta's cdl patterns if available
            try:
                # Check for basic candlestick patterns
                patterns = {}
                
                try:
                    patterns['CDL_DOJI'] = ta.cdl_pattern(df['open'], df['high'], df['low'], df['close'], name="doji")
                except Exception as e:
                    logger.warning(f"Could not calculate DOJI pattern: {str(e)}")
                
                # Add available patterns to the dataframe
                for pattern_name, pattern_series in patterns.items():
                    df[pattern_name] = pattern_series
                
                logger.info(f"Added {len(patterns)} pattern indicators from pandas-ta")
            except Exception as e:
                logger.warning(f"Error using pandas-ta patterns: {str(e)}")
            
            # For patterns not directly available in pandas-ta, implement simple logic
            # Simple implementations of common patterns
            
            # Morning Star - three candle pattern with a small middle candle
            try:
                # First candle: bearish
                cond1 = (df['close'].shift(2) < df['open'].shift(2))
                # Second candle: small body
                cond2 = (abs(df['close'].shift(1) - df['open'].shift(1)) < 0.3 * (df['high'].shift(1) - df['low'].shift(1)))
                # Third candle: bullish
                cond3 = (df['close'] > df['open'])
                
                # Combine conditions
                df['CDL_MORNINGSTAR'] = np.where(
                    cond1 & cond2 & cond3,  # Use & for element-wise AND
                    100, 0
                )
                logger.info("Added Morning Star pattern")
            except Exception as e:
                logger.warning(f"Error calculating Morning Star pattern: {str(e)}")
            
            # Evening Star - three candle pattern with a small middle candle at the top
            try:
                # First candle: bullish
                cond1 = (df['close'].shift(2) > df['open'].shift(2))
                # Second candle: small body
                cond2 = (abs(df['close'].shift(1) - df['open'].shift(1)) < 0.3 * (df['high'].shift(1) - df['low'].shift(1)))
                # Third candle: bearish
                cond3 = (df['close'] < df['open'])
                
                # Combine conditions
                df['CDL_EVENINGSTAR'] = np.where(
                    cond1 & cond2 & cond3,  # Use & for element-wise AND
                    -100, 0
                )
                logger.info("Added Evening Star pattern")
            except Exception as e:
                logger.warning(f"Error calculating Evening Star pattern: {str(e)}")
            
            # Dark Cloud Cover - bearish reversal pattern
            try:
                # Prior candle: bullish
                cond1 = (df['close'].shift(1) > df['open'].shift(1))
                # Open above prior high
                cond2 = (df['open'] > df['high'].shift(1))
                # Close below midpoint
                cond3 = (df['close'] < (df['open'].shift(1) + df['close'].shift(1))/2)
                
                # Combine conditions
                df['CDL_DARKCLOUDCOVER'] = np.where(
                    cond1 & cond2 & cond3,  # Use & for element-wise AND
                    -100, 0
                )
                logger.info("Added Dark Cloud Cover pattern")
            except Exception as e:
                logger.warning(f"Error calculating Dark Cloud Cover pattern: {str(e)}")
            
            # Piercing - bullish reversal pattern
            try:
                # Prior candle: bearish
                cond1 = (df['close'].shift(1) < df['open'].shift(1))
                # Open below prior low
                cond2 = (df['open'] < df['low'].shift(1))
                # Close above midpoint
                cond3 = (df['close'] > (df['open'].shift(1) + df['close'].shift(1))/2)
                
                # Combine conditions
                df['CDL_PIERCING'] = np.where(
                    cond1 & cond2 & cond3,  # Use & for element-wise AND
                    100, 0
                )
                logger.info("Added Piercing pattern")
            except Exception as e:
                logger.warning(f"Error calculating Piercing pattern: {str(e)}")
            
            # Spinning Top - small body with longer wicks
            try:
                # Small body
                cond1 = (abs(df['close'] - df['open']) < 0.3 * (df['high'] - df['low']))
                # Long wicks
                cond2 = (df['high'] - df['low'] > 2 * abs(df['close'] - df['open']))
                
                # Determine if bullish or bearish
                bullish = df['close'] > df['open']
                
                # Combine conditions
                df['CDL_SPINNINGTOP'] = np.where(
                    cond1 & cond2,  # Use & for element-wise AND
                    np.where(bullish, 100, -100), 0
                )
                logger.info("Added Spinning Top pattern")
            except Exception as e:
                logger.warning(f"Error calculating Spinning Top pattern: {str(e)}")
            
            # Count patterns added
            added_patterns = [col for col in df.columns if col.startswith('CDL_')]
            logger.info(f"Added {len(added_patterns)} price pattern features total")
            
            self.data = df
            return df
            
        except Exception as e:
            logger.error(f"Error in add_price_patterns: {str(e)}")
            # Return unmodified data if error occurred
            return self.data
    
    def add_market_regime_features(self, window=20):
        """Add market regime features based on volatility and trend.
        
        Args:
            window: Window size for calculations
            
        Returns:
            DataFrame with added market regime features or None if no data was loaded
        """
        if self.data is None:
            logger.error("No data to add market regime features to")
            return None
            
        logger.info("Adding market regime features...")
        
        df = self.data.copy()
        
        # Check if close column exists
        if 'close' not in df.columns:
            logger.warning("No 'close' column found, skipping market regime features")
            return df
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate volatility
        df['volatility'] = df['returns'].rolling(window=window).std()
        
        # Calculate trend
        df['sma_short'] = df['close'].rolling(window=window//2).mean()
        df['sma_long'] = df['close'].rolling(window=window).mean()
        df['trend'] = df['sma_short'] - df['sma_long']
        
        # Define market regimes
        # 1. High volatility, uptrend
        # 2. High volatility, downtrend
        # 3. Low volatility, uptrend
        # 4. Low volatility, downtrend
        
        # Get median volatility as threshold
        vol_median = df['volatility'].median()
        
        # Classify regimes
        conditions = [
            (df['volatility'] > vol_median) & (df['trend'] > 0),
            (df['volatility'] > vol_median) & (df['trend'] <= 0),
            (df['volatility'] <= vol_median) & (df['trend'] > 0),
            (df['volatility'] <= vol_median) & (df['trend'] <= 0)
        ]
        
        choices = [1, 2, 3, 4]
        df['market_regime'] = np.select(conditions, choices, default=0)
        
        # Create one-hot encoded features
        for regime in range(1, 5):
            df[f'regime_{regime}'] = (df['market_regime'] == regime).astype(int)
        
        # Drop intermediate columns
        df.drop(['returns', 'sma_short', 'sma_long'], axis=1, inplace=True)
        
        self.data = df
        logger.info("Added market regime features")
        return df
    
    def add_custom_features(self):
        """Add custom features based on domain knowledge.
        
        Returns:
            DataFrame with added custom features or None if no data was loaded
        """
        if self.data is None:
            logger.error("No data to add custom features to")
            return None
            
        logger.info("Adding custom features...")
        
        df = self.data.copy()
        
        # Check for required columns
        if 'close' not in df.columns:
            logger.warning("No 'close' column found, skipping some custom features")
        else:
            # Price momentum features
            for period in [5, 10, 20]:
                # Momentum (current price / price n periods ago - 1)
                df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
                
                # Acceleration (momentum change)
                df[f'momentum_accel_{period}'] = df[f'momentum_{period}'] - df[f'momentum_{period}'].shift(1)
        
        # Volume-based features (if volume data is available)
        if 'volume' in df.columns and 'close' in df.columns:
            # Volume momentum
            for period in [5, 10, 20]:
                df[f'volume_momentum_{period}'] = df['volume'] / df['volume'].shift(period) - 1
            
            # Price-volume relationship
            df['price_volume_corr_5'] = df['close'].rolling(5).corr(df['volume'])
            df['price_volume_corr_10'] = df['close'].rolling(10).corr(df['volume'])
            df['price_volume_corr_20'] = df['close'].rolling(20).corr(df['volume'])
            
            # Money Flow: price * volume
            df['money_flow'] = df['close'] * df['volume']
            df['money_flow_sma_5'] = df['money_flow'].rolling(5).mean()
            df['money_flow_sma_20'] = df['money_flow'].rolling(20).mean()
        
        # Volatility features (if high/low data is available)
        if all(col in df.columns for col in ['high', 'low', 'close']):
            # True Range
            df['true_range'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    np.abs(df['high'] - df['close'].shift(1)),
                    np.abs(df['low'] - df['close'].shift(1))
                )
            )
            df['atr_5'] = df['true_range'].rolling(5).mean()
            df['atr_14'] = df['true_range'].rolling(14).mean()
        
        # Count consecutive up/down days
        if 'close' in df.columns:
            df['price_change'] = df['close'].diff()
            df['up_day'] = (df['price_change'] > 0).astype(int)
            df['down_day'] = (df['price_change'] < 0).astype(int)
            
            # Count consecutive up/down days
            df['up_streak'] = df['up_day'].groupby((df['up_day'] != df['up_day'].shift(1)).cumsum()).cumsum()
            df['down_streak'] = df['down_day'].groupby((df['down_day'] != df['down_day'].shift(1)).cumsum()).cumsum()
            
            # Reset streaks when they end
            df.loc[df['up_day'] == 0, 'up_streak'] = 0
            df.loc[df['down_day'] == 0, 'down_streak'] = 0
        
        self.data = df
        logger.info("Added custom features")
        return df
    
    def add_dimensionality_reduction(self, n_components=10, method='pca'):
        """Reduce dimensionality of features using PCA or other methods.
        
        Args:
            n_components: Number of components to keep
            method: Method to use ('pca', 'kernel_pca', etc.)
            
        Returns:
            DataFrame with added dimensionality reduction components or None if no data was loaded
        """
        if self.data is None:
            logger.error("No data to apply dimensionality reduction to")
            return None
                
        logger.info(f"Applying dimensionality reduction using {method}...")
        
        try:
            df = self.data.copy()
            
            # Separate target (close price) from features
            if 'close' in df.columns:
                target = df['close']
                features = df.drop('close', axis=1)
            else:
                features = df.copy()
            
            # Select only numeric columns
            numeric_features = features.select_dtypes(include=[np.number])
            
            if numeric_features.empty:
                logger.warning("No numeric features for dimensionality reduction")
                return df
            
            # Handle NaN values before dimensionality reduction
            # Use forward/backward fill to handle NaNs
            numeric_features = numeric_features.ffill().bfill()
            
            # Check if there are still NaN values
            if numeric_features.isnull().any().any():
                logger.warning("Cannot perform dimensionality reduction due to remaining NaN values")
                return df
            
            # Apply dimensionality reduction
            if method == 'pca':
                from sklearn.decomposition import PCA
                n_components = min(n_components, numeric_features.shape[1], numeric_features.shape[0])
                reducer = PCA(n_components=n_components)
            elif method == 'kernel_pca':
                from sklearn.decomposition import KernelPCA
                n_components = min(n_components, numeric_features.shape[1], numeric_features.shape[0])
                reducer = KernelPCA(n_components=n_components, kernel='rbf')
            else:
                logger.warning(f"Unknown method: {method}")
                return df
            
            # Fit and transform
            reduced_features = reducer.fit_transform(numeric_features)
            
            # Create new dataframe with reduced features
            reduced_df = pd.DataFrame(
                reduced_features,
                index=numeric_features.index,
                columns=[f'{method}_component_{i+1}' for i in range(reduced_features.shape[1])]
            )
            
            # Add back the target
            if 'close' in df.columns:
                reduced_df['close'] = target
            
            logger.info(f"Reduced features from {numeric_features.shape[1]} to {reduced_features.shape[1]} dimensions")
            
            # Don't replace existing data, just add the components
            for col in reduced_df.columns:
                if col != 'close':
                    df[col] = reduced_df[col]
            
            self.data = df
            return df
                
        except Exception as e:
            logger.error(f"Error applying dimensionality reduction: {str(e)}")
            return df
    
    def add_anomaly_detection_features(self):
        """Add anomaly detection features using clustering.
        
        Returns:
            DataFrame with added anomaly detection features or None if no data was loaded
        """
        if self.data is None:
            logger.error("No data to add anomaly detection features to")
            return None
                
        logger.info("Adding anomaly detection features...")
        
        try:
            df = self.data.copy()
            
            # Select key features for anomaly detection
            key_features = [
                'close', 'volume', 'RSI_14', 'volatility', 'momentum_10'
            ]
            # Filter to only include columns that exist
            key_features = [f for f in key_features if f in df.columns]
            
            if len(key_features) < 2:
                logger.warning("Not enough key features for anomaly detection")
                return df
            
            # Prepare clean data for clustering - handle NaN values first
            feature_df = df[key_features].copy()
            feature_df = feature_df.ffill().bfill()
            
            # Skip anomaly detection if we still have NaN values
            if feature_df.isnull().any().any():
                logger.warning("Cannot perform anomaly detection due to remaining NaN values")
                return df
            
            # K-means clustering
            kmeans = KMeans(n_clusters=4, random_state=42)
            df['cluster'] = kmeans.fit_predict(feature_df)
            
            # Calculate distance to cluster center (anomaly score)
            centers = kmeans.cluster_centers_
            
            # For each data point, calculate distance to its cluster center
            anomaly_scores = []
            for i in range(len(df)):
                if i < len(feature_df):  # Safety check
                    cluster_id = df.iloc[i]['cluster']
                    point = feature_df.iloc[i].values
                    if cluster_id < len(centers):  # Safety check
                        center = centers[int(cluster_id)]  # Ensure integer index
                        distance = np.linalg.norm(point - center)
                        anomaly_scores.append(distance)
                    else:
                        anomaly_scores.append(0)
                else:
                    anomaly_scores.append(0)
            
            df['anomaly_score'] = anomaly_scores
            
            # Normalize anomaly score
            if df['anomaly_score'].max() > df['anomaly_score'].min():
                df['anomaly_score'] = (df['anomaly_score'] - df['anomaly_score'].min()) / (df['anomaly_score'].max() - df['anomaly_score'].min())
            
            # Flag potential anomalies
            if len(df) > 0:
                threshold = df['anomaly_score'].quantile(0.95) if len(df) > 20 else df['anomaly_score'].median()
                df['is_anomaly'] = (df['anomaly_score'] > threshold).astype(int)
            
            logger.info("Added anomaly detection features")
            self.data = df
            return df
                
        except Exception as e:
            logger.error(f"Error adding anomaly detection features: {str(e)}")
            # Return the unmodified dataframe
            return self.data
    
    def remove_nan_values(self):
        """Handle NaN values in the data by imputing rather than removing rows.
        
        Returns:
            DataFrame with NaN values handled
        """
        if self.data is None:
            logger.error("No data to handle NaN values")
            return None
                
        logger.info("Handling NaN values...")
        
        # Get original data shape
        original_shape = self.data.shape
        
        # Rather than removing rows, use forward-fill and backward-fill to impute NaNs
        # This preserves our dataset size
        df = self.data.copy()
        
        # First try forward fill (use previous valid value)
        df = df.ffill()
        
        # Then use backward fill for any remaining NaNs (important for start of series)
        df = df.bfill()
        
        # For any remaining NaNs (rare case), fill with column median
        for col in df.columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                if pd.isna(median_val):  # If median is also NaN
                    median_val = 0
                df[col] = df[col].fillna(median_val)
        
        # Log the results
        nan_count_after = df.isnull().sum().sum()
        logger.info(f"NaN values handled. Original shape: {original_shape}, NaN count after: {nan_count_after}")
        
        self.data = df
        return df
    
    def engineer_features(self):
        """Perform all feature engineering steps.
        
        Returns:
            DataFrame of engineered data or None if an error occurred
        """
        logger.info("Starting feature engineering...")
        
        try:
            # Load the data
            if self.load_data() is None:
                return None
            
            # Add technical indicators
            self.add_technical_indicators()
            
            # Add price patterns
            self.add_price_patterns()
            
            # Add market regime features
            self.add_market_regime_features()
            
            # Add custom features
            self.add_custom_features()
            
            # Add anomaly detection features
            self.add_anomaly_detection_features()
            
            # Add dimensionality reduction
            self.add_dimensionality_reduction()
            
            # Remove NaN values
            self.remove_nan_values()
            
            # Save engineered features
            self.save_data()
            
            logger.info("Feature engineering completed")
            return self.data
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            return None
    
    def save_data(self):
        """Save engineered data to CSV file."""
        if self.data is not None:
            save_dataframe(self.data, self.output_file)
        else:
            logger.warning("No engineered data to save")


def engineer_features(input_file=PROCESSED_DATA_PATH, output_file=FEATURED_DATA_PATH):
    """Main function to engineer features for Bitcoin price data.
    
    Args:
        input_file: Path to the processed data file
        output_file: Path to save the featured data
        
    Returns:
        DataFrame of engineered data or None if an error occurred
    """
    # Initialize feature engineer
    engineer = FeatureEngineer(input_file, output_file)
    
    # Run feature engineering
    data = engineer.engineer_features()
    
    return data


if __name__ == "__main__":
    # This allows running the module directly for testing
    print("Engineering features for Bitcoin price data...")
    data = engineer_features()
    
    if data is not None:
        print(f"Successfully engineered features for {len(data)} records")
        print("\nSample data:")
        print(data.head())
        print("\nFeature count:", len(data.columns))
    else:
        print("Failed to engineer features")