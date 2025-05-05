"""Deep learning models for Bitcoin price prediction"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import (
    FEATURED_DATA_PATH, MODEL_DIR, RANDOM_STATE, TEST_SIZE, VAL_SIZE,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, EARLY_STOPPING_PATIENCE
)
from src.utils.helpers import (
    logger, save_model, save_results, calculate_metrics,
    plot_price_prediction
)

class DeepLearningModels:
    """Class for training deep learning models for Bitcoin price prediction."""
    
    def __init__(self, data_file=FEATURED_DATA_PATH, target_col='close', sequence_length=10):
        """Initialize the model trainer.
        
        Args:
            data_file: Path to the featured data file
            target_col: Target column to predict
            sequence_length: Length of input sequences for sequential models
        """
        self.data_file = data_file
        self.target_col = target_col
        self.sequence_length = sequence_length
        self.data = None
        self.X_train_seq = None
        self.y_train_seq = None
        self.X_val_seq = None
        self.y_val_seq = None
        self.X_test_seq = None
        self.y_test_seq = None
        self.models = {}
        self.results = {}
        self.scaler = None
    
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
    
    def prepare_sequence_data(self):
        """Prepare data for sequence models (LSTM, GRU, etc.).
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test) or None if an error occurred
        """
        if self.data is None:
            logger.error("No data to prepare")
            return None
            
        logger.info("Preparing data for sequence learning...")
        
        # Make sure target column exists
        if self.target_col not in self.data.columns:
            logger.error(f"Target column '{self.target_col}' not found in data")
            return None
        
        try:
            # Normalize data for better model performance
            df = self.data.copy()
            
            # Keep target column for later
            target_series = df[self.target_col]
            
            # Select numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            
            # Normalize numeric data
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(numeric_df)
            scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=numeric_df.columns)
            
            # Create sequences
            X_seq = []
            y_seq = []
            
            for i in range(len(scaled_df) - self.sequence_length):
                # Get sequence
                seq = scaled_df.iloc[i:i+self.sequence_length].values
                # Get target (next value after sequence)
                target_idx = i + self.sequence_length
                if self.target_col in scaled_df.columns:
                    target = scaled_df.iloc[target_idx][self.target_col]
                else:
                    # If target column was not in numeric columns
                    target = target_series.iloc[target_idx]
                
                X_seq.append(seq)
                y_seq.append(target)
            
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            # Split data
            train_size = int(len(X_seq) * (1 - TEST_SIZE - VAL_SIZE))
            val_size = int(len(X_seq) * VAL_SIZE)
            
            X_train = X_seq[:train_size]
            y_train = y_seq[:train_size]
            
            X_val = X_seq[train_size:train_size+val_size]
            y_val = y_seq[train_size:train_size+val_size]
            
            X_test = X_seq[train_size+val_size:]
            y_test = y_seq[train_size+val_size:]
            
            logger.info(f"Training set shape: {X_train.shape}")
            logger.info(f"Validation set shape: {X_val.shape}")
            logger.info(f"Test set shape: {X_test.shape}")
            
            self.X_train_seq = X_train
            self.y_train_seq = y_train
            self.X_val_seq = X_val
            self.y_val_seq = y_val
            self.X_test_seq = X_test
            self.y_test_seq = y_test
            
            return X_train, y_train, X_val, y_val, X_test, y_test
            
        except Exception as e:
            logger.error(f"Error preparing sequence data: {str(e)}")
            return None
    
    def train_lstm(self, units=100, dropout=0.2):
        """Train an LSTM model.
        
        Args:
            units: Number of LSTM units
            dropout: Dropout rate
            
        Returns:
            Tuple of (model, results) or None if an error occurred
        """
        if self.X_train_seq is None or self.y_train_seq is None:
            logger.error("Data not prepared for sequence learning")
            return None
            
        logger.info("Training LSTM model...")
        
        try:
            # Get dimensions
            n_features = self.X_train_seq.shape[2]
            
            # Create model
            model = Sequential([
                LSTM(units, return_sequences=True, input_shape=(self.sequence_length, n_features)),
                Dropout(dropout),
                LSTM(units//2),
                Dropout(dropout),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=LEARNING_RATE),
                loss='mse'
            )
            
            # Define callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            )
            
            model_path = os.path.join(MODEL_DIR, 'lstm_model.h5')
            checkpoint = ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True
            )
            
            # Train model
            history = model.fit(
                self.X_train_seq, self.y_train_seq,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(self.X_val_seq, self.y_val_seq),
                callbacks=[early_stopping, checkpoint],
                verbose=1
            )
            
            # Make predictions
            train_pred = model.predict(self.X_train_seq).flatten()
            val_pred = model.predict(self.X_val_seq).flatten()
            test_pred = model.predict(self.X_test_seq).flatten()
            
            # Calculate metrics
            train_metrics = calculate_metrics(self.y_train_seq, train_pred)
            val_metrics = calculate_metrics(self.y_val_seq, val_pred)
            test_metrics = calculate_metrics(self.y_test_seq, test_pred)
            
            # Store model and results
            self.models['LSTM'] = model
            self.results['LSTM'] = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'train_pred': train_pred.tolist(),
                'val_pred': val_pred.tolist(),
                'test_pred': test_pred.tolist(),
                'test_actual': self.y_test_seq.tolist(),
                'history': {
                    'loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history['val_loss']]
                }
            }
            
            logger.info(f"LSTM model trained, test RMSE: {test_metrics['RMSE']:.4f}")
            logger.info(f"Directional Accuracy: {test_metrics['Directional Accuracy']:.4f}")
            
            return model, self.results['LSTM']
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {str(e)}")
            return None
    
    def train_gru(self, units=100, dropout=0.2):
        """Train a GRU model.
        
        Args:
            units: Number of GRU units
            dropout: Dropout rate
            
        Returns:
            Tuple of (model, results) or None if an error occurred
        """
        if self.X_train_seq is None or self.y_train_seq is None:
            logger.error("Data not prepared for sequence learning")
            return None
            
        logger.info("Training GRU model...")
        
        try:
            # Get dimensions
            n_features = self.X_train_seq.shape[2]
            
            # Create model
            model = Sequential([
                GRU(units, return_sequences=True, input_shape=(self.sequence_length, n_features)),
                Dropout(dropout),
                GRU(units//2),
                Dropout(dropout),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=LEARNING_RATE),
                loss='mse'
            )
            
            # Define callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            )
            
            model_path = os.path.join(MODEL_DIR, 'gru_model.h5')
            checkpoint = ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True
            )
            
            # Train model
            history = model.fit(
                self.X_train_seq, self.y_train_seq,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(self.X_val_seq, self.y_val_seq),
                callbacks=[early_stopping, checkpoint],
                verbose=1
            )
            
            # Make predictions
            train_pred = model.predict(self.X_train_seq).flatten()
            val_pred = model.predict(self.X_val_seq).flatten()
            test_pred = model.predict(self.X_test_seq).flatten()
            
            # Calculate metrics
            train_metrics = calculate_metrics(self.y_train_seq, train_pred)
            val_metrics = calculate_metrics(self.y_val_seq, val_pred)
            test_metrics = calculate_metrics(self.y_test_seq, test_pred)
            
            # Store model and results
            self.models['GRU'] = model
            self.results['GRU'] = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'train_pred': train_pred.tolist(),
                'val_pred': val_pred.tolist(),
                'test_pred': test_pred.tolist(),
                'test_actual': self.y_test_seq.tolist(),
                'history': {
                    'loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history['val_loss']]
                }
            }
            
            logger.info(f"GRU model trained, test RMSE: {test_metrics['RMSE']:.4f}")
            logger.info(f"Directional Accuracy: {test_metrics['Directional Accuracy']:.4f}")
            
            return model, self.results['GRU']
            
        except Exception as e:
            logger.error(f"Error training GRU model: {str(e)}")
            return None
    
    def train_bidirectional_lstm(self, units=100, dropout=0.2):
        """Train a Bidirectional LSTM model.
        
        Args:
            units: Number of LSTM units
            dropout: Dropout rate
            
        Returns:
            Tuple of (model, results) or None if an error occurred
        """
        if self.X_train_seq is None or self.y_train_seq is None:
            logger.error("Data not prepared for sequence learning")
            return None
            
        logger.info("Training Bidirectional LSTM model...")
        
        try:
            # Get dimensions
            n_features = self.X_train_seq.shape[2]
            
            # Create model
            model = Sequential([
                Bidirectional(LSTM(units, return_sequences=True), input_shape=(self.sequence_length, n_features)),
                Dropout(dropout),
                Bidirectional(LSTM(units//2)),
                Dropout(dropout),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=LEARNING_RATE),
                loss='mse'
            )
            
            # Define callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            )
            
            model_path = os.path.join(MODEL_DIR, 'bidirectional_lstm_model.h5')
            checkpoint = ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True
            )
            
            # Train model
            history = model.fit(
                self.X_train_seq, self.y_train_seq,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(self.X_val_seq, self.y_val_seq),
                callbacks=[early_stopping, checkpoint],
                verbose=1
            )
            
            # Make predictions
            train_pred = model.predict(self.X_train_seq).flatten()
            val_pred = model.predict(self.X_val_seq).flatten()
            test_pred = model.predict(self.X_test_seq).flatten()
            
            # Calculate metrics
            train_metrics = calculate_metrics(self.y_train_seq, train_pred)
            val_metrics = calculate_metrics(self.y_val_seq, val_pred)
            test_metrics = calculate_metrics(self.y_test_seq, test_pred)
            
            # Store model and results
            self.models['BidirectionalLSTM'] = model
            self.results['BidirectionalLSTM'] = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'train_pred': train_pred.tolist(),
                'val_pred': val_pred.tolist(),
                'test_pred': test_pred.tolist(),
                'test_actual': self.y_test_seq.tolist(),
                'history': {
                    'loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history['val_loss']]
                }
            }
            
            logger.info(f"Bidirectional LSTM model trained, test RMSE: {test_metrics['RMSE']:.4f}")
            logger.info(f"Directional Accuracy: {test_metrics['Directional Accuracy']:.4f}")
            
            return model, self.results['BidirectionalLSTM']
            
        except Exception as e:
            logger.error(f"Error training Bidirectional LSTM model: {str(e)}")
            return None
    
    def train_all_models(self):
        """Train all deep learning models.
        
        Returns:
            Dictionary of models and results
        """
        logger.info("Training all deep learning models...")
        
        # Load and prepare data
        self.load_data()
        self.prepare_sequence_data()
        
        # Train each model
        self.train_lstm()
        self.train_gru()
        self.train_bidirectional_lstm()
        
        # Save results
        save_results(self.results, 'deep_learning_models_results')
        
        logger.info("All deep learning models trained successfully")
        
        return self.models, self.results
    
    def generate_visualizations(self, output_dir='results'):
        """Generate visualizations for the trained models.
        
        Args:
            output_dir: Directory to save visualizations
        """
        if not self.models or not self.results:
            logger.error("No models or results to visualize")
            return
            
        logger.info("Generating visualizations for deep learning models...")
        
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
            
            # Training history
            if 'history' in results:
                # Plot training history
                plt.figure(figsize=(10, 6))
                plt.plot(results['history']['loss'], label='Training Loss')
                plt.plot(results['history']['val_loss'], label='Validation Loss')
                plt.title(f'{model_name} Training History')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(os.path.join(output_dir, f'training_history_{model_name}.png'))
                plt.close()
        
        logger.info("Visualizations generated successfully")


def train_deep_learning_models(data_file=FEATURED_DATA_PATH, target_col='close', sequence_length=10):
    """Main function to train deep learning models.
    
    Args:
        data_file: Path to the featured data file
        target_col: Target column to predict
        sequence_length: Length of input sequences for sequential models
        
    Returns:
        Dictionary of models and results
    """
    # Initialize model trainer
    trainer = DeepLearningModels(data_file, target_col, sequence_length)
    
    # Train all models
    models, results = trainer.train_all_models()
    
    # Generate visualizations
    trainer.generate_visualizations()
    
    return models, results


if __name__ == "__main__":
    # This allows running the module directly for testing
    import matplotlib.pyplot as plt
    
    print("Training deep learning models...")
    models, results = train_deep_learning_models()
    
    if models:
        print("Models trained successfully")
        print("\nModel performance (Test RMSE):")
        for model_name, result in results.items():
            print(f"{model_name}: {result['test_metrics']['RMSE']:.4f}")
    else:
        print("Failed to train models")