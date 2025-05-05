"""Reinforcement learning models for Bitcoin price prediction"""

import os
import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import FEATURED_DATA_PATH, MODEL_DIR, RANDOM_STATE, TEST_SIZE, RL_ALGORITHM, LEARNING_RATE
from src.utils.helpers import logger, save_model, save_results

class BitcoinPriceEnv(gym.Env):
    """Custom Environment for RL-based Bitcoin Price Prediction."""
    
    def __init__(self, data, window_size=10, initial_balance=10000):
        """Initialize the environment.
        
        Args:
            data: DataFrame of Bitcoin data
            window_size: Window size for observations
            initial_balance: Initial portfolio balance
        """
        super(BitcoinPriceEnv, self).__init__()
        
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        
        # Current position in the data
        self.current_step = window_size
        
        # Define action and observation space
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observations: price history window + position state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size * self.data.shape[1] + 1,), 
            dtype=np.float32
        )
        
        # Trading state
        self.balance = self.initial_balance
        self.btc_held = 0
        self.portfolio_value = self.initial_balance
        self.position = 0  # 0 = no position, 1 = long
        
        # To track returns
        self.returns = []
        self.portfolio_values = []
        self.actions_taken = []
    
    def reset(self):
        """Reset the environment.
        
        Returns:
            Initial observation
        """
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.btc_held = 0
        self.portfolio_value = self.initial_balance
        self.position = 0
        self.returns = []
        self.portfolio_values = []
        self.actions_taken = []
        
        return self._get_observation()
    
    def step(self, action):
        """Take a step in the environment.
        
        Args:
            action: Action to take (0 = Hold, 1 = Buy, 2 = Sell)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Get current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Calculate reward based on action and next price movement
        reward = 0
        done = False
        
        # Execute action
        self.actions_taken.append(action)
        
        if action == 0:  # Hold
            pass
        elif action == 1 and self.position == 0:  # Buy
            # Buy as much BTC as possible with current balance
            self.btc_held = self.balance / current_price
            self.balance = 0
            self.position = 1
        elif action == 2 and self.position == 1:  # Sell
            # Sell all BTC
            self.balance = self.btc_held * current_price
            self.btc_held = 0
            self.position = 0
        
        # Move to the next step
        self.current_step += 1
        
        # Calculate portfolio value
        next_price = self.data.iloc[self.current_step]['close']
        self.portfolio_value = self.balance + (self.btc_held * next_price)
        self.portfolio_values.append(self.portfolio_value)
        
        # Calculate reward (percentage change in portfolio value)
        price_change = (next_price - current_price) / current_price
        if self.position == 1:  # If holding BTC
            reward = price_change
        
        # Save return for this step
        self.returns.append(reward)
        
        # Check if done
        if self.current_step >= len(self.data) - 1:
            done = True
        
        # Get next observation
        next_observation = self._get_observation()
        
        # Info
        info = {
            'portfolio_value': self.portfolio_value,
            'price_change': price_change,
            'current_price': current_price,
            'next_price': next_price,
            'reward': reward
        }
        
        return next_observation, reward, done, info
    
    def _get_observation(self):
        """Get the current observation.
        
        Returns:
            Array of observations
        """
        # Get window of data
        window = self.data.iloc[self.current_step - self.window_size:self.current_step]
        
        # Flatten the window
        window_flat = window.values.flatten()
        
        # Add position state
        observation = np.append(window_flat, self.position)
        
        return observation
    
    def get_sharpe_ratio(self):
        """Calculate Sharpe ratio of returns.
        
        Returns:
            Sharpe ratio
        """
        if len(self.returns) == 0:
            return 0
            
        # Calculate annualized Sharpe ratio
        mean_return = np.mean(self.returns)
        std_return = np.std(self.returns)
        
        if std_return == 0:
            return 0
            
        # Assuming daily returns
        sharpe_ratio = (mean_return * 252) / (std_return * np.sqrt(252))
        
        return sharpe_ratio
    
    def get_portfolio_performance(self):
        """Get portfolio performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if len(self.portfolio_values) == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        # Calculate total return
        total_return = (self.portfolio_values[-1] / self.initial_balance) - 1
        
        # Calculate Sharpe ratio
        sharpe_ratio = self.get_sharpe_ratio()
        
        # Calculate maximum drawdown
        peak = self.portfolio_values[0]
        max_drawdown = 0
        
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

class RLModels:
    """Class for training reinforcement learning models for Bitcoin price prediction."""
    
    def __init__(self, data_file=FEATURED_DATA_PATH, window_size=10):
        """Initialize the model trainer.
        
        Args:
            data_file: Path to the featured data file
            window_size: Window size for observations
        """
        self.data_file = data_file
        self.window_size = window_size
        self.data = None
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.results = {}
    
    def load_data(self):
        """Load data from CSV file.
        
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
        """Prepare data for reinforcement learning.
        
        Returns:
            Tuple of (train_data, test_data) or None if an error occurred
        """
        if self.data is None:
            logger.error("No data to prepare")
            return None
            
        logger.info("Preparing data for reinforcement learning...")
        
        # Make sure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for RL: {missing_cols}")
        
        # Normalize data for better model performance
        df = self.data.copy()
        
        # Split data for training and testing
        train_size = int(len(df) * (1 - TEST_SIZE))
        
        self.train_data = df.iloc[:train_size]
        self.test_data = df.iloc[train_size:]
        
        logger.info(f"Training data shape: {self.train_data.shape}")
        logger.info(f"Test data shape: {self.test_data.shape}")
        
        return self.train_data, self.test_data
    
    def train_rl_model(self, algorithm=RL_ALGORITHM, total_timesteps=50000):
        """Train a reinforcement learning model.
        
        Args:
            algorithm: RL algorithm to use (PPO, A2C, SAC)
            total_timesteps: Total number of timesteps to train for
            
        Returns:
            Tuple of (model, results) or None if an error occurred
        """
        if self.train_data is None:
            logger.error("Data not prepared for reinforcement learning")
            return None
            
        logger.info(f"Training {algorithm} reinforcement learning model...")
        
        try:
            # Create environment
            env = BitcoinPriceEnv(self.train_data, window_size=self.window_size)
            env = DummyVecEnv([lambda: env])
            
            # Create model based on algorithm
            if algorithm == 'PPO':
                model = PPO(
                    'MlpPolicy',
                    env,
                    learning_rate=LEARNING_RATE,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    clip_range_vf=None,
                    ent_coef=0.01,
                    verbose=1
                )
            elif algorithm == 'A2C':
                model = A2C(
                    'MlpPolicy',
                    env,
                    learning_rate=LEARNING_RATE,
                    n_steps=5,
                    gamma=0.99,
                    gae_lambda=0.95,
                    ent_coef=0.01,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    rms_prop_eps=1e-5,
                    use_rms_prop=True,
                    normalize_advantage=True,
                    verbose=1
                )
            elif algorithm == 'SAC':
                model = SAC(
                    'MlpPolicy',
                    env,
                    learning_rate=LEARNING_RATE,
                    buffer_size=10000,
                    learning_starts=100,
                    batch_size=64,
                    tau=0.005,
                    gamma=0.99,
                    train_freq=1,
                    gradient_steps=1,
                    action_noise=None,
                    ent_coef='auto',
                    target_update_interval=1,
                    target_entropy='auto',
                    verbose=1
                )
            else:
                logger.error(f"Unknown algorithm: {algorithm}")
                return None
            
            # Train the model
            model.learn(total_timesteps=total_timesteps)
            
            # Save model
            model_path = os.path.join(MODEL_DIR, f"rl_{algorithm.lower()}")
            model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Evaluate on test data
            results = self.evaluate_model(model, algorithm)
            
            # Store model and results
            self.models[algorithm] = model
            self.results[algorithm] = results
            
            logger.info(f"{algorithm} model trained and evaluated")
            
            return model, results
            
        except Exception as e:
            logger.error(f"Error training {algorithm} model: {str(e)}")
            return None
    
    def evaluate_model(self, model, algorithm):
        """Evaluate a trained model on test data.
        
        Args:
            model: Trained RL model
            algorithm: Algorithm name
            
        Returns:
            Dictionary of evaluation results
        """
        if self.test_data is None:
            logger.error("No test data for evaluation")
            return None
            
        logger.info(f"Evaluating {algorithm} model on test data...")
        
        # Create test environment
        test_env = BitcoinPriceEnv(self.test_data, window_size=self.window_size)
        test_env = DummyVecEnv([lambda: test_env])
        
        # Run test episodes
        obs = test_env.reset()
        done = False
        
        test_rewards = []
        actions = []
        portfolio_values = []
        
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done_arr, info = test_env.step(action)
            
            done = done_arr[0]
            reward = reward[0]
            info = info[0]
            
            test_rewards.append(reward)
            actions.append(action[0])
            portfolio_values.append(info['portfolio_value'])
        
        # Get performance metrics
        performance = test_env.envs[0].get_portfolio_performance()
        total_return = performance['total_return']
        sharpe_ratio = performance['sharpe_ratio']
        max_drawdown = performance['max_drawdown']
        
        # Calculate action distribution
        action_counts = np.bincount(actions, minlength=3)
        action_distribution = action_counts / len(actions)
        
        # Store results
        results = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'action_distribution': action_distribution.tolist(),
            'actions': actions,
            'portfolio_values': portfolio_values,
            'test_rewards': test_rewards
        }
        
        logger.info(f"{algorithm} model evaluated, total return: {total_return:.4f}, Sharpe ratio: {sharpe_ratio:.4f}")
        
        return results
    
    def train_all_models(self, algorithms=None, total_timesteps=50000):
        """Train all reinforcement learning models.
        
        Args:
            algorithms: List of algorithms to train, or None for default
            total_timesteps: Total number of timesteps to train for
            
        Returns:
            Dictionary of models and results
        """
        if algorithms is None:
            algorithms = ['PPO', 'A2C', 'SAC']
            
        logger.info(f"Training reinforcement learning models: {algorithms}")
        
        # Load and prepare data
        self.load_data()
        self.prepare_data()
        
        # Train each model
        for algorithm in algorithms:
            self.train_rl_model(algorithm, total_timesteps)
        
        # Save results
        save_results(self.results, 'rl_models_results')
        
        logger.info("All RL models trained successfully")
        
        return self.models, self.results
    
    def generate_visualizations(self, output_dir='results'):
        """Generate visualizations for the trained models.
        
        Args:
            output_dir: Directory to save visualizations
        """
        if not self.models or not self.results:
            logger.error("No models or results to visualize")
            return
            
        logger.info("Generating visualizations for RL models...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Import matplotlib here to avoid dependency issues
        import matplotlib.pyplot as plt
        
        # Portfolio performance plot
        plt.figure(figsize=(12, 6))
        
        # Get price data for reference
        if self.test_data is not None:
            # Calculate price returns
            price_returns = self.test_data['close'].pct_change().cumsum()
            # Normalize to start at 1
            price_values = (1 + price_returns) * 10000
            # Plot price
            plt.plot(price_values.values, label='Buy & Hold', color='black', linestyle='--')
        
        # Plot portfolio values for each model
        for model_name, results in self.results.items():
            if 'portfolio_values' in results:
                plt.plot(results['portfolio_values'], label=model_name)
        
        plt.title('RL Model Portfolio Performance')
        plt.xlabel('Time Step')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'rl_portfolio_performance.png'))
        plt.close()
        
        # Action distribution plot
        plt.figure(figsize=(10, 6))
        
        # Collect data for plotting
        action_labels = ['Hold', 'Buy', 'Sell']
        models = list(self.results.keys())
        
        # Create bar positions
        bar_width = 0.25
        r = np.arange(3)
        
        # Plot bars for each model
        for i, model_name in enumerate(models):
            if 'action_distribution' in self.results[model_name]:
                plt.bar(
                    r + i * bar_width, 
                    self.results[model_name]['action_distribution'], 
                    width=bar_width, 
                    label=model_name
                )
        
        # Finalize plot
        plt.title('Action Distribution by Model')
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.xticks(r + bar_width, action_labels)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'rl_action_distribution.png'))
        plt.close()
        
        # Performance comparison table
        performance_data = []
        for model_name, results in self.results.items():
            performance_data.append({
                'Model': model_name,
                'Total Return': results.get('total_return', 0),
                'Sharpe Ratio': results.get('sharpe_ratio', 0),
                'Max Drawdown': results.get('max_drawdown', 0)
            })
        
        # Create and save table as csv
        pd.DataFrame(performance_data).to_csv(os.path.join(output_dir, 'rl_performance_metrics.csv'), index=False)
        
        logger.info("Visualizations generated successfully")


def train_rl_models(data_file=FEATURED_DATA_PATH, window_size=10, algorithms=None, total_timesteps=50000):
    """Main function to train reinforcement learning models.
    
    Args:
        data_file: Path to the featured data file
        window_size: Window size for observations
        algorithms: List of algorithms to train, or None for default
        total_timesteps: Total number of timesteps to train for
        
    Returns:
        Dictionary of models and results
    """
    # Initialize RL trainer
    trainer = RLModels(data_file, window_size)
    
    # Train all models
    models, results = trainer.train_all_models(algorithms, total_timesteps)
    
    # Generate visualizations
    trainer.generate_visualizations()
    
    return models, results


if __name__ == "__main__":
    # This allows running the module directly for testing
    print("Training reinforcement learning models...")
    models, results = train_rl_models(algorithms=['PPO'], total_timesteps=10000)  # Use shorter training for testing
    
    if models:
        print("Models trained successfully")
        print("\nModel performance:")
        for model_name, result in results.items():
            print(f"{model_name}:")
            print(f"  Total Return: {result['total_return']:.4f}")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.4f}")
    else:
        print("Failed to train models")