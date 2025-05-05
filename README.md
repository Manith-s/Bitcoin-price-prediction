# Bitcoin Price Prediction

A comprehensive machine learning project for predicting Bitcoin prices using multiple advanced models and sophisticated feature engineering techniques.

## 📊 Project Overview

This project implements a complete end-to-end machine learning pipeline for Bitcoin price prediction, from data collection through model training to evaluation and visualization. It leverages multiple data sources, advanced feature engineering techniques, and a variety of machine learning models to provide accurate price predictions and market insights.

### Key Components:

1. **Data Collection** - Multi-source historical Bitcoin price data acquisition
2. **Data Preprocessing** - Cleaning, normalization, and preparation
3. **Feature Engineering** - Technical indicators, market regimes, and price patterns
4. **Model Training** - Traditional ML, deep learning, ensemble, and reinforcement learning approaches
5. **Model Evaluation** - Comprehensive comparison metrics
6. **Visualization** - Interactive charts and analysis dashboards

## 🌟 Features

- **Multi-source Data Collection**: Fetches data from Binance, CryptoCompare, CCXT, and Yahoo Finance
- **Extensive Feature Engineering**: Creates 100+ features including technical indicators, price patterns, and market regimes
- **Multiple Model Implementation**: Implements and compares traditional ML, deep learning, ensemble, and RL models
- **Robust Evaluation Framework**: Evaluates using RMSE, MAE, directional accuracy, and more
- **Interactive Visualizations**: Generates candlestick charts, technical indicators, performance comparisons
- **Market Regime Detection**: Identifies different market states for adaptive trading strategies
- **Anomaly Detection**: Identifies unusual price patterns and potential market opportunities

## 🧠 Models Implemented

### Traditional Machine Learning
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

### Deep Learning
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Bidirectional LSTM

### Ensemble Methods
- Voting Ensemble
- Stacking Ensemble
- Blending Ensemble

### Reinforcement Learning
- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)

## 🗂️ Project Structure
bitcoin-price-prediction/
├── config/                     # Configuration parameters
│   └── config.py               # Project-wide settings
├── data/                       # Data storage directory
├── logs/                       # Log files
├── models/                     # Trained model files
├── results/                    # Evaluation results and visualizations
├── scripts/                    # Execution scripts
│   ├── collect_data.py         # Data collection script
│   ├── preprocess_data.py      # Data preprocessing
│   ├── engineer_features.py    # Feature engineering
│   ├── train_models.py         # Model training
│   ├── evaluate_models.py      # Model evaluation
│   └── visualize_data.py       # Data visualization
├── src/                        # Source code
│   ├── data/                   # Data handling modules
│   │   ├── collector.py        # Data collection
│   │   ├── engineer.py         # Feature engineering
│   │   └── preprocessor.py     # Data preprocessing
│   ├── models/                 # Model implementations
│   │   ├── traditional.py      # Traditional ML models
│   │   ├── deep_learning.py    # Deep learning models
│   │   ├── ensemble.py         # Ensemble models
│   │   └── rl.py               # Reinforcement learning
│   ├── evaluation/             # Evaluation modules
│   │   ├── evaluator.py        # Model evaluation
│   │   └── visualizer.py       # Results visualization
│   └── utils/                  # Utility functions
│       └── helpers.py          # Helper functions
├── main.py                     # Main execution script
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation

## 🔍 Key Findings

Analysis of various models shows:

- **Best Overall Model**: Stacking Ensemble combining gradient boosting models
- **Highest Accuracy**: CatBoost with custom feature engineering
- **Most Important Features**: RSI, Bollinger Bands, and volume momentum indicators
- **Directional Accuracy**: Up to 75% accuracy in predicting price movement direction
- **Market Regimes**: Performance varies significantly across different market regimes

Check the `results/` directory for detailed visualizations and performance metrics.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required libraries (see requirements.txt)

### Installation

```bash
# Clone this repository
git clone https://github.com/Manith-s/Bitcoin-price-prediction.git
cd Bitcoin-price-prediction

# Install dependencies
pip install -r requirements.txt

Usage
Running the Full Pipeline
# Run the entire pipeline from data collection to visualization
python main.py --mode full

# Data collection
python scripts/collect_data.py --symbol BTC --quote USD --interval 1d

# Data preprocessing
python scripts/preprocess_data.py

# Feature engineering
python scripts/engineer_features.py

# Model training (all models)
python scripts/train_models.py

# Model training (specific model type)
python scripts/train_models.py --model-type deep-learning

# Model evaluation
python scripts/evaluate_models.py --summary-report

# Data visualization
python scripts/visualize_data.py --visualization-type all

📈 Example Results
The project generates various visualizations to help understand Bitcoin price dynamics and model performance:

Price history and volume analysis
Technical indicator effectiveness
Model performance comparisons
Feature importance analysis
Market regime classification
Interactive dashboards for exploration

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🔗 Additional Resources
Bitcoin Data Sources
Technical Analysis Indicators
Machine Learning for Trading


