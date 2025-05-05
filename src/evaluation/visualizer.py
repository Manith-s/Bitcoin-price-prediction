"""Data visualization module for Bitcoin price prediction"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import FEATURED_DATA_PATH, RESULTS_DIR, CRYPTO_SYMBOL, QUOTE_CURRENCY
from src.utils.helpers import logger, load_dataframe

class DataVisualizer:
    """Class to create visualizations of Bitcoin data and predictions."""
    
    def __init__(self, data_file=FEATURED_DATA_PATH, results_dir=RESULTS_DIR):
        """Initialize DataVisualizer with data file and results directory.
        
        Args:
            data_file: Path to the featured data file
            results_dir: Directory to save visualization results
        """
        self.data_file = data_file
        self.results_dir = results_dir
        self.data = None
    
    def load_data(self):
        """Load data from CSV file.
        
        Returns:
            DataFrame of loaded data or None if the file doesn't exist
        """
        self.data = load_dataframe(self.data_file)
        return self.data
    
    def plot_price_history(self, save_path='price_history.png', show_plot=False):
        """Plot Bitcoin price history.
        
        Args:
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if self.data is None:
            logger.error("No data to visualize")
            return None
            
        logger.info("Creating price history plot...")
        
        # Check if 'close' column exists
        if 'close' not in self.data.columns:
            logger.warning("No 'close' column found in data")
            return None
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.data.index, self.data['close'])
        
        plt.title(f'{CRYPTO_SYMBOL}/{QUOTE_CURRENCY} Price History')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        
        # Save figure
        save_path = os.path.join(self.results_dir, save_path)
        plt.savefig(save_path)
        logger.info(f"Price history plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return plt.gcf()
    
    def plot_price_with_volume(self, save_path='price_volume.png', show_plot=False):
        """Plot Bitcoin price with volume.
        
        Args:
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if self.data is None:
            logger.error("No data to visualize")
            return None
            
        logger.info("Creating price with volume plot...")
        
        # Check if required columns exist
        required_cols = ['close', 'volume']
        if not all(col in self.data.columns for col in required_cols):
            logger.warning(f"Missing required columns: {[col for col in required_cols if col not in self.data.columns]}")
            return None
        
        # Create plot
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot price on primary y-axis
        ax1.plot(self.data.index, self.data['close'], color='blue')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Create secondary y-axis for volume
        ax2 = ax1.twinx()
        ax2.bar(self.data.index, self.data['volume'], alpha=0.3, color='gray')
        ax2.set_ylabel('Volume', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        
        plt.title(f'{CRYPTO_SYMBOL}/{QUOTE_CURRENCY} Price and Volume')
        plt.grid(True)
        
        # Save figure
        save_path = os.path.join(self.results_dir, save_path)
        plt.savefig(save_path)
        logger.info(f"Price with volume plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_candlestick(self, days=90, save_path='candlestick.html', show_plot=False):
        """Plot Bitcoin candlestick chart using Plotly.
        
        Args:
            days: Number of days to include in the chart, or None for all data
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Plotly figure
        """
        if self.data is None:
            logger.error("No data to visualize")
            return None
            
        logger.info("Creating candlestick plot...")
        
        # Check if required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in self.data.columns for col in required_cols):
            logger.warning(f"Missing required columns: {[col for col in required_cols if col not in self.data.columns]}")
            return None
        
        # Limit to specified number of days
        if days:
            data = self.data.iloc[-days:]
        else:
            data = self.data
        
        # Create plot using plotly
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            increasing_line_color='green',
            decreasing_line_color='red'
        )])
        
        fig.update_layout(
            title=f'{CRYPTO_SYMBOL}/{QUOTE_CURRENCY} Candlestick Chart',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False
        )
        
        # Save figure
        save_path = os.path.join(self.results_dir, save_path)
        fig.write_html(save_path)
        logger.info(f"Candlestick plot saved to {save_path}")
        
        if show_plot:
            fig.show()
        
        return fig
    
    def plot_technical_indicators(self, indicators=None, save_path='technical_indicators.html', show_plot=False):
        """Plot Bitcoin price with technical indicators using Plotly.
        
        Args:
            indicators: Dictionary of indicators to include in the plot
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Plotly figure
        """
        if self.data is None:
            logger.error("No data to visualize")
            return None
            
        logger.info("Creating technical indicators plot...")
        
        # Default indicators
        if indicators is None:
            indicators = {
                'RSI_14': {'name': 'RSI (14)', 'panel': 'lower'},
                'MACD': {'name': 'MACD', 'panel': 'lower'},
                'Bollinger_upper_20': {'name': 'Bollinger Upper', 'panel': 'main'},
                'Bollinger_middle_20': {'name': 'Bollinger Middle', 'panel': 'main'},
                'Bollinger_lower_20': {'name': 'Bollinger Lower', 'panel': 'main'}
            }
        
        # Check if required columns exist
        required_cols = ['close'] + list(indicators.keys())
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            # Remove missing indicators
            for col in missing_cols:
                if col in indicators:
                    del indicators[col]
        
        if not indicators:
            logger.warning("No valid indicators to plot")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3]
        )
        
        # Add price to main panel
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['close'],
                name='Price',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Add indicators
        for col, props in indicators.items():
            if col in self.data.columns:
                if props['panel'] == 'main':
                    # Add to main panel
                    fig.add_trace(
                        go.Scatter(
                            x=self.data.index,
                            y=self.data[col],
                            name=props['name'],
                            line=dict(dash='dash')
                        ),
                        row=1, col=1
                    )
                else:
                    # Add to lower panel
                    fig.add_trace(
                        go.Scatter(
                            x=self.data.index,
                            y=self.data[col],
                            name=props['name']
                        ),
                        row=2, col=1
                    )
        
        # Update layout
        fig.update_layout(
            title=f'{CRYPTO_SYMBOL}/{QUOTE_CURRENCY} with Technical Indicators',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            height=800
        )
        
        # Save figure
        save_path = os.path.join(self.results_dir, save_path)
        fig.write_html(save_path)
        logger.info(f"Technical indicators plot saved to {save_path}")
        
        if show_plot:
            fig.show()
        
        return fig
    
    def plot_correlation_matrix(self, columns=None, save_path='correlation_matrix.png', show_plot=False):
        """Plot correlation matrix of features.
        
        Args:
            columns: List of columns to include in the matrix
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if self.data is None:
            logger.error("No data to visualize")
            return None
            
        logger.info("Creating correlation matrix plot...")
        
        # Select columns for correlation
        if columns is None:
            # Use key columns
            key_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'RSI_14', 'MACD', 'Bollinger_width_20', 'volatility',
                'momentum_10', 'market_regime'
            ]
            columns = [col for col in key_columns if col in self.data.columns]
        
        if not columns:
            logger.warning("No valid columns for correlation matrix")
            return None
        
        # Calculate correlation matrix
        corr = self.data[columns].corr()
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmax=1,
            vmin=-1,
            center=0,
            square=True,
            linewidths=.5,
            annot=True,
            fmt='.2f'
        )
        
        plt.title('Feature Correlation Matrix')
        
        # Save figure
        save_path = os.path.join(self.results_dir, save_path)
        plt.savefig(save_path)
        logger.info(f"Correlation matrix plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return plt.gcf()
    
    def plot_feature_distributions(self, columns=None, save_path='feature_distributions.png', show_plot=False):
        """Plot distribution of key features.
        
        Args:
            columns: List of columns to include in the distributions
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if self.data is None:
            logger.error("No data to visualize")
            return None
            
        logger.info("Creating feature distributions plot...")
        
        # Select columns for distributions
        if columns is None:
            # Use key columns
            key_columns = [
                'close', 'volume', 'RSI_14', 'volatility',
                'momentum_10', 'market_regime'
            ]
            columns = [col for col in key_columns if col in self.data.columns]
        
        if not columns:
            logger.warning("No valid columns for distributions")
            return None
        
        # Determine grid size
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        
        # Flatten axes array for easier indexing
        if n_rows > 1 or n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Plot distributions
        for i, col in enumerate(columns):
            if i < len(axes):
                ax = axes[i]
                sns.histplot(self.data[col].dropna(), kde=True, ax=ax)
                ax.set_title(f'Distribution of {col}')
                ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(columns), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.results_dir, save_path)
        plt.savefig(save_path)
        logger.info(f"Feature distributions plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_returns_analysis(self, save_path='returns_analysis.png', show_plot=False):
        """Plot analysis of returns.
        
        Args:
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if self.data is None:
            logger.error("No data to visualize")
            return None
            
        logger.info("Creating returns analysis plot...")
        
        # Check if close column exists
        if 'close' not in self.data.columns:
            logger.warning("No 'close' column found in data")
            return None
        
        # Calculate returns
        returns = self.data['close'].pct_change()
        log_returns = np.log(self.data['close'] / self.data['close'].shift(1))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot returns over time
        axes[0, 0].plot(returns.index, returns, alpha=0.7)
        axes[0, 0].set_title('Daily Returns Over Time')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot log returns over time
        axes[0, 1].plot(log_returns.index, log_returns, alpha=0.7, color='green')
        axes[0, 1].set_title('Daily Log Returns Over Time')
        axes[0, 1].set_ylabel('Log Return')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot return distribution
        sns.histplot(returns.dropna(), kde=True, ax=axes[1, 0])
        axes[1, 0].set_title('Distribution of Daily Returns')
        axes[1, 0].set_xlabel('Return')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot log return distribution
        sns.histplot(log_returns.dropna(), kde=True, ax=axes[1, 1], color='green')
        axes[1, 1].set_title('Distribution of Daily Log Returns')
        axes[1, 1].set_xlabel('Log Return')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.results_dir, save_path)
        plt.savefig(save_path)
        logger.info(f"Returns analysis plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_volatility_analysis(self, window=30, save_path='volatility_analysis.png', show_plot=False):
        """Plot volatility analysis.
        
        Args:
            window: Window size for rolling volatility
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if self.data is None:
            logger.error("No data to visualize")
            return None
            
        logger.info("Creating volatility analysis plot...")
        
        # Check if close column exists
        if 'close' not in self.data.columns:
            logger.warning("No 'close' column found in data")
            return None
        
        # Calculate returns and rolling volatility
        returns = self.data['close'].pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        
        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot price
        axes[0].plot(self.data.index, self.data['close'])
        axes[0].set_title(f'{CRYPTO_SYMBOL}/{QUOTE_CURRENCY} Price')
        axes[0].set_ylabel('Price')
        axes[0].grid(True, alpha=0.3)
        
        # Plot volatility
        axes[1].plot(volatility.index, volatility, color='red')
        axes[1].set_title(f'{window}-Day Rolling Volatility (Annualized)')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Volatility')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.results_dir, save_path)
        plt.savefig(save_path)
        logger.info(f"Volatility analysis plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_seasonality_analysis(self, save_path='seasonality_analysis.png', show_plot=False):
        """Plot seasonality analysis.
        
        Args:
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if self.data is None:
            logger.error("No data to visualize")
            return None
            
        logger.info("Creating seasonality analysis plot...")
        
        # Check if close column exists
        if 'close' not in self.data.columns:
            logger.warning("No 'close' column found in data")
            return None
        
        # Calculate returns
        returns = self.data['close'].pct_change()
        
        # Extract date components
        data = returns.dropna().to_frame()
        data['year'] = data.index.year
        data['month'] = data.index.month
        data['day_of_week'] = data.index.dayofweek
        data['day_of_month'] = data.index.day
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Monthly seasonality
        monthly_returns = data.groupby('month')['close'].mean()
        axes[0, 0].bar(monthly_returns.index, monthly_returns, color='blue', alpha=0.7)
        axes[0, 0].set_title('Average Returns by Month')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Average Return')
        axes[0, 0].set_xticks(range(1, 13))
        axes[0, 0].grid(True, alpha=0.3)
        
        # Day of week seasonality
        day_of_week_returns = data.groupby('day_of_week')['close'].mean()
        axes[0, 1].bar(day_of_week_returns.index, day_of_week_returns, color='green', alpha=0.7)
        axes[0, 1].set_title('Average Returns by Day of Week')
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Average Return')
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        axes[0, 1].grid(True, alpha=0.3)
        
        # Day of month seasonality
        day_of_month_returns = data.groupby('day_of_month')['close'].mean()
        axes[1, 0].bar(day_of_month_returns.index, day_of_month_returns, color='purple', alpha=0.7)
        axes[1, 0].set_title('Average Returns by Day of Month')
        axes[1, 0].set_xlabel('Day of Month')
        axes[1, 0].set_ylabel('Average Return')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Yearly seasonality
        yearly_returns = data.groupby('year')['close'].mean()
        axes[1, 1].bar(yearly_returns.index, yearly_returns, color='orange', alpha=0.7)
        axes[1, 1].set_title('Average Returns by Year')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Average Return')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.results_dir, save_path)
        plt.savefig(save_path)
        logger.info(f"Seasonality analysis plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_market_regimes(self, save_path='market_regimes.png', show_plot=False):
        """Plot market regimes analysis.
        
        Args:
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if self.data is None:
            logger.error("No data to visualize")
            return None
            
        logger.info("Creating market regimes plot...")
        
        # Check if required columns exist
        required_cols = ['close', 'market_regime']
        if not all(col in self.data.columns for col in required_cols):
            logger.warning(f"Missing required columns: {[col for col in required_cols if col not in self.data.columns]}")
            return None
        
        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot price
        axes[0].plot(self.data.index, self.data['close'])
        axes[0].set_title(f'{CRYPTO_SYMBOL}/{QUOTE_CURRENCY} Price')
        axes[0].set_ylabel('Price')
        axes[0].grid(True, alpha=0.3)
        
        # Plot market regimes
        import matplotlib.cm as cm
        regimes = self.data['market_regime'].values
        max_regime = max(regimes)
        colors = cm.viridis(regimes / max(max_regime, 1))
        
        axes[1].scatter(
            self.data.index,
            self.data['market_regime'],
            c=colors,
            alpha=0.7
        )
        axes[1].set_title('Market Regimes')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Regime')
        axes[1].grid(True, alpha=0.3)
        
        # Add legend for market regimes
        from matplotlib.lines import Line2D
        
        # Get unique regimes
        unique_regimes = sorted(self.data['market_regime'].unique())
        
        # Create legend elements
        legend_elements = []
        regime_labels = {
            1: 'High Vol, Uptrend',
            2: 'High Vol, Downtrend',
            3: 'Low Vol, Uptrend',
            4: 'Low Vol, Downtrend'
        }
        
        for regime in unique_regimes:
            if regime > 0:  # Skip regime 0 (undefined)
                label = regime_labels.get(regime, f'Regime {regime}')
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor=cm.viridis(regime / max(max_regime, 1)), 
                           markersize=10, label=label)
                )
        
        axes[1].legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.results_dir, save_path)
        plt.savefig(save_path)
        logger.info(f"Market regimes plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def create_interactive_dashboard(self, save_path='bitcoin_dashboard.html', show_dashboard=False):
        """Create an interactive dashboard with Plotly.
        
        Args:
            save_path: Path to save the dashboard
            show_dashboard: Whether to display the dashboard
            
        Returns:
            Plotly figure
        """
        if self.data is None:
            logger.error("No data to visualize")
            return None
            
        logger.info("Creating interactive dashboard...")
        
        # Check if required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return None
        
        # Create dashboard using plotly
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(
                f'{CRYPTO_SYMBOL}/{QUOTE_CURRENCY} Price', 
                'Volume', 
                'Returns', 
                'Technical Indicators'
            )
        )
        
        # Add price candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['open'],
                high=self.data['high'],
                low=self.data['low'],
                close=self.data['close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Add volume
        fig.add_trace(
            go.Bar(
                x=self.data.index,
                y=self.data['volume'],
                name='Volume',
                marker_color='rgba(0, 0, 255, 0.3)'
            ),
            row=2, col=1
        )
        
        # Add returns
        returns = self.data['close'].pct_change() * 100  # Percentage
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=returns,
                name='Daily Returns (%)',
                line=dict(color='purple')
            ),
            row=3, col=1
        )
        
        # Add technical indicators
        indicators = {
            'RSI_14': {'color': 'orange'},
            'MACD': {'color': 'blue'},
            'Bollinger_width_20': {'color': 'green'}
        }
        
        for indicator, props in indicators.items():
            if indicator in self.data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index,
                        y=self.data[indicator],
                        name=indicator,
                        line=dict(color=props['color'])
                    ),
                    row=4, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=f'{CRYPTO_SYMBOL}/{QUOTE_CURRENCY} Interactive Dashboard',
            xaxis_rangeslider_visible=False,
            height=1000,
            width=1200,
            showlegend=True,
            legend=dict(orientation='h', y=1.02)
        )
        
        # Update axes
        fig.update_yaxes(title_text='Price', row=1, col=1)
        fig.update_yaxes(title_text='Volume', row=2, col=1)
        fig.update_yaxes(title_text='Returns (%)', row=3, col=1)
        fig.update_yaxes(title_text='Value', row=4, col=1)
        fig.update_xaxes(title_text='Date', row=4, col=1)
        
        # Save dashboard
        save_path = os.path.join(self.results_dir, save_path)
        fig.write_html(save_path)
        logger.info(f"Interactive dashboard saved to {save_path}")
        
        if show_dashboard:
            fig.show()
        
        return fig
    
    def create_all_visualizations(self, show_plots=False):
        """Create all visualizations.
        
        Args:
            show_plots: Whether to display the plots
            
        Returns:
            Dictionary of created figures
        """
        logger.info("Creating all visualizations...")
        
        # Load data
        self.load_data()
        
        # Create visualizations
        figures = {}
        
        # Basic visualizations
        figures['price_history'] = self.plot_price_history(show_plot=show_plots)
        figures['price_volume'] = self.plot_price_with_volume(show_plot=show_plots)
        figures['candlestick'] = self.plot_candlestick(show_plot=show_plots)
        figures['technical_indicators'] = self.plot_technical_indicators(show_plot=show_plots)
        
        # Analysis visualizations
        figures['correlation_matrix'] = self.plot_correlation_matrix(show_plot=show_plots)
        figures['feature_distributions'] = self.plot_feature_distributions(show_plot=show_plots)
        figures['returns_analysis'] = self.plot_returns_analysis(show_plot=show_plots)
        figures['volatility_analysis'] = self.plot_volatility_analysis(show_plot=show_plots)
        figures['seasonality_analysis'] = self.plot_seasonality_analysis(show_plot=show_plots)
        
        # Advanced visualizations
        figures['market_regimes'] = self.plot_market_regimes(show_plot=show_plots)
        figures['dashboard'] = self.create_interactive_dashboard(show_dashboard=show_plots)
        
        logger.info("All visualizations created")
        
        return figures


def create_visualizations(data_file=FEATURED_DATA_PATH, show_plots=False):
    """Main function to create visualizations for Bitcoin price data.
    
    Args:
        data_file: Path to the featured data file
        show_plots: Whether to display the plots
        
    Returns:
        Dictionary of created figures
    """
    # Initialize visualizer
    visualizer = DataVisualizer(data_file)
    
    # Create all visualizations
    figures = visualizer.create_all_visualizations(show_plots)
    
    return figures


if __name__ == "__main__":
    # This allows running the module directly for testing
    print("Creating visualizations for Bitcoin price data...")
    figures = create_visualizations(show_plots=True)
    
    if figures:
        print("Visualizations created successfully")
        print("\nCreated visualizations:")
        for name in figures.keys():
            print(f"- {name}")
    else:
        print("Failed to create visualizations")