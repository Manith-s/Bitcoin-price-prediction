"""Data collection module for Bitcoin price prediction"""

import os
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
import ccxt
import yfinance as yf
from binance.client import Client
from cryptocmd import CmcScraper

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import START_DATE, END_DATE, CRYPTO_SYMBOL, QUOTE_CURRENCY, TIME_INTERVAL, RAW_DATA_PATH
from src.utils.helpers import logger, save_dataframe

class DataCollector:
    """Class to collect historical and real-time cryptocurrency data."""
    
    def __init__(self, symbol=CRYPTO_SYMBOL, quote=QUOTE_CURRENCY, 
                 interval=TIME_INTERVAL, start_date=START_DATE, end_date=END_DATE,
                 output_file=RAW_DATA_PATH):
        """Initialize DataCollector with default parameters.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            quote: Quote currency (e.g., 'USD')
            interval: Time interval (e.g., '1d', '1h', '15m')
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
            output_file: Path to save the collected data
        """
        self.symbol = symbol
        self.quote = quote
        self.pair = f"{symbol}/{quote}"
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.output_file = output_file
        self.data = None
        
    def fetch_from_ccxt(self, exchange_id='binance'):
        """Fetch data using CCXT library.
        
        Args:
            exchange_id: The exchange to fetch data from (default: 'binance')
            
        Returns:
            DataFrame of historical data or None if an error occurred
        """
        try:
            logger.info(f"Fetching data from {exchange_id} using CCXT...")
            
            # Initialize exchange
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                'enableRateLimit': True,
            })
            
            # Convert timeframe to milliseconds
            timeframe_ms = {
                '1m': 60 * 1000,
                '5m': 5 * 60 * 1000,
                '15m': 15 * 60 * 1000,
                '1h': 60 * 60 * 1000,
                '4h': 4 * 60 * 60 * 1000,
                '1d': 24 * 60 * 60 * 1000,
            }
            
            # Convert dates to timestamps
            since = int(datetime.strptime(self.start_date, '%Y-%m-%d').timestamp() * 1000)
            until = int(datetime.strptime(self.end_date, '%Y-%m-%d').timestamp() * 1000)
            
            # Fetch all data in chunks
            all_candles = []
            while since < until:
                logger.info(f"Fetching data from {datetime.fromtimestamp(since/1000)}")
                candles = exchange.fetch_ohlcv(self.pair, self.interval, since)
                
                if not candles or len(candles) == 0:
                    break
                
                all_candles.extend(candles)
                since = candles[-1][0] + timeframe_ms.get(self.interval, 24 * 60 * 60 * 1000)
                
                # Rate limiting
                time.sleep(exchange.rateLimit / 1000)
            
            # Convert to dataframe
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            self.data = df
            logger.info(f"Successfully fetched {len(df)} records from {exchange_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from {exchange_id}: {str(e)}")
            return None
    
    def fetch_from_yfinance(self):
        """Fetch data using Yahoo Finance.
        
        Returns:
            DataFrame of historical data or None if an error occurred
        """
        try:
            logger.info("Fetching data from Yahoo Finance...")
            ticker = f"{self.symbol}-{self.quote}"
            data = yf.download(ticker, start=self.start_date, end=self.end_date, interval=self.interval)
            
            if data.empty:
                logger.warning("No data returned from Yahoo Finance")
                return None
            
            self.data = data
            logger.info(f"Successfully fetched {len(data)} records from Yahoo Finance")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance: {str(e)}")
            return None
    
    def fetch_from_cryptocompare(self):
        """Fetch data from CryptoCompare API.
        
        Returns:
            DataFrame of historical data or None if an error occurred
        """
        try:
            logger.info("Fetching data from CryptoCompare...")
            
            # Convert interval to seconds
            interval_seconds = {
                '1m': 60,
                '5m': 300,
                '15m': 900,
                '1h': 3600,
                '4h': 14400,
                '1d': 86400,
            }
            
            # API endpoint
            base_url = "https://min-api.cryptocompare.com/data/v2/"
            
            if self.interval == '1d':
                url = f"{base_url}histoday"
            elif self.interval.endswith('h'):
                url = f"{base_url}histohour"
            else:
                url = f"{base_url}histominute"
            
            # Calculate limit and timestamps
            start_ts = int(datetime.strptime(self.start_date, '%Y-%m-%d').timestamp())
            end_ts = int(datetime.strptime(self.end_date, '%Y-%m-%d').timestamp())
            
            # Fetch data in chunks due to API limitations
            all_data = []
            current_ts = end_ts
            
            while current_ts > start_ts:
                params = {
                    'fsym': self.symbol,
                    'tsym': self.quote,
                    'limit': 2000,  # Max limit
                    'toTs': current_ts
                }
                
                response = requests.get(url, params=params)
                data = response.json()
                
                if 'Data' not in data or 'Data' not in data['Data']:
                    break
                
                candles = data['Data']['Data']
                if not candles:
                    break
                
                all_data = candles + all_data
                current_ts = candles[0]['time'] - interval_seconds.get(self.interval, 86400)
                
                # Rate limiting
                time.sleep(0.25)
            
            # Convert to dataframe
            df = pd.DataFrame(all_data)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volumefrom', 'volumeto']]
            df.rename(columns={'volumefrom': 'volume', 'volumeto': 'quote_volume'}, inplace=True)
            
            self.data = df
            logger.info(f"Successfully fetched {len(df)} records from CryptoCompare")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from CryptoCompare: {str(e)}")
            return None
    
    def fetch_from_binance(self):
        """Fetch data from Binance API.
        
        Returns:
            DataFrame of historical data or None if an error occurred
        """
        try:
            logger.info("Fetching data from Binance...")
            
            client = Client("", "")  # Public API access doesn't require keys
            
            # Convert interval format
            interval_map = {
                '1m': Client.KLINE_INTERVAL_1MINUTE,
                '5m': Client.KLINE_INTERVAL_5MINUTE,
                '15m': Client.KLINE_INTERVAL_15MINUTE,
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '4h': Client.KLINE_INTERVAL_4HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY,
            }
            binance_interval = interval_map.get(self.interval, Client.KLINE_INTERVAL_1DAY)
            
            # Binance symbol format
            binance_symbol = f"{self.symbol}{self.quote}"
            
            # Fetch klines
            klines = client.get_historical_klines(
                binance_symbol,
                binance_interval,
                self.start_date,
                self.end_date
            )
            
            # Convert to dataframe
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignored'
            ])
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
                df[col] = df[col].astype(float)
            
            # Convert timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Keep only necessary columns
            df = df[['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']]
            
            self.data = df
            logger.info(f"Successfully fetched {len(df)} records from Binance")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from Binance: {str(e)}")
            return None
    
    def fetch_data(self, sources=None):
        """Fetch data from multiple sources and use the first successful one.
        
        Args:
            sources: List of sources to try in order (default: ['binance', 'ccxt', 'cryptocompare', 'yfinance'])
            
        Returns:
            DataFrame of historical data or None if all sources failed
        """
        if sources is None:
            sources = ['binance', 'ccxt', 'cryptocompare', 'yfinance']
        
        for source in sources:
            if source == 'binance':
                data = self.fetch_from_binance()
            elif source == 'ccxt':
                data = self.fetch_from_ccxt()
            elif source == 'cryptocompare':
                data = self.fetch_from_cryptocompare()
            elif source == 'yfinance':
                data = self.fetch_from_yfinance()
            else:
                logger.warning(f"Unknown source: {source}")
                continue
                
            if data is not None and not data.empty:
                logger.info(f"Successfully fetched data from {source}")
                self.save_data()
                return data
        
        logger.error("Failed to fetch data from any source")
        return None
    
    def save_data(self):
        """Save data to CSV file."""
        if self.data is not None and not self.data.empty:
            save_dataframe(self.data, self.output_file)
        else:
            logger.warning("No data to save")
    
    def load_data(self):
        """Load data from CSV file.
        
        Returns:
            DataFrame of historical data or None if the file doesn't exist
        """
        if os.path.exists(self.output_file):
            self.data = pd.read_csv(self.output_file, index_col=0, parse_dates=True)
            logger.info(f"Data loaded from {self.output_file}")
            return self.data
        else:
            logger.warning(f"File {self.output_file} does not exist")
            return None
    
    def get_latest_data(self, exchange_id='binance'):
        """Get real-time latest data.
        
        Args:
            exchange_id: The exchange to fetch data from (default: 'binance')
            
        Returns:
            Dictionary with latest market data or None if an error occurred
        """
        try:
            logger.info(f"Fetching latest data from {exchange_id}...")
            
            # Initialize exchange
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                'enableRateLimit': True,
            })
            
            # Fetch ticker
            ticker = exchange.fetch_ticker(self.pair)
            
            # Fetch recent trades
            trades = exchange.fetch_trades(self.pair, limit=100)
            
            # Fetch order book
            order_book = exchange.fetch_order_book(self.pair)
            
            latest_data = {
                'price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume_24h': ticker['quoteVolume'],
                'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000),
                'trades': trades,
                'bids': order_book['bids'],
                'asks': order_book['asks']
            }
            
            logger.info(f"Successfully fetched latest data from {exchange_id}")
            return latest_data
            
        except Exception as e:
            logger.error(f"Error fetching latest data from {exchange_id}: {str(e)}")
            return None


def collect_data(symbol=CRYPTO_SYMBOL, quote=QUOTE_CURRENCY, interval=TIME_INTERVAL,
                 start_date=START_DATE, end_date=END_DATE, output_file=RAW_DATA_PATH,
                 sources=None):
    """Main function to collect cryptocurrency data.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC')
        quote: Quote currency (e.g., 'USD')
        interval: Time interval (e.g., '1d', '1h', '15m')
        start_date: Start date for data collection (YYYY-MM-DD)
        end_date: End date for data collection (YYYY-MM-DD)
        output_file: Path to save the collected data
        sources: List of sources to try in order
        
    Returns:
        DataFrame of historical data or None if all sources failed
    """
    # Initialize collector
    collector = DataCollector(
        symbol=symbol,
        quote=quote,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        output_file=output_file
    )
    
    # Try to load existing data first
    data = collector.load_data()
    
    # If no data or requesting update
    if data is None:
        # Fetch new data
        data = collector.fetch_data(sources)
    
    return data


if __name__ == "__main__":
    # This allows running the module directly for testing
    print("Collecting Bitcoin price data...")
    data = collect_data()
    
    if data is not None:
        print(f"Successfully collected {len(data)} records")
        print("\nSample data:")
        print(data.head())
    else:
        print("Failed to collect data")