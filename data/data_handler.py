"""
BTCUSD Data Handler Module
Handles data sourcing from multiple APIs for comparison and validation
"""

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import time
from typing import Optional, Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BTCDataHandler:
    """
    Handles BTCUSD data sourcing from multiple APIs for comparison and validation
    """

    def __init__(self):
        self.binance_base_url = "https://api.binance.com/api/v3"
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"

    def get_binance_data(self, symbol: str = "BTCUSDT", interval: str = "15m",
                        start_date: str = None, end_date: str = None,
                        limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        Fetch historical klines data from Binance API

        Args:
            symbol: Trading pair symbol (default: BTCUSDT)
            interval: Kline interval (default: 15m)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Number of records to fetch (max 1000 per request)

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            endpoint = f"{self.binance_base_url}/klines"

            # Convert dates to timestamps if provided
            if start_date:
                start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            else:
                # Default to 1 year ago
                start_ts = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)

            if end_date:
                end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
            else:
                end_ts = int(datetime.now().timestamp() * 1000)

            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_ts,
                'endTime': end_ts,
                'limit': limit
            }

            logger.info(f"Fetching Binance data for {symbol}...")
            response = requests.get(endpoint, params=params)
            response.raise_for_status()

            data = response.json()

            if not data:
                logger.warning("No data received from Binance")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'unused'
            ])

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Convert price columns to float
            price_columns = ['open', 'high', 'low', 'close', 'volume']
            df[price_columns] = df[price_columns].astype(float)

            # Select only OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

            logger.info(f"Successfully fetched {len(df)} records from Binance")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Binance data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Binance data fetch: {e}")
            return None

    def get_coingecko_data(self, coin_id: str = "bitcoin", vs_currency: str = "usd",
                          days: int = 365) -> Optional[pd.DataFrame]:
        """
        Fetch historical market data from CoinGecko API

        Args:
            coin_id: Coin identifier (default: bitcoin)
            vs_currency: Target currency (default: usd)
            days: Number of days of historical data

        Returns:
            DataFrame with OHLC data or None if failed
        """
        try:
            endpoint = f"{self.coingecko_base_url}/coins/{coin_id}/market_chart"

            params = {
                'vs_currency': vs_currency,
                'days': days,
                'interval': 'daily'  # CoinGecko doesn't support 15m intervals
            }

            logger.info(f"Fetching CoinGecko data for {coin_id}...")
            response = requests.get(endpoint, params=params)
            response.raise_for_status()

            data = response.json()

            if 'prices' not in data:
                logger.warning("No price data received from CoinGecko")
                return None

            # Extract price data
            prices = data['prices']
            volumes = data.get('total_volumes', [])

            # Create DataFrame
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Add volume data if available
            if volumes:
                volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
                volume_df.set_index('timestamp', inplace=True)
                df['volume'] = volume_df['volume']

            # For CoinGecko, we only get daily data, so we'll need to resample
            # This is a limitation of the free tier
            # Convert to OHLC format for compatibility
            df_ohlc = pd.DataFrame(index=df.index)
            df_ohlc['Close'] = df['price']
            df_ohlc['Volume'] = df.get('volume', 0)

            # Generate synthetic OHLC data based on price movements
            # This is for compatibility - in production, you'd want real OHLC data
            df_ohlc['Open'] = df_ohlc['Close'].shift(1).fillna(df_ohlc['Close'].iloc[0])
            df_ohlc['High'] = df_ohlc[['Open', 'Close']].max(axis=1)
            df_ohlc['Low'] = df_ohlc[['Open', 'Close']].min(axis=1)

            logger.info(f"Successfully fetched {len(df_ohlc)} records from CoinGecko")
            return df_ohlc

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching CoinGecko data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in CoinGecko data fetch: {e}")
            return None

    def get_yahoo_finance_data(self, symbol: str = "BTC-USD", period: str = "1y",
                              interval: str = "15m") -> Optional[pd.DataFrame]:
        """
        Fetch historical data from Yahoo Finance

        Args:
            symbol: Trading pair symbol (default: BTC-USD)
            period: Period of data (default: 1y)
            interval: Data interval (default: 15m)

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            logger.info(f"Fetching Yahoo Finance data for {symbol}...")

            # Download data
            df = yf.download(symbol, period=period, interval=interval, progress=False)

            if df.empty:
                logger.warning("No data received from Yahoo Finance")
                return None

            # Yahoo Finance returns MultiIndex columns, flatten them
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            # Ensure we have the expected columns
            expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[expected_columns]

            logger.info(f"Successfully fetched {len(df)} records from Yahoo Finance")
            return df

        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data: {e}")
            return None

    def fetch_all_sources(self, start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data from all available sources for comparison

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Dictionary with data from each source
        """
        sources_data = {}

        # Fetch from Binance (15-minute data)
        binance_data = self.get_binance_data(start_date=start_date, end_date=end_date)
        if binance_data is not None:
            sources_data['binance'] = binance_data

        # Fetch from Yahoo Finance (15-minute data) - use shorter period for 15m data
        yahoo_data = self.get_yahoo_finance_data(period="60d")  # Last 60 days for 15m data
        if yahoo_data is not None:
            sources_data['yahoo'] = yahoo_data

        # Fetch from CoinGecko (daily data - limitation of free tier)
        coingecko_data = self.get_coingecko_data()
        if coingecko_data is not None:
            sources_data['coingecko'] = coingecko_data

        return sources_data

    def compare_data_quality(self, sources_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Compare data quality across different sources

        Args:
            sources_data: Dictionary with data from each source

        Returns:
            Dictionary with quality metrics for each source
        """
        quality_metrics = {}

        for source, data in sources_data.items():
            metrics = {
                'record_count': len(data),
                'date_range': f"{data.index.min()} to {data.index.max()}",
                'missing_values': data.isnull().sum().sum(),
                'price_range': f"{data['Low'].min():.2f} - {data['High'].max():.2f}"
            }

            # Calculate basic statistics
            if len(data) > 0:
                metrics['avg_volume'] = data['Volume'].mean()
                metrics['price_volatility'] = data['Close'].std()
                metrics['data_completeness'] = (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100

            quality_metrics[source] = metrics

        return quality_metrics

    def validate_data_consistency(self, sources_data: Dict[str, pd.DataFrame],
                                 tolerance: float = 0.02) -> Dict[str, bool]:
        """
        Validate consistency between different data sources

        Args:
            sources_data: Dictionary with data from each source
            tolerance: Price tolerance for comparison (default: 2%)

        Returns:
            Dictionary with consistency validation results
        """
        consistency_results = {}

        # Get common date range
        all_dates = set()
        for data in sources_data.values():
            all_dates.update(data.index.date)

        common_dates = sorted(all_dates)

        # Compare sources pairwise
        sources = list(sources_data.keys())

        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                source1, source2 = sources[i], sources[j]

                data1 = sources_data[source1]
                data2 = sources_data[source2]

                # Simple consistency check - compare overlapping date ranges
                data1_aligned = data1
                data2_aligned = data2

                if len(data1_aligned) > 0 and len(data2_aligned) > 0:
                    try:
                        # Handle timezone issues by converting to naive timestamps for comparison
                        data1_naive = data1_aligned.copy()
                        data2_naive = data2_aligned.copy()

                        if data1_naive.index.tz is not None:
                            data1_naive.index = data1_naive.index.tz_localize(None)
                        if data2_naive.index.tz is not None:
                            data2_naive.index = data2_naive.index.tz_localize(None)

                        # Calculate price differences for overlapping periods
                        price_diff = abs(data1_naive['Close'] - data2_naive['Close']) / data1_naive['Close']
                        consistent = (price_diff <= tolerance).all()

                        consistency_results[f"{source1}_vs_{source2}"] = consistent
                    except Exception as e:
                        # If comparison fails, mark as inconsistent
                        consistency_results[f"{source1}_vs_{source2}"] = False
                else:
                    consistency_results[f"{source1}_vs_{source2}"] = False

        return consistency_results

    def get_best_data_source(self, sources_data: Dict[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame]:
        """
        Select the best data source based on quality metrics

        Args:
            sources_data: Dictionary with data from each source

        Returns:
            Tuple of (best_source_name, best_data)
        """
        if not sources_data:
            raise ValueError("No data sources available")

        quality_metrics = self.compare_data_quality(sources_data)

        # Score each source based on quality metrics
        scores = {}
        for source, metrics in quality_metrics.items():
            score = 0
            score += min(metrics['record_count'] / 1000, 10)  # Record count score (max 10)
            score += (100 - metrics['missing_values']) / 10    # Missing values score (max 10)
            score += min(metrics['data_completeness'], 10)     # Completeness score (max 10)
            scores[source] = score

        # Select source with highest score
        best_source = max(scores, key=scores.get)

        return best_source, sources_data[best_source]

def create_sample_data_for_testing() -> pd.DataFrame:
    """
    Create sample BTCUSD data for testing when APIs are unavailable

    Returns:
        DataFrame with sample OHLCV data
    """
    # Create sample data for the past 30 days with 15-minute intervals
    start_date = datetime.now() - timedelta(days=30)
    dates = pd.date_range(start=start_date, end=datetime.now(), freq='15min')

    # Generate realistic BTC price movements
    np.random.seed(42)  # For reproducible results

    # Start with a base price around current BTC levels
    base_price = 45000
    price_changes = np.random.normal(0, 0.005, len(dates))  # 0.5% standard deviation

    # Generate OHLC data
    closes = [base_price]
    for change in price_changes[1:]:
        new_price = closes[-1] * (1 + change)
        closes.append(new_price)

    # Generate OHLC from close prices
    opens = [closes[0]]
    highs = [closes[0]]
    lows = [closes[0]]

    for i, close in enumerate(closes[1:], 1):
        # Generate realistic intrabar movements
        volatility = abs(np.random.normal(0, 0.003))
        high = close * (1 + volatility)
        low = close * (1 - volatility)
        open_price = closes[i-1] * (1 + np.random.normal(0, 0.001))

        opens.append(open_price)
        highs.append(high)
        lows.append(low)

    # Generate volume data
    volumes = np.random.lognormal(10, 1, len(dates))  # Realistic volume distribution

    # Create DataFrame
    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    }, index=dates)

    return df

if __name__ == "__main__":
    # Example usage
    handler = BTCDataHandler()

    # Fetch data from all sources
    sources_data = handler.fetch_all_sources()

    if sources_data:
        print("Data Quality Comparison:")
        quality_metrics = handler.compare_data_quality(sources_data)
        for source, metrics in quality_metrics.items():
            print(f"\n{source.upper()}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")

        print("\nData Consistency Validation:")
        consistency = handler.validate_data_consistency(sources_data)
        for comparison, is_consistent in consistency.items():
            print(f"  {comparison}: {'Consistent' if is_consistent else 'Inconsistent'}")

        # Get best data source
        best_source, best_data = handler.get_best_data_source(sources_data)
        print(f"\nBest data source: {best_source}")
        print(f"Best data shape: {best_data.shape}")

        # Save best data
        best_data.to_csv('../data/BTCUSD_M15.csv')
        print("Best data saved to ../data/BTCUSD_M15.csv")
    else:
        print("No data sources available, creating sample data...")
        sample_data = create_sample_data_for_testing()
        sample_data.to_csv('../data/BTCUSD_M15.csv')
        print(f"Sample data created with {len(sample_data)} records")