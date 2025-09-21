"""
Cryptocurrency Data Preprocessing Module
Handles preprocessing of crypto market data for trading strategy optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import pytz
from typing import Optional, Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CryptoDataPreprocessor:
    """
    Preprocesses cryptocurrency market data for trading strategy optimization
    Handles 24/7 market characteristics and creates features for strategy analysis
    """

    def __init__(self, timezone: str = "UTC"):
        """
        Initialize the preprocessor with timezone settings

        Args:
            timezone: Target timezone for data processing (default: UTC)
        """
        self.timezone = pytz.timezone(timezone)

    def load_and_preprocess_data(self, file_path: str, symbol: str = "BTCUSD") -> pd.DataFrame:
        """
        Load and preprocess cryptocurrency data from CSV file

        Args:
            file_path: Path to the CSV file containing OHLCV data
            symbol: Trading symbol (default: BTCUSD)

        Returns:
            Preprocessed DataFrame with additional features
        """
        try:
            logger.info(f"Loading data from {file_path}...")

            # Load data
            df = pd.read_csv(file_path, parse_dates=['Datetime'] if 'Datetime' in pd.read_csv(file_path, nrows=0).columns else True)

            # Handle different column naming conventions
            if 'Datetime' in df.columns:
                df.set_index('Datetime', inplace=True)
            elif not isinstance(df.index, pd.DatetimeIndex):
                # Try to use first column as datetime index
                df.set_index(pd.to_datetime(df.iloc[:, 0]), inplace=True)
                df = df.iloc[:, 1:]  # Remove the datetime column

            # Ensure proper column names
            expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df.columns = expected_columns[:len(df.columns)]

            logger.info(f"Loaded {len(df)} records for {symbol}")

            # Apply preprocessing pipeline
            df = self._add_temporal_features(df)
            df = self._add_price_features(df)
            df = self._add_volume_features(df)
            df = self._add_volatility_features(df)
            df = self._add_market_regime_features(df)

            return df

        except Exception as e:
            logger.error(f"Error loading and preprocessing data: {e}")
            raise

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features for crypto market analysis

        Args:
            df: Input DataFrame with OHLCV data

        Returns:
            DataFrame with temporal features added
        """
        logger.info("Adding temporal features...")

        # Ensure timezone-aware datetime index
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')

        # Add basic temporal features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = df.index.dayofweek >= 5

        # Add crypto-specific time features (24/7 market)
        df['hour_of_day'] = df.index.hour
        df['minute_of_hour'] = df.index.minute

        # Add session indicators (though crypto is 24/7, we can identify high activity periods)
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] <= 8)).astype(int)  # 00:00-08:00 UTC
        df['european_session'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)  # 08:00-16:00 UTC
        df['american_session'] = ((df['hour'] >= 16) & (df['hour'] <= 23)).astype(int)  # 16:00-23:00 UTC

        # Add market volatility periods (typically higher during certain hours)
        df['high_volatility_period'] = (
            (df['hour'].isin([0, 1, 2, 3, 4, 5, 6, 7])) |  # Early morning UTC
            (df['hour'].isin([12, 13, 14, 15, 16, 17, 18, 19]))  # Afternoon UTC
        ).astype(int)

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features for strategy analysis

        Args:
            df: Input DataFrame with temporal features

        Returns:
            DataFrame with price features added
        """
        logger.info("Adding price features...")

        # Basic price changes
        df['price_change'] = df['Close'].pct_change()
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

        # Price gaps (though less common in crypto due to 24/7 trading)
        df['gap'] = df['Open'] - df['Close'].shift(1)
        df['gap_pct'] = df['gap'] / df['Close'].shift(1)

        # Price ranges
        df['daily_range'] = df['High'] - df['Low']
        df['daily_range_pct'] = df['daily_range'] / df['Close']

        # Intraday price movements
        df['body_size'] = abs(df['Close'] - df['Open'])
        df['upper_shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['lower_shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']

        # Price momentum features
        for period in [5, 10, 20, 50]:
            df[f'momentum_{period}'] = df['Close'].pct_change(period)
            df[f'roc_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)

        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()

        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        df['bb_std'] = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based features for market analysis

        Args:
            df: Input DataFrame with price features

        Returns:
            DataFrame with volume features added
        """
        logger.info("Adding volume features...")

        # Volume moving averages
        for period in [5, 10, 20, 50]:
            df[f'volume_sma_{period}'] = df['Volume'].rolling(window=period).mean()

        # Volume ratios
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

        # Volume price confirmation
        df['volume_price_trend'] = (df['price_change'] > 0) & (df['Volume'] > df['Volume'].rolling(window=5).mean())

        # On Balance Volume (OBV)
        df['obv'] = 0
        df.loc[df['Close'] > df['Close'].shift(1), 'obv'] = df['Volume']
        df.loc[df['Close'] < df['Close'].shift(1), 'obv'] = -df['Volume']
        df['obv'] = df['obv'].cumsum()

        # Volume Weighted Average Price (VWAP) - simplified
        df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based features for risk assessment

        Args:
            df: Input DataFrame with volume features

        Returns:
            DataFrame with volatility features added
        """
        logger.info("Adding volatility features...")

        # Rolling volatility measures
        for period in [5, 10, 20, 50]:
            df[f'volatility_{period}'] = df['log_return'].rolling(window=period).std() * np.sqrt(period)
            df[f'price_volatility_{period}'] = df['Close'].rolling(window=period).std()

        # Average True Range (ATR) - simplified version
        df['true_range'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )

        for period in [5, 10, 14, 20]:
            df[f'atr_{period}'] = df['true_range'].rolling(window=period).mean()

        # Parkinson volatility (uses high-low range)
        df['parkinson_vol'] = (1 / (4 * np.log(2))) * (df['High'] / df['Low']).apply(np.log) ** 2
        df['parkinson_vol'] = df['parkinson_vol'].rolling(window=20).mean() * np.sqrt(252)  # Annualized

        # Garman-Klass volatility (more sophisticated)
        df['garman_klass_vol'] = (
            0.5 * (df['High'] / df['Low']).apply(np.log) ** 2 -
            (2 * np.log(2) - 1) * (df['Close'] / df['Open']).apply(np.log) ** 2
        )
        df['garman_klass_vol'] = df['garman_klass_vol'].rolling(window=20).mean() * np.sqrt(252)

        return df

    def _add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime classification features

        Args:
            df: Input DataFrame with volatility features

        Returns:
            DataFrame with market regime features added
        """
        logger.info("Adding market regime features...")

        # Trend strength indicators
        df['trend_strength'] = abs(df['momentum_20'])
        df['trend_direction'] = np.sign(df['momentum_20'])

        # Market regime classification based on volatility and trend
        df['regime_volatility'] = pd.qcut(df['volatility_20'], q=3, labels=['low', 'medium', 'high'])

        # Combined regime indicator
        df['market_regime'] = 'sideways'  # default
        df.loc[(df['trend_strength'] > 0.02) & (df['regime_volatility'] == 'low'), 'market_regime'] = 'trending_low_vol'
        df.loc[(df['trend_strength'] > 0.02) & (df['regime_volatility'] == 'medium'), 'market_regime'] = 'trending_medium_vol'
        df.loc[(df['trend_strength'] > 0.02) & (df['regime_volatility'] == 'high'), 'market_regime'] = 'trending_high_vol'
        df.loc[(df['trend_strength'] <= 0.02) & (df['regime_volatility'] == 'low'), 'market_regime'] = 'sideways_low_vol'
        df.loc[(df['trend_strength'] <= 0.02) & (df['regime_volatility'] == 'medium'), 'market_regime'] = 'sideways_medium_vol'
        df.loc[(df['trend_strength'] <= 0.02) & (df['regime_volatility'] == 'high'), 'market_regime'] = 'volatile_sideways'

        # Add regime dummy variables
        regime_dummies = pd.get_dummies(df['market_regime'], prefix='regime')
        df = pd.concat([df, regime_dummies], axis=1)

        return df

    def create_orb_features(self, df: pd.DataFrame, range_start_hour: int = 0,
                           range_end_hour: int = 1, range_minutes: int = 0) -> pd.DataFrame:
        """
        Create Opening Range Breakout (ORB) specific features for crypto markets

        Args:
            df: Input DataFrame with basic features
            range_start_hour: Hour when opening range starts (default: 0 for midnight)
            range_end_hour: Hour when opening range ends (default: 1)
            range_minutes: Additional minutes for range end (default: 0)

        Returns:
            DataFrame with ORB features added
        """
        logger.info("Adding ORB-specific features...")

        # Define opening range period for each day
        df['date'] = df.index.date

        # For crypto, we can use different opening range periods
        # Default is first hour of UTC day (00:00-01:00)
        df['in_opening_range'] = (
            (df['hour'] == range_start_hour) &
            (df['minute_of_hour'] >= 0) &
            (df['hour'] < range_end_hour)
        ) | (
            (df['hour'] == range_end_hour) &
            (df['minute_of_hour'] < range_minutes)
        )

        # Calculate opening range for each day
        opening_ranges = []
        for date in df['date'].unique():
            day_data = df[df['date'] == date]
            range_data = day_data[day_data['in_opening_range']]

            if len(range_data) > 0:
                opening_range = {
                    'date': date,
                    'orb_high': range_data['High'].max(),
                    'orb_low': range_data['Low'].min(),
                    'orb_open': range_data.iloc[0]['Open'],
                    'orb_close': range_data.iloc[-1]['Close'],
                    'orb_volume': range_data['Volume'].sum()
                }
            else:
                # Fallback to first candle of the day
                first_candle = day_data.iloc[0] if len(day_data) > 0 else None
                if first_candle is not None:
                    opening_range = {
                        'date': date,
                        'orb_high': first_candle['High'],
                        'orb_low': first_candle['Low'],
                        'orb_open': first_candle['Open'],
                        'orb_close': first_candle['Close'],
                        'orb_volume': first_candle['Volume']
                    }
                else:
                    opening_range = {
                        'date': date,
                        'orb_high': np.nan,
                        'orb_low': np.nan,
                        'orb_open': np.nan,
                        'orb_close': np.nan,
                        'orb_volume': np.nan
                    }

            opening_ranges.append(opening_range)

        # Create opening range DataFrame
        orb_df = pd.DataFrame(opening_ranges)
        orb_df.set_index('date', inplace=True)

        # Merge with main DataFrame
        df = df.merge(orb_df, left_on='date', right_index=True, how='left')

        # Calculate ORB breakout signals
        df['orb_range'] = df['orb_high'] - df['orb_low']
        df['orb_breakout_up'] = (df['High'] > df['orb_high']) & (df['Close'] > df['orb_high'])
        df['orb_breakout_down'] = (df['Low'] < df['orb_low']) & (df['Close'] < df['orb_low'])

        # ORB-based features
        df['orb_position'] = (df['Close'] - df['orb_low']) / (df['orb_high'] - df['orb_low'])
        df['orb_range_pct'] = df['orb_range'] / df['orb_close']

        return df

    def filter_by_market_regime(self, df: pd.DataFrame, allowed_regimes: List[str] = None) -> pd.DataFrame:
        """
        Filter data based on market regime for strategy optimization

        Args:
            df: Input DataFrame with regime features
            allowed_regimes: List of allowed market regimes (default: all)

        Returns:
            Filtered DataFrame
        """
        if allowed_regimes is None:
            allowed_regimes = ['trending_low_vol', 'trending_medium_vol', 'sideways_low_vol']

        logger.info(f"Filtering data for regimes: {allowed_regimes}")

        regime_mask = df['market_regime'].isin(allowed_regimes)
        filtered_df = df[regime_mask].copy()

        logger.info(f"Filtered from {len(df)} to {len(filtered_df)} records")

        return filtered_df

    def create_rolling_windows(self, df: pd.DataFrame, window_size: int = 20,
                              step_size: int = 1) -> List[pd.DataFrame]:
        """
        Create rolling windows for walk-forward analysis

        Args:
            df: Input DataFrame
            window_size: Size of each rolling window in periods
            step_size: Step size for rolling windows

        Returns:
            List of DataFrames for each window
        """
        logger.info(f"Creating rolling windows (size={window_size}, step={step_size})...")

        windows = []
        start_idx = 0

        while start_idx + window_size <= len(df):
            end_idx = start_idx + window_size
            window_df = df.iloc[start_idx:end_idx].copy()
            windows.append(window_df)
            start_idx += step_size

        logger.info(f"Created {len(windows)} rolling windows")

        return windows

    def save_processed_data(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Save processed data to CSV file

        Args:
            df: Processed DataFrame
            file_path: Output file path
        """
        try:
            logger.info(f"Saving processed data to {file_path}...")

            # Create a copy to avoid modifying original
            save_df = df.copy()

            # Reset index to save datetime as column
            save_df.reset_index(inplace=True)

            # Save to CSV
            save_df.to_csv(file_path, index=False)

            logger.info(f"Successfully saved {len(save_df)} records to {file_path}")

        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise

def main():
    """Example usage of the CryptoDataPreprocessor"""
    preprocessor = CryptoDataPreprocessor()

    # Load and preprocess BTCUSD data
    try:
        df = preprocessor.load_and_preprocess_data('../data/BTCUSD_M15.csv', 'BTCUSD')

        # Add ORB features (using first hour of UTC day as opening range)
        df = preprocessor.create_orb_features(df, range_start_hour=0, range_end_hour=1)

        # Filter for favorable trading regimes
        df_filtered = preprocessor.filter_by_market_regime(df)

        # Save processed data
        preprocessor.save_processed_data(df_filtered, '../data/BTCUSD_M15_processed.csv')

        print(f"Preprocessing completed successfully!")
        print(f"Original data shape: {df.shape}")
        print(f"Filtered data shape: {df_filtered.shape}")
        print(f"Available features: {len(df_filtered.columns)}")

    except Exception as e:
        print(f"Error in preprocessing: {e}")

if __name__ == "__main__":
    main()