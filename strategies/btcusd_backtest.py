import pandas as pd
import numpy as np
from datetime import datetime, time
import pandas_ta as ta
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration
symbols = ['BTCUSD']
systems = ['Strat']
starting_balance = 100
risk_per_trade = 0.02

# Crypto-specific parameters (adjusted for hourly data)
range_start_time = time(0, 0)  # 00:00 UTC
range_end_time = time(1, 0)    # 01:00 UTC
latest_entry_time = time(23, 59)
timezone = 'America/New_York'

exit_eod = False
trailing_stop = True
take_partial = False
trailing_multiplier = 1.5

# Download BTCUSD data from Yahoo Finance
def get_price_data(symbol):
    print(f'Downloading {symbol} data...')
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(days=60)  # Yahoo Finance limit for 15m data

    ticker = yf.Ticker('BTC-USD')
    df = ticker.history(start=start_date, end=end_date, interval='1h')

    # Reset index to make Datetime a column
    df = df.reset_index()

    # Rename columns to match expected format
    df = df.rename(columns={
        'Date': 'Datetime',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    })

    # Ensure Datetime is in the right format
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')

    # Convert to the specified timezone
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(timezone)

    return df

def calculate_inputs(df):
    df.index = pd.to_datetime(df.index)
    df['Date'] = df.index.date
    df['Time'] = df.index.time
    df['Last_Candle'] = df['Date'] != df['Date'].shift(-1)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    df_open_range = df[df['Time'] == range_start_time]
    opening_range = df_open_range.groupby('Date').agg(
        Open_Range_High=('High', 'max'),
        Open_Range_Low=('Low', 'min')
    )

    df = df.join(opening_range, on='Date')
    df = df.drop(['Time', 'Date'], axis=1)
    return df

def generate_signals(df, s, atr_sl, tp_ratio):
    df['Hour'] = df.index.hour
    df['Breakout_Above'] = df['Close'] - df['Open_Range_High']
    df['Open_Range_Width'] = df['Open_Range_High'] - df['Open_Range_Low']

    c1 = (df['Open'] <= df['Open_Range_High']) & (df['Close'] > df['Open_Range_High'])
    c2 = (df.index.time >= range_end_time) & (df.index.time < time(23, 59))
    c3 = df['Open_Range_High'].notna()
    c4 = df.index.time <= latest_entry_time
    c5 = df['ATR'].notna()

    if s == 'Strat':
        df[f'{s}_Signal'] = c1.shift(1) & c2 & c3 & c4 & c5.shift(1)
        df['SL'] = df['Open_Range_Low']
        stop_dist = df['Open'] - df['SL']
        df['TP'] = df['Open'] + stop_dist * tp_ratio

    return df

def generate_trades(df, s, mult):
    trades_list = []
    trade_open = False
    balance = starting_balance
    equity = starting_balance
    balance_history = []
    equity_history = []
    trailing = False

    open_prices = df['Open'].values
    high_prices = df['High'].values
    low_prices = df['Low'].values
    close_prices = df['Close'].values
    atr_values = df['ATR'].values if 'ATR' in df.columns else None
    prev_atr_values = df['ATR'].shift(1).values if 'ATR' in df.columns else None
    sl_values = df['SL'].values
    tp_values = df['TP'].values
    signal_values = df[f'{s}_Signal'].values
    range_high = df['Open_Range_High'].values
    range_low = df['Open_Range_Low'].values
    last_candle_values = df['Last_Candle'].values
    index_values = df.index

    for i in range(len(df)):
        if not trade_open and signal_values[i]:
            entry_date = index_values[i]
            entry_price = open_prices[i]
            sl = sl_values[i]
            tp = tp_values[i]
            risk_amount = balance * risk_per_trade
            position_size = 0.01 if entry_price == sl else risk_amount / abs(entry_price - sl)
            trade_open = True
            trailing = False

        if trade_open:
            low = low_prices[i]
            high = high_prices[i]
            open_price = open_prices[i]
            close = close_prices[i]
            last_candle = last_candle_values[i]

            floating_pnl = (high - entry_price) * position_size
            equity = balance + floating_pnl

            if low <= sl:
                exit_price = open_price if open_price <= sl else sl
                trade_open = False
            elif high >= tp:
                if trailing_stop:
                    trailing = True
                    if take_partial:
                        partial_exit_price = open_price if open_price >= tp else tp
                        position_size *= 0.5
                        pnl = (partial_exit_price - entry_price) * position_size
                        balance += pnl
                    tp = 100000000000
                else:
                    exit_price = open_price if open_price >= tp else tp
                    trade_open = False
            elif exit_eod and last_candle:
                exit_price = close
                trade_open = False
            elif trailing:
                new_stop = open_price - (prev_atr_values[i] * mult)
                if new_stop > sl:
                    sl = new_stop

            if not trade_open:
                exit_date = index_values[i]
                trade_open = False
                pnl = (exit_price - entry_price) * position_size
                balance += pnl
                trade = [entry_date, entry_price, exit_date, exit_price, position_size, pnl, balance, True]
                trades_list.append(trade)

        balance_history.append(balance)
        equity_history.append(equity)

    trades = pd.DataFrame(trades_list, columns=['Entry_Date', 'Entry_Price', 'Exit_Date', 'Exit_Price', 'Position_Size', 'PnL', 'Balance', 'Sys_Trade'])
    trades[f'{s}_Return'] = trades.Balance / trades.Balance.shift(1)

    dur = []
    for i, row in trades.iterrows():
        d1 = row.Entry_Date
        d2 = row.Exit_Date
        dur.append(np.busday_count(d1.date(), d2.date()) + 1)
    trades[f'{s}_Duration'] = dur

    returns = pd.DataFrame(index=trades.Exit_Date)
    entries = pd.DataFrame(index=trades.Entry_Date)

    entries[f'{s}_Entry_Price'] = pd.Series(trades.Entry_Price).values
    returns[f'{s}_Ret'] = pd.Series(trades[f'{s}_Return']).values
    returns[f'{s}_Trade'] = pd.Series(trades.Sys_Trade).values
    returns[f'{s}_Duration'] = pd.Series(trades[f'{s}_Duration']).values
    returns[f'{s}_PnL'] = pd.Series(trades.PnL).values
    returns[f'{s}_Balance'] = pd.Series(trades.Balance).values

    df = pd.concat([df, returns, entries], axis=1)
    df[f'{s}_Ret'] = df[f'{s}_Ret'].fillna(1)
    df[f'{s}_Trade'] = df[f'{s}_Trade'].infer_objects(copy=False)

    df[f'{s}_Bal'] = pd.Series(balance_history, index=df.index).ffill()
    df[f'{s}_Equity'] = pd.Series(equity_history, index=df.index).ffill()

    active_trades = np.where(df[f'{s}_Trade'] == True, True, False)
    df[f'{s}_In_Market'] = df[f'{s}_Trade'].copy()

    for count, t in enumerate(active_trades):
        if t == True:
            dur_val = df[f'{s}_Duration'].iat[count]
            for i in range(int(dur_val)):
                df[f'{s}_In_Market'].iat[count - i] = True

    return df, trades

def backtest(price, atr_sl, tp_ratio, mult):
    price = calculate_inputs(price)

    for s in systems:
        price = generate_signals(price, s, atr_sl, tp_ratio)
        price, trades = generate_trades(price, s, mult)

    for s in systems:
        price[f'{s}_Peak'] = price[f'{s}_Bal'].cummax()
        price[f'{s}_DD'] = price[f'{s}_Bal'] - price[f'{s}_Peak']

    return price, trades

# Main execution
if __name__ == "__main__":
    print('Starting BTCUSD ORB backtest...')

    # Get data
    price = get_price_data('BTCUSD')
    print(f'Data shape: {price.shape}')
    print(f'Date range: {price.index.min()} to {price.index.max()}')

    # Save data to CSV for reference
    price.to_csv('BTCUSD_M15_data.csv')
    print('Data saved to BTCUSD_M15_data.csv')

    # Run backtest with one parameter combination for testing
    print('Running backtest...')
    result, trades = backtest(price, 1.0, 1.5, 1.0)

    print('Backtest completed!')
    print(f'Number of trades: {len(trades)}')
    print(f'Final balance: ${result["Strat_Bal"].iloc[-1]:.2f}')
    print(f'Starting balance: ${result["Strat_Bal"].iloc[0]:.2f}')

    # Calculate performance metrics
    years = (result.index[-1] - result.index[0]).days / 365.25
    total_return = ((result['Strat_Bal'].iloc[-1] / result['Strat_Bal'].iloc[0]) - 1) * 100
    cagr = ((((result['Strat_Bal'].iloc[-1] / result['Strat_Bal'].iloc[0]) ** (1/years)) - 1) * 100)
    max_dd = ((result['Strat_DD'] / result['Strat_Peak']).min()) * 100

    win_rate = (trades['Strat_Return'] > 1).sum() / len(trades) * 100

    print(f'Total Return: {total_return:.2f}%')
    print(f'CAGR: {cagr:.2f}%')
    print(f'Max Drawdown: {max_dd:.2f}%')
    print(f'Win Rate: {win_rate:.2f}%')

    # Save results
    trades.to_csv('BTCUSD_trades.csv', index=False)
    result.to_csv('BTCUSD_result.csv')
    print('Results saved to BTCUSD_trades.csv and BTCUSD_result.csv')
    print('BTCUSD ORB backtest completed successfully!')