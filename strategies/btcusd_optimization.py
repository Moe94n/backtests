import pandas as pd
import numpy as np
from datetime import datetime, time
import pandas_ta as ta
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product
import warnings
warnings.filterwarnings('ignore')

def get_price_data():
    """Download BTCUSD data from Yahoo Finance"""
    print('Downloading BTCUSD data...')
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(days=60)

    ticker = yf.Ticker('BTC-USD')
    df = ticker.history(start=start_date, end=end_date, interval='1h')

    df = df.reset_index()
    df = df.rename(columns={
        'Date': 'Datetime',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    })

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('America/New_York')

    return df

def calculate_inputs(df, range_start_time, range_end_time):
    """Calculate strategy inputs with flexible opening range"""
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

def generate_signals(df, s, atr_sl, tp_ratio, range_start_time, range_end_time, latest_entry_time):
    """Generate trading signals with flexible parameters"""
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

def generate_trades(df, s, mult, risk_per_trade, exit_eod, trailing_stop, take_partial):
    """Generate trades with flexible parameters"""
    trades_list = []
    trade_open = False
    balance = 100  # Starting balance
    equity = 100
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

def backtest(price, params):
    """Run backtest with given parameters"""
    range_start_time = params['range_start_time']
    range_end_time = params['range_end_time']
    atr_sl = params['atr_sl']
    tp_ratio = params['tp_ratio']
    mult = params['mult']
    risk_per_trade = params['risk_per_trade']
    exit_eod = params['exit_eod']
    trailing_stop = params['trailing_stop']
    take_partial = params['take_partial']
    latest_entry_time = params['latest_entry_time']

    price_copy = price.copy()
    price_copy = calculate_inputs(price_copy, range_start_time, range_end_time)

    systems = ['Strat']
    for s in systems:
        price_copy = generate_signals(price_copy, s, atr_sl, tp_ratio, range_start_time, range_end_time, latest_entry_time)
        price_copy, trades = generate_trades(price_copy, s, mult, risk_per_trade, exit_eod, trailing_stop, take_partial)

    for s in systems:
        price_copy[f'{s}_Peak'] = price_copy[f'{s}_Bal'].cummax()
        price_copy[f'{s}_DD'] = price_copy[f'{s}_Bal'] - price_copy[f'{s}_Peak']

    return price_copy, trades

def calculate_metrics(result, trades):
    """Calculate comprehensive performance metrics"""
    metrics = {}

    if len(result) == 0 or len(trades) == 0:
        return {
            'total_return': 0,
            'cagr': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_trades': 0,
            'sharpe_ratio': 0,
            'avg_win': 0,
            'avg_loss': 0
        }

    years = (result.index[-1] - result.index[0]).days / 365.25
    total_return = ((result['Strat_Bal'].iloc[-1] / result['Strat_Bal'].iloc[0]) - 1) * 100
    cagr = ((((result['Strat_Bal'].iloc[-1] / result['Strat_Bal'].iloc[0]) ** (1/years)) - 1) * 100) if years > 0 else 0
    max_dd = ((result['Strat_DD'] / result['Strat_Peak']).min()) * 100

    win_rate = (trades['Strat_Return'] > 1).sum() / len(trades) * 100

    total_profit = trades[trades['PnL'] > 0]['PnL'].sum()
    total_loss = abs(trades[trades['PnL'] < 0]['PnL'].sum())
    profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')

    avg_win = trades[trades['PnL'] > 0]['PnL'].mean()
    avg_loss = trades[trades['PnL'] < 0]['PnL'].mean()

    # Calculate Sharpe ratio (simplified)
    returns = trades['Strat_Return'] - 1
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365) if returns.std() != 0 else 0

    return {
        'total_return': total_return,
        'cagr': cagr,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': len(trades),
        'sharpe_ratio': sharpe_ratio,
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }

def optimize_strategy():
    """Run comprehensive parameter optimization"""
    print('Starting BTCUSD ORB Strategy Optimization...')

    # Get data
    price = get_price_data()
    print(f'Data shape: {price.shape}')

    # Define parameter ranges for optimization
    range_start_times = [time(0, 0), time(1, 0), time(2, 0), time(6, 0), time(12, 0)]
    range_end_times = [time(1, 0), time(2, 0), time(3, 0), time(7, 0), time(13, 0)]
    atr_sl_range = [0.5, 1.0, 1.5, 2.0]
    tp_ratio_range = [1.0, 1.5, 2.0, 2.5, 3.0]
    mult_range = [1.0, 1.5, 2.0, 2.5]
    risk_range = [0.01, 0.02, 0.03]
    latest_entry_times = [time(23, 59), time(20, 0), time(18, 0)]

    # Create all parameter combinations
    param_combinations = list(product(range_start_times, range_end_times, atr_sl_range,
                                    tp_ratio_range, mult_range, risk_range, latest_entry_times))

    print(f'Testing {len(param_combinations)} parameter combinations...')

    results = []

    for i, (range_start, range_end, atr_sl, tp_ratio, mult, risk, latest_entry) in enumerate(param_combinations):
        params = {
            'range_start_time': range_start,
            'range_end_time': range_end,
            'atr_sl': atr_sl,
            'tp_ratio': tp_ratio,
            'mult': mult,
            'risk_per_trade': risk,
            'exit_eod': False,
            'trailing_stop': True,
            'take_partial': False,
            'latest_entry_time': latest_entry
        }

        try:
            result, trades = backtest(price, params)
            metrics = calculate_metrics(result, trades)
            metrics.update(params)
            results.append(metrics)

            if (i + 1) % 50 == 0:
                print(f'Progress: {i + 1}/{len(param_combinations)} combinations tested')

        except Exception as e:
            print(f'Error with parameters {params}: {e}')
            continue

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by different metrics to find best combinations
    top_by_return = results_df.sort_values('total_return', ascending=False).head(10)
    top_by_sharpe = results_df.sort_values('sharpe_ratio', ascending=False).head(10)
    top_by_profit_factor = results_df.sort_values('profit_factor', ascending=False).head(10)
    top_by_win_rate = results_df.sort_values('win_rate', ascending=False).head(10)

    return results_df, top_by_return, top_by_sharpe, top_by_profit_factor, top_by_win_rate

def main():
    """Main optimization function"""
    results_df, top_by_return, top_by_sharpe, top_by_profit_factor, top_by_win_rate = optimize_strategy()

    print('\n=== OPTIMIZATION RESULTS ===')
    print(f'Total combinations tested: {len(results_df)}')
    print()

    print('TOP 5 BY TOTAL RETURN:')
    print(top_by_return[['total_return', 'cagr', 'max_drawdown', 'win_rate', 'profit_factor', 'total_trades']].head())
    print()

    print('TOP 5 BY SHARPE RATIO:')
    print(top_by_sharpe[['sharpe_ratio', 'total_return', 'cagr', 'max_drawdown', 'win_rate', 'profit_factor']].head())
    print()

    print('TOP 5 BY PROFIT FACTOR:')
    print(top_by_profit_factor[['profit_factor', 'total_return', 'cagr', 'max_drawdown', 'win_rate']].head())
    print()

    print('TOP 5 BY WIN RATE:')
    print(top_by_win_rate[['win_rate', 'total_return', 'cagr', 'max_drawdown', 'profit_factor']].head())
    print()

    # Save results
    results_df.to_csv('BTCUSD_optimization_results.csv', index=False)
    top_by_return.to_csv('BTCUSD_top_by_return.csv', index=False)
    top_by_sharpe.to_csv('BTCUSD_top_by_sharpe.csv', index=False)

    print('Optimization results saved to CSV files')
    print('Best parameters identified for further testing')

    return results_df

if __name__ == "__main__":
    main()