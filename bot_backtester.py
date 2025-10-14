# bot_backtester.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from binance.client import Client
import json
import os
from typing import Dict, List, Tuple

# Import bot configuration
from btc_trading_bot import (
    API_KEY, API_SECRET, SYMBOL, RSI_BUY, RSI_SELL, 
    rsi, moving_average, STOP_LOSS_PERCENT
)

class BotBacktester:
    def __init__(self, initial_balance: float = 10000):
        self.client = Client(API_KEY, API_SECRET, testnet=True)
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.btc_balance = 0
        self.trades = []
        self.portfolio_history = []
        
    def get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Get historical kline data for backtesting"""
        try:
            # Get data for specified days
            klines = self.client.get_historical_klines(
                SYMBOL, 
                Client.KLINE_INTERVAL_1HOUR, 
                f"{days} days ago UTC"
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to proper data types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df['rsi'] = rsi(df['close'])
        df['ma20'] = moving_average(df['close'], 20)
        df['ma50'] = moving_average(df['close'], 50)
        
        # Trend analysis
        df['trend_bullish'] = df['ma20'] > df['ma50']
        df['price_above_ma20'] = df['close'] > df['ma20']
        
        return df
    
    def simulate_trade(self, action: str, price: float, timestamp: pd.Timestamp, reason: str):
        """Simulate a trade execution"""
        if action == 'BUY' and self.current_balance > 0:
            # Buy BTC with available USDT
            btc_amount = (self.current_balance * 0.999) / price  # 0.1% fee
            self.btc_balance += btc_amount
            self.current_balance = 0
            
            trade = {
                'timestamp': timestamp,
                'action': action,
                'price': price,
                'btc_amount': btc_amount,
                'usdt_amount': btc_amount * price,
                'reason': reason,
                'portfolio_value': self.get_portfolio_value(price)
            }
            self.trades.append(trade)
            
        elif action == 'SELL' and self.btc_balance > 0:
            # Sell BTC for USDT
            usdt_amount = (self.btc_balance * price) * 0.999  # 0.1% fee
            self.current_balance = usdt_amount
            
            trade = {
                'timestamp': timestamp,
                'action': action,
                'price': price,
                'btc_amount': self.btc_balance,
                'usdt_amount': usdt_amount,
                'reason': reason,
                'portfolio_value': self.get_portfolio_value(price)
            }
            self.trades.append(trade)
            self.btc_balance = 0
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value"""
        return self.current_balance + (self.btc_balance * current_price)
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """Run the backtesting simulation"""
        position = None
        entry_price = None
        
        for timestamp, row in df.iterrows():
            current_price = row['close']
            current_rsi = row['rsi']
            
            # Skip if RSI is NaN
            if pd.isna(current_rsi):
                continue
            
            # Record portfolio value
            portfolio_value = self.get_portfolio_value(current_price)
            self.portfolio_history.append({
                'timestamp': timestamp,
                'price': current_price,
                'portfolio_value': portfolio_value,
                'rsi': current_rsi
            })
            
            # Stop-loss check
            if position == 'LONG' and entry_price:
                loss_percent = (entry_price - current_price) / entry_price
                if loss_percent >= STOP_LOSS_PERCENT:
                    self.simulate_trade('SELL', current_price, timestamp, 'Stop-Loss')
                    position = None
                    entry_price = None
                    continue
            
            # Buy signal
            if (current_rsi <= RSI_BUY and 
                position != 'LONG' and 
                row['trend_bullish'] and 
                row['price_above_ma20']):
                
                self.simulate_trade('BUY', current_price, timestamp, 'RSI Buy Signal')
                position = 'LONG'
                entry_price = current_price
            
            # Sell signal
            elif current_rsi >= RSI_SELL and position == 'LONG':
                self.simulate_trade('SELL', current_price, timestamp, 'RSI Sell Signal')
                position = None
                entry_price = None
        
        # Close any open position at the end
        if position == 'LONG':
            final_price = df['close'].iloc[-1]
            self.simulate_trade('SELL', final_price, df.index[-1], 'End of Period')
        
        return self.calculate_performance_metrics()
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        final_value = self.get_portfolio_value(self.trades[-1]['price'])
        total_return = (final_value - self.initial_balance) / self.initial_balance
        
        # Calculate trade statistics
        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']
        
        profitable_trades = 0
        total_trades = min(len(buy_trades), len(sell_trades))
        
        trade_returns = []
        for i in range(total_trades):
            if i < len(sell_trades):
                buy_price = buy_trades[i]['price']
                sell_price = sell_trades[i]['price']
                trade_return = (sell_price - buy_price) / buy_price
                trade_returns.append(trade_return)
                if trade_return > 0:
                    profitable_trades += 1
        
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        avg_return_per_trade = np.mean(trade_returns) if trade_returns else 0
        
        # Calculate maximum drawdown
        portfolio_values = [p['portfolio_value'] for p in self.portfolio_history]
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': final_value,
            'total_return_pct': total_return * 100,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate_pct': win_rate,
            'avg_return_per_trade_pct': avg_return_per_trade * 100,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': self.calculate_sharpe_ratio(trade_returns)
        }
    
    def calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Assuming risk-free rate of 2% annually, convert to hourly
        risk_free_rate = 0.02 / (365 * 24)
        
        return (mean_return - risk_free_rate) / std_return
    
    def plot_results(self, save_path: str = 'backtest_results.png'):
        """Plot backtesting results"""
        if not self.portfolio_history:
            print("No data to plot")
            return
        
        df_portfolio = pd.DataFrame(self.portfolio_history)
        df_trades = pd.DataFrame(self.trades)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Price and Portfolio Value
        ax1.plot(df_portfolio['timestamp'], df_portfolio['price'], label='BTC Price', alpha=0.7)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(df_portfolio['timestamp'], df_portfolio['portfolio_value'], 
                     label='Portfolio Value', color='green', alpha=0.8)
        
        # Mark trades
        if not df_trades.empty:
            buy_trades = df_trades[df_trades['action'] == 'BUY']
            sell_trades = df_trades[df_trades['action'] == 'SELL']
            
            ax1.scatter(buy_trades['timestamp'], buy_trades['price'], 
                       color='green', marker='^', s=100, label='Buy', zorder=5)
            ax1.scatter(sell_trades['timestamp'], sell_trades['price'], 
                       color='red', marker='v', s=100, label='Sell', zorder=5)
        
        ax1.set_ylabel('BTC Price (USDT)')
        ax1_twin.set_ylabel('Portfolio Value (USDT)')
        ax1.set_title('BTC Price and Portfolio Performance')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: RSI with signals
        ax2.plot(df_portfolio['timestamp'], df_portfolio['rsi'], label='RSI', color='purple')
        ax2.axhline(y=RSI_BUY, color='green', linestyle='--', alpha=0.7, label=f'Buy Level ({RSI_BUY})')
        ax2.axhline(y=RSI_SELL, color='red', linestyle='--', alpha=0.7, label=f'Sell Level ({RSI_SELL})')
        ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        ax2.set_ylabel('RSI')
        ax2.set_title('RSI Indicator')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Plot 3: Cumulative Returns
        initial_value = df_portfolio['portfolio_value'].iloc[0]
        cumulative_returns = (df_portfolio['portfolio_value'] / initial_value - 1) * 100
        ax3.plot(df_portfolio['timestamp'], cumulative_returns, label='Strategy Returns', color='blue')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_ylabel('Cumulative Returns (%)')
        ax3.set_xlabel('Date')
        ax3.set_title('Cumulative Returns')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Results saved to {save_path}")
    
    def save_results(self, metrics: Dict, filename: str = 'backtest_results.json'):
        """Save backtest results to file"""
        results = {
            'metrics': metrics,
            'trades': self.trades,
            'portfolio_history': self.portfolio_history[-10:],  # Last 10 records
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Detailed results saved to {filename}")

def run_comprehensive_backtest(days: int = 30, custom_params=None):
    """Run a comprehensive backtest"""
    print(f"Starting backtest for the last {days} days...")
    
    backtester = BotBacktester(initial_balance=10000)
    
    # Get historical data
    print("Fetching historical data...")
    df = backtester.get_historical_data(days)
    
    if df.empty:
        print("Failed to fetch historical data")
        return
    
    print(f"Data fetched: {len(df)} records from {df.index[0]} to {df.index[-1]}")
    
    # Calculate indicators
    print("Calculating technical indicators...")
    df = backtester.calculate_indicators(df)
    
    # Use custom parameters if provided
    if custom_params:
        print(f"Using custom parameters: {custom_params}")
        # Temporarily modify global parameters for testing
        global RSI_BUY, RSI_SELL
        original_rsi_buy, original_rsi_sell = RSI_BUY, RSI_SELL
        RSI_BUY = custom_params.get('rsi_buy', RSI_BUY)
        RSI_SELL = custom_params.get('rsi_sell', RSI_SELL)
        
        # Run backtest
        print("Running backtest simulation...")
        metrics = backtester.run_backtest(df)
        
        # Restore original parameters
        RSI_BUY, RSI_SELL = original_rsi_buy, original_rsi_sell
    else:
        # Run backtest with default parameters
        print("Running backtest simulation...")
        metrics = backtester.run_backtest(df)
    
    # Display results
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    
    if 'error' in metrics:
        print(f"Error: {metrics['error']}")
        return
    
    print(f"Initial Balance: ${metrics['initial_balance']:,.2f}")
    print(f"Final Balance: ${metrics['final_balance']:,.2f}")
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Profitable Trades: {metrics['profitable_trades']}")
    print(f"Win Rate: {metrics['win_rate_pct']:.2f}%")
    print(f"Average Return per Trade: {metrics['avg_return_per_trade_pct']:.2f}%")
    print(f"Maximum Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    
    # Plot results
    print("\nGenerating charts...")
    backtester.plot_results()
    
    # Save results
    backtester.save_results(metrics)
    
    # Recommendations
    print("\n" + "="*50)
    print("RECOMMENDATIONS")
    print("="*50)
    
    if metrics['total_return_pct'] > 0:
        print("✅ Strategy is profitable in the tested period")
    else:
        print("❌ Strategy shows losses in the tested period")
    
    if metrics['win_rate_pct'] > 50:
        print("✅ Good win rate")
    else:
        print("⚠️ Low win rate - consider adjusting parameters")
    
    if metrics['max_drawdown_pct'] > 20:
        print("⚠️ High maximum drawdown - consider risk management")
    else:
        print("✅ Acceptable drawdown levels")
    
    if metrics['sharpe_ratio'] > 1:
        print("✅ Good risk-adjusted returns")
    elif metrics['sharpe_ratio'] > 0:
        print("⚠️ Moderate risk-adjusted returns")
    else:
        print("❌ Poor risk-adjusted returns")
    
    return metrics

if __name__ == "__main__":
    # Run backtest for different periods
    periods = [7, 14, 30]
    
    for days in periods:
        print(f"\n{'='*60}")
        print(f"BACKTESTING FOR {days} DAYS")
        print(f"{'='*60}")
        
        try:
            metrics = run_comprehensive_backtest(days)
            if metrics:
                print(f"\nBacktest for {days} days completed successfully")
        except Exception as e:
            print(f"Error in {days}-day backtest: {e}")
        
        print("\n" + "-"*60)