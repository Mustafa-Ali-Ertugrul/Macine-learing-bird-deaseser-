# bot_optimizer.py
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from bot_backtester import BotBacktester
import json
from datetime import datetime

class BotOptimizer:
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.optimization_results = []
        
    def optimize_parameters(self, df: pd.DataFrame, param_ranges: dict) -> pd.DataFrame:
        """Optimize bot parameters using grid search"""
        print("Starting parameter optimization...")
        
        # Generate all parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        combinations = list(product(*param_values))
        
        print(f"Testing {len(combinations)} parameter combinations...")
        
        results = []
        
        for i, params in enumerate(combinations):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(combinations)} ({i/len(combinations)*100:.1f}%)")
            
            # Create parameter dictionary
            param_dict = dict(zip(param_names, params))
            
            try:
                # Run backtest with these parameters
                metrics = self.run_single_backtest(df, param_dict)
                
                if metrics and 'error' not in metrics:
                    result = {
                        **param_dict,
                        **metrics,
                        'profit_factor': self.calculate_profit_factor(metrics),
                        'risk_reward_ratio': self.calculate_risk_reward_ratio(metrics)
                    }
                    results.append(result)
                    
            except Exception as e:
                print(f"Error with parameters {param_dict}: {e}")
                continue
        
        self.optimization_results = results
        return pd.DataFrame(results)
    
    def run_single_backtest(self, df: pd.DataFrame, params: dict) -> dict:
        """Run backtest with specific parameters"""
        backtester = BotBacktester(self.initial_balance)
        
        # Apply parameters
        rsi_buy = params.get('rsi_buy', 30)
        rsi_sell = params.get('rsi_sell', 70)
        stop_loss = params.get('stop_loss_percent', 0.02)
        ma_short = params.get('ma_short', 20)
        ma_long = params.get('ma_long', 50)
        
        # Calculate indicators with custom parameters
        df_copy = df.copy()
        df_copy['rsi'] = self.calculate_rsi(df_copy['close'])
        df_copy[f'ma{ma_short}'] = df_copy['close'].rolling(ma_short).mean()
        df_copy[f'ma{ma_long}'] = df_copy['close'].rolling(ma_long).mean()
        df_copy['trend_bullish'] = df_copy[f'ma{ma_short}'] > df_copy[f'ma{ma_long}']
        df_copy['price_above_ma'] = df_copy['close'] > df_copy[f'ma{ma_short}']
        
        # Run simulation with custom parameters
        return self.simulate_strategy(backtester, df_copy, rsi_buy, rsi_sell, stop_loss)
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def simulate_strategy(self, backtester, df: pd.DataFrame, rsi_buy: int, rsi_sell: int, stop_loss: float) -> dict:
        """Simulate trading strategy with given parameters"""
        position = None
        entry_price = None
        
        for timestamp, row in df.iterrows():
            current_price = row['close']
            current_rsi = row['rsi']
            
            if pd.isna(current_rsi):
                continue
            
            # Record portfolio value
            portfolio_value = backtester.get_portfolio_value(current_price)
            backtester.portfolio_history.append({
                'timestamp': timestamp,
                'price': current_price,
                'portfolio_value': portfolio_value,
                'rsi': current_rsi
            })
            
            # Stop-loss check
            if position == 'LONG' and entry_price:
                loss_percent = (entry_price - current_price) / entry_price
                if loss_percent >= stop_loss:
                    backtester.simulate_trade('SELL', current_price, timestamp, 'Stop-Loss')
                    position = None
                    entry_price = None
                    continue
            
            # Buy signal
            if (current_rsi <= rsi_buy and 
                position != 'LONG' and 
                row['trend_bullish'] and 
                row['price_above_ma']):
                
                backtester.simulate_trade('BUY', current_price, timestamp, 'RSI Buy Signal')
                position = 'LONG'
                entry_price = current_price
            
            # Sell signal
            elif current_rsi >= rsi_sell and position == 'LONG':
                backtester.simulate_trade('SELL', current_price, timestamp, 'RSI Sell Signal')
                position = None
                entry_price = None
        
        # Close any open position
        if position == 'LONG':
            final_price = df['close'].iloc[-1]
            backtester.simulate_trade('SELL', final_price, df.index[-1], 'End of Period')
        
        return backtester.calculate_performance_metrics()
    
    def calculate_profit_factor(self, metrics: dict) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if metrics['total_trades'] == 0:
            return 0
        
        profitable_trades = metrics['profitable_trades']
        losing_trades = metrics['total_trades'] - profitable_trades
        
        if losing_trades == 0:
            return float('inf') if profitable_trades > 0 else 0
        
        # Simplified calculation
        avg_win = abs(metrics['avg_return_per_trade_pct']) if metrics['avg_return_per_trade_pct'] > 0 else 1
        avg_loss = abs(metrics['avg_return_per_trade_pct']) if metrics['avg_return_per_trade_pct'] < 0 else 1
        
        gross_profit = profitable_trades * avg_win
        gross_loss = losing_trades * avg_loss
        
        return gross_profit / gross_loss if gross_loss > 0 else 0
    
    def calculate_risk_reward_ratio(self, metrics: dict) -> float:
        """Calculate risk-reward ratio"""
        if metrics['max_drawdown_pct'] == 0:
            return float('inf') if metrics['total_return_pct'] > 0 else 0
        
        return metrics['total_return_pct'] / metrics['max_drawdown_pct']
    
    def find_best_parameters(self, results_df: pd.DataFrame, metric: str = 'total_return_pct') -> dict:
        """Find best parameters based on specified metric"""
        if results_df.empty:
            return {}
        
        best_idx = results_df[metric].idxmax()
        best_params = results_df.loc[best_idx].to_dict()
        
        return best_params
    
    def plot_optimization_results(self, results_df: pd.DataFrame, save_path: str = 'optimization_results.png'):
        """Plot optimization results"""
        if results_df.empty:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. RSI Buy vs Total Return
        if 'rsi_buy' in results_df.columns:
            sns.boxplot(data=results_df, x='rsi_buy', y='total_return_pct', ax=axes[0,0])
            axes[0,0].set_title('RSI Buy Level vs Total Return')
            axes[0,0].set_ylabel('Total Return (%)')
        
        # 2. RSI Sell vs Total Return
        if 'rsi_sell' in results_df.columns:
            sns.boxplot(data=results_df, x='rsi_sell', y='total_return_pct', ax=axes[0,1])
            axes[0,1].set_title('RSI Sell Level vs Total Return')
            axes[0,1].set_ylabel('Total Return (%)')
        
        # 3. Win Rate vs Total Return
        axes[0,2].scatter(results_df['win_rate_pct'], results_df['total_return_pct'], alpha=0.6)
        axes[0,2].set_xlabel('Win Rate (%)')
        axes[0,2].set_ylabel('Total Return (%)')
        axes[0,2].set_title('Win Rate vs Total Return')
        
        # 4. Max Drawdown vs Total Return
        axes[1,0].scatter(results_df['max_drawdown_pct'], results_df['total_return_pct'], alpha=0.6)
        axes[1,0].set_xlabel('Max Drawdown (%)')
        axes[1,0].set_ylabel('Total Return (%)')
        axes[1,0].set_title('Risk vs Return')
        
        # 5. Sharpe Ratio distribution
        axes[1,1].hist(results_df['sharpe_ratio'], bins=20, alpha=0.7)
        axes[1,1].set_xlabel('Sharpe Ratio')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Sharpe Ratio Distribution')
        
        # 6. Parameter heatmap (if RSI parameters exist)
        if 'rsi_buy' in results_df.columns and 'rsi_sell' in results_df.columns:
            pivot_table = results_df.pivot_table(
                values='total_return_pct', 
                index='rsi_buy', 
                columns='rsi_sell', 
                aggfunc='mean'
            )
            sns.heatmap(pivot_table, annot=True, fmt='.1f', ax=axes[1,2], cmap='RdYlGn')
            axes[1,2].set_title('RSI Parameters Heatmap (Return %)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Optimization results saved to {save_path}")
    
    def generate_optimization_report(self, results_df: pd.DataFrame) -> str:
        """Generate optimization report"""
        if results_df.empty:
            return "No optimization results available"
        
        report = []
        report.append("PARAMETER OPTIMIZATION REPORT")
        report.append("=" * 50)
        report.append(f"Total combinations tested: {len(results_df)}")
        report.append(f"Successful backtests: {len(results_df[results_df['total_return_pct'].notna()])}")
        report.append("")
        
        # Best parameters by different metrics
        metrics = ['total_return_pct', 'win_rate_pct', 'sharpe_ratio', 'profit_factor']
        
        for metric in metrics:
            if metric in results_df.columns:
                best = self.find_best_parameters(results_df, metric)
                report.append(f"Best {metric.replace('_', ' ').title()}:")
                report.append(f"  Value: {best.get(metric, 'N/A'):.2f}")
                if 'rsi_buy' in best:
                    report.append(f"  RSI Buy: {best['rsi_buy']}")
                if 'rsi_sell' in best:
                    report.append(f"  RSI Sell: {best['rsi_sell']}")
                if 'stop_loss_percent' in best:
                    report.append(f"  Stop Loss: {best['stop_loss_percent']:.1%}")
                report.append("")
        
        # Statistics
        report.append("PERFORMANCE STATISTICS:")
        report.append(f"Average Return: {results_df['total_return_pct'].mean():.2f}%")
        report.append(f"Best Return: {results_df['total_return_pct'].max():.2f}%")
        report.append(f"Worst Return: {results_df['total_return_pct'].min():.2f}%")
        report.append(f"Average Win Rate: {results_df['win_rate_pct'].mean():.2f}%")
        report.append(f"Average Sharpe Ratio: {results_df['sharpe_ratio'].mean():.3f}")
        
        return "\n".join(report)
    
    def save_optimization_results(self, results_df: pd.DataFrame, filename: str = 'optimization_results.json'):
        """Save optimization results"""
        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'total_combinations': len(results_df),
            'results': results_df.to_dict('records'),
            'best_parameters': {
                'by_return': self.find_best_parameters(results_df, 'total_return_pct'),
                'by_sharpe': self.find_best_parameters(results_df, 'sharpe_ratio'),
                'by_win_rate': self.find_best_parameters(results_df, 'win_rate_pct')
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        print(f"Optimization results saved to {filename}")

def run_parameter_optimization(days: int = 30):
    """Run comprehensive parameter optimization"""
    print("Starting comprehensive parameter optimization...")
    
    # Get historical data
    backtester = BotBacktester()
    df = backtester.get_historical_data(days)
    
    if df.empty:
        print("Failed to fetch historical data")
        return
    
    print(f"Using {len(df)} data points for optimization")
    
    # Define parameter ranges to test
    param_ranges = {
        'rsi_buy': [20, 25, 30, 35],
        'rsi_sell': [65, 70, 75, 80],
        'stop_loss_percent': [0.01, 0.02, 0.03, 0.05],
        'ma_short': [10, 20, 30],
        'ma_long': [40, 50, 60]
    }
    
    # Run optimization
    optimizer = BotOptimizer()
    results_df = optimizer.optimize_parameters(df, param_ranges)
    
    if results_df.empty:
        print("No successful optimization results")
        return
    
    print(f"\nOptimization completed! {len(results_df)} successful combinations")
    
    # Generate report
    report = optimizer.generate_optimization_report(results_df)
    print("\n" + report)
    
    # Plot results
    optimizer.plot_optimization_results(results_df)
    
    # Save results
    optimizer.save_optimization_results(results_df)
    
    # Show top 5 results
    print("\nTOP 5 PARAMETER COMBINATIONS BY TOTAL RETURN:")
    print("=" * 60)
    top_5 = results_df.nlargest(5, 'total_return_pct')
    
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"{i}. Return: {row['total_return_pct']:.2f}% | "
              f"RSI: {row['rsi_buy']}-{row['rsi_sell']} | "
              f"Stop Loss: {row['stop_loss_percent']:.1%} | "
              f"Win Rate: {row['win_rate_pct']:.1f}%")
    
    return results_df

if __name__ == "__main__":
    # Run optimization for different time periods
    periods = [14, 30]
    
    for days in periods:
        print(f"\n{'='*70}")
        print(f"PARAMETER OPTIMIZATION FOR {days} DAYS")
        print(f"{'='*70}")
        
        try:
            results = run_parameter_optimization(days)
            if results is not None and not results.empty:
                print(f"\nOptimization for {days} days completed successfully")
        except Exception as e:
            print(f"Error in {days}-day optimization: {e}")
        
        print("\n" + "-"*70)