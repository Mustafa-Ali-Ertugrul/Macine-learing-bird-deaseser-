# run_bot_analysis.py
"""
Bot Analysis Runner - Test and optimize the trading bot

This script provides a simple interface to:
1. Run backtests on historical data
2. Optimize parameters
3. Generate performance reports
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from bot_backtester import run_comprehensive_backtest
    from bot_optimizer import run_parameter_optimization
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required packages are installed:")
    print("pip install matplotlib seaborn pandas numpy python-binance")
    sys.exit(1)

def print_menu():
    """Print the main menu"""
    print("\n" + "="*60)
    print("BTC TRADING BOT ANALYSIS SUITE")
    print("="*60)
    print("1. Run Quick Backtest (7 days)")
    print("2. Run Standard Backtest (30 days)")
    print("3. Run Extended Backtest (60 days)")
    print("4. Optimize Parameters (14 days)")
    print("5. Optimize Parameters (30 days)")
    print("6. Run All Tests")
    print("7. Exit")
    print("-"*60)

def run_quick_backtest():
    """Run a quick 7-day backtest"""
    print("\nRunning quick backtest (7 days)...")
    try:
        metrics = run_comprehensive_backtest(7)
        if metrics:
            print("\n✅ Quick backtest completed successfully!")
            return True
    except Exception as e:
        print(f"❌ Error in quick backtest: {e}")
        return False

def run_standard_backtest():
    """Run a standard 30-day backtest"""
    print("\nRunning standard backtest (30 days)...")
    try:
        metrics = run_comprehensive_backtest(30)
        if metrics:
            print("\n✅ Standard backtest completed successfully!")
            return True
    except Exception as e:
        print(f"❌ Error in standard backtest: {e}")
        return False

def run_extended_backtest():
    """Run an extended 60-day backtest"""
    print("\nRunning extended backtest (60 days)...")
    try:
        metrics = run_comprehensive_backtest(60)
        if metrics:
            print("\n✅ Extended backtest completed successfully!")
            return True
    except Exception as e:
        print(f"❌ Error in extended backtest: {e}")
        return False

def run_parameter_optimization_14():
    """Run parameter optimization on 14 days"""
    print("\nRunning parameter optimization (14 days)...")
    print("⚠️ This may take several minutes...")
    try:
        results = run_parameter_optimization(14)
        if results is not None and not results.empty:
            print("\n✅ Parameter optimization completed successfully!")
            return True
    except Exception as e:
        print(f"❌ Error in parameter optimization: {e}")
        return False

def run_parameter_optimization_30():
    """Run parameter optimization on 30 days"""
    print("\nRunning parameter optimization (30 days)...")
    print("⚠️ This may take several minutes...")
    try:
        results = run_parameter_optimization(30)
        if results is not None and not results.empty:
            print("\n✅ Parameter optimization completed successfully!")
            return True
    except Exception as e:
        print(f"❌ Error in parameter optimization: {e}")
        return False

def run_all_tests():
    """Run all available tests"""
    print("\nRunning all tests...")
    print("This will take some time. Please be patient.")
    
    tests = [
        ("Quick Backtest", run_quick_backtest),
        ("Standard Backtest", run_standard_backtest),
        ("Parameter Optimization (14 days)", run_parameter_optimization_14)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔄 Starting {test_name}...")
        success = test_func()
        results.append((test_name, success))
        
        if success:
            print(f"✅ {test_name} completed")
        else:
            print(f"❌ {test_name} failed")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("Checking prerequisites...")
    
    # Check if API keys are set
    api_key = os.getenv('BINANCE_KEY')
    api_secret = os.getenv('BINANCE_SECRET')
    
    if not api_key or not api_secret:
        print("⚠️ Warning: API keys not found in environment variables")
        print("Some tests may fail without proper API configuration")
        print("\nTo set API keys:")
        print('$env:BINANCE_KEY="your_api_key"')
        print('$env:BINANCE_SECRET="your_secret_key"')
        return False
    else:
        print("✅ API keys found")
        return True

def main():
    """Main function"""
    print(f"Bot Analysis Suite - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    api_configured = check_prerequisites()
    
    if not api_configured:
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    while True:
        print_menu()
        
        try:
            choice = input("Select an option (1-7): ").strip()
            
            if choice == '1':
                run_quick_backtest()
            elif choice == '2':
                run_standard_backtest()
            elif choice == '3':
                run_extended_backtest()
            elif choice == '4':
                run_parameter_optimization_14()
            elif choice == '5':
                run_parameter_optimization_30()
            elif choice == '6':
                run_all_tests()
            elif choice == '7':
                print("\nThank you for using Bot Analysis Suite!")
                print("Generated files:")
                print("- backtest_results.png (charts)")
                print("- backtest_results.json (detailed results)")
                print("- optimization_results.png (optimization charts)")
                print("- optimization_results.json (optimization data)")
                break
            else:
                print("❌ Invalid choice. Please select 1-7.")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            continue
        
        # Pause before showing menu again
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()