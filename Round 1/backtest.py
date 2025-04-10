import os
import subprocess
import itertools
import csv

import numpy as np
PERIOD = np.arange(10, 55, 5)
OVERSOLD = np.arange(20, 40, 2)
OVERBOUGHT = np.arange(60, 80, 2)
AGGRESSION_FACTOR = np.arange(0.2, 0.8, 0.1)
LOW_BUY = np.arange(5, 25, 5)
HIGH_SELL = np.arange(75, 95, 5)
BUY_AMOUNT = np.arange(5, 16, 5)

def run_backtest(period, oversold, overbought, aggression_factor, low_buy, high_sell, buy_amount):
    """
    Runs prosperity3bt for one set of parameters.
    Returns the backtester output (stdout) so we can parse it if needed.
    """
    # 1) Set environment variables so trader.py can see them
    os.environ["PERIOD"] = str(period)
    os.environ["OVERSOLD"] = str(oversold)
    os.environ["OVERBOUGHT"] = str(overbought)
    os.environ["AGGRESSION_FACTOR"] = str(aggression_factor)
    os.environ["LOW_BUY"] = str(low_buy)
    os.environ["HIGH_SELL"] = str(high_sell)
    os.environ["BUY_AMOUNT"] = str(buy_amount)

    # 2) Build the command exactly as you do manually:
    #    prosperity3bt trader.py 1
    cmd = ["prosperity3bt", "trader.py", "1"]

    # 3) Run the command and capture output
    output = subprocess.check_output(cmd).decode("utf-8")
    
    # 4) Parse or return the raw output
    return output

if __name__ == "__main__":
    # Create a CSV file to store the results
    csv_filename = 'backtest_results.csv'
    fieldnames = ['period', 'oversold', 'overbought', 'aggression_factor', 'low_buy', 'high_sell', 'buy_amount', 'pnl']
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Loop over all combos of param values
        for period, oversold, overbought, aggression_factor, low_buy, high_sell, buy_amount\
              in itertools.product(PERIOD, OVERSOLD, OVERBOUGHT, AGGRESSION_FACTOR, LOW_BUY, HIGH_SELL, BUY_AMOUNT):
            print(f"Running backtest with PERIOD={period}, OVERSOLD={oversold}, OVERBOUGHT={overbought}, \
                  AGGRESSION_FACTOR={aggression_factor}, LOW_BUY={low_buy}, HIGH_SELL={high_sell}, BUY_AMOUNT={buy_amount}")
            output = run_backtest(period, oversold, overbought, aggression_factor, low_buy, high_sell, buy_amount)
            
            # Debug: Print the raw output
            print("Raw output:")
            print(output)
            
            # Find the final total profit after the "Profit summary" section
            pnl = None
            lines = output.splitlines()
            
            # Look for the line that contains "Total profit:" after "Profit summary"
            profit_summary_found = False
            for line in lines:
                if "Profit summary:" in line:
                    profit_summary_found = True
                    continue
                
                if profit_summary_found and "Total profit:" in line:
                    parts = line.split(":")
                    if len(parts) == 2:
                        try:
                            pnl = float(parts[1].strip().replace(',', ''))
                            print(f"Found final total profit: {pnl}")
                        except ValueError:
                            print(f"Failed to parse final profit value: '{parts[1].strip()}'")
                            pnl = None
                    break
            
            # Collect the results in a dictionary
            row = {
                "period": period,
                "oversold": oversold,
                "overbought": overbought,
                "aggression_factor": aggression_factor,
                "low_buy": low_buy,
                "high_sell": high_sell,
                "buy_amount": buy_amount,
                "pnl": pnl
            }
            print(row)
            print("\n")
            
            # Write the row to the CSV file
            writer.writerow(row)
            
            # Flush the CSV file to make sure data is written immediately
            csvfile.flush()
    
    print(f"Backtest results have been saved to {csv_filename}")