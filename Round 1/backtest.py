import os
import subprocess
import itertools
import csv
import numpy as np
import multiprocessing
import time
import random
from functools import partial

# Define parameter ranges - consider reducing the grid size for initial testing
PERIOD = np.arange(10, 55, 5)
OVERSOLD = np.arange(20, 40, 2)
OVERBOUGHT = np.arange(60, 80, 2)
AGGRESSION_FACTOR = np.arange(0.2, 0.8, 0.1)
LOW_BUY = np.arange(5, 25, 5)
HIGH_SELL = np.arange(75, 95, 5)
BUY_AMOUNT = np.arange(5, 16, 5)

# Set this to True to use sampling instead of full grid search
USE_SAMPLING = True
# Number of parameter combinations to sample if USE_SAMPLING is True
SAMPLE_SIZE = 1000
# Set early stopping threshold - if profit is below this, stop the process early
EARLY_STOP_THRESHOLD = -10000

def run_backtest(params):
    """
    Runs prosperity3bt for one set of parameters.
    Returns the parameters and profit in a dictionary.
    """
    period, oversold, overbought, aggression_factor, low_buy, high_sell, buy_amount = params
    
    # Print parameter combinations being tested
    print(f"Testing: PERIOD={period}, OVERSOLD={oversold}, OVERBOUGHT={overbought}, "
          f"AGGRESSION_FACTOR={aggression_factor}, LOW_BUY={low_buy}, HIGH_SELL={high_sell}, "
          f"BUY_AMOUNT={buy_amount}")
    
    # 1) Set environment variables so trader.py can see them
    os.environ["PERIOD"] = str(period)
    os.environ["OVERSOLD"] = str(oversold)
    os.environ["OVERBOUGHT"] = str(overbought)
    os.environ["AGGRESSION_FACTOR"] = str(aggression_factor)
    os.environ["LOW_BUY"] = str(low_buy)
    os.environ["HIGH_SELL"] = str(high_sell)
    os.environ["BUY_AMOUNT"] = str(buy_amount)

    # 2) Build the command exactly as you do manually:
    cmd = ["prosperity3bt", "optimized_trader.py", "1"]  # Use trader.py by default, change to optimized_trader.py if needed

    # 3) Run the command and capture output
    try:
        start_time = time.time()
        output = subprocess.check_output(cmd, stderr=subprocess.PIPE).decode("utf-8")
        elapsed_time = time.time() - start_time

        # 4) Parse the output to extract the final profit number
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
                    except ValueError:
                        pnl = None
                break
        
        # Create result dictionary with all required fields
        result = {
            "period": period,
            "oversold": oversold,
            "overbought": overbought,
            "aggression_factor": aggression_factor,
            "low_buy": low_buy,
            "high_sell": high_sell,
            "buy_amount": buy_amount,
            "pnl": pnl,
            "runtime": elapsed_time
        }
        
        print(f"Completed with PnL: {pnl}, Runtime: {elapsed_time:.2f}s")
        return result
    
    except subprocess.CalledProcessError as e:
        print(f"Error running backtest: {e}")
        print(f"Error output: {e.stderr.decode('utf-8') if e.stderr else 'None'}")
        return {
            "period": period,
            "oversold": oversold,
            "overbought": overbought,
            "aggression_factor": aggression_factor,
            "low_buy": low_buy,
            "high_sell": high_sell,
            "buy_amount": buy_amount,
            "pnl": None,
            "runtime": None,
            "error": str(e)
        }

def write_results_to_csv(results, csv_filename):
    """Write backtest results to CSV file with all correct column names"""
    # Define the fieldnames for the CSV
    fieldnames = ['period', 'oversold', 'overbought', 'aggression_factor', 
                  'low_buy', 'high_sell', 'buy_amount', 'pnl', 'runtime']
    
    # Check if file exists to determine if we need to write the header
    file_exists = os.path.isfile(csv_filename) and os.path.getsize(csv_filename) > 0
    
    # Open the CSV file in append mode
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if file doesn't exist or is empty
        if not file_exists:
            writer.writeheader()
        
        # Write result rows
        if isinstance(results, list):
            for result in results:
                # Filter the result dictionary to only include fields in fieldnames
                filtered_result = {k: v for k, v in result.items() if k in fieldnames}
                writer.writerow(filtered_result)
        else:
            # Filter the result dictionary to only include fields in fieldnames
            filtered_result = {k: v for k, v in results.items() if k in fieldnames}
            writer.writerow(filtered_result)
        
        # Ensure data is written to disk
        csvfile.flush()

if __name__ == "__main__":
    # Define CSV filename for results
    csv_filename = 'backtest_results_weighted.csv'
    
    # Create the CSV file with headers (always create a new file)
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'period', 'oversold', 'overbought', 'aggression_factor', 
            'low_buy', 'high_sell', 'buy_amount', 'pnl', 'runtime'
        ])
        writer.writeheader()
    
    start_time = time.time()
    
    # Generate parameter combinations
    if (USE_SAMPLING):
        # Random sampling of parameter space
        all_combinations = list(itertools.product(
            PERIOD, OVERSOLD, OVERBOUGHT, AGGRESSION_FACTOR, LOW_BUY, HIGH_SELL, BUY_AMOUNT
        ))
        
        # Ensure OVERSOLD < OVERBOUGHT for all combinations
        valid_combinations = [combo for combo in all_combinations if combo[1] < combo[2]]
        
        if len(valid_combinations) > SAMPLE_SIZE:
            param_combinations = random.sample(valid_combinations, SAMPLE_SIZE)
        else:
            param_combinations = valid_combinations
        
        print(f"Randomly sampled {len(param_combinations)} parameter combinations")
    else:
        # Full grid search with constraint that OVERSOLD < OVERBOUGHT
        param_combinations = [
            (period, oversold, overbought, aggression_factor, low_buy, high_sell, buy_amount)
            for period, oversold, overbought, aggression_factor, low_buy, high_sell, buy_amount
            in itertools.product(PERIOD, OVERSOLD, OVERBOUGHT, AGGRESSION_FACTOR, LOW_BUY, HIGH_SELL, BUY_AMOUNT)
            if oversold < overbought
        ]
        
        print(f"Generated {len(param_combinations)} parameter combinations for grid search")
    
    # Determine number of processes to use (leave one core free)
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_processes} processes for parallel backtest execution")
    
    # Run backtests in parallel using a process pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Process results as they complete
        batch_size = 5  # Process results in small batches to write to CSV regularly
        results_batch = []
        
        # Use imap_unordered for slightly better performance
        for result in pool.imap_unordered(run_backtest, param_combinations):
            results_batch.append(result)
            
            # Write results to CSV in batches
            if len(results_batch) >= batch_size:
                write_results_to_csv(results_batch, csv_filename)
                results_batch = []
        
        # Write any remaining results
        if results_batch:
            write_results_to_csv(results_batch, csv_filename)
    
    total_time = time.time() - start_time
    print(f"All backtests completed in {total_time:.2f} seconds")
    print(f"Results saved to {csv_filename}")