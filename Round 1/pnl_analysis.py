import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import os
import seaborn as sns

# Set the style for the plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)
plt.rcParams['font.size'] = 12

# Read the CSV file with backtest results
file_path = os.path.join(os.path.dirname(__file__), 'backtest_results.csv')
data = pd.read_csv(file_path)

# Create a figure with 6 subplots (2x3 grid)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()  # Flatten to make indexing easier

# Parameters to analyze (x-axes)
params = ['period', 'oversold', 'overbought', 'aggression_factor', 'low_buy', 'high_sell']

# Function to format PNL values on y-axis
def format_pnl(x, pos):
    if abs(x) >= 1000:
        return f'{int(x/1000)}k'
    return f'{int(x)}'

# Loop through parameters and create plots
for i, param in enumerate(params):
    # Get the current axis
    ax = axes[i]
    
    # Create scatter plot
    scatter = ax.scatter(data[param], data['pnl'], 
                         alpha=0.6, 
                         c=data['pnl'], 
                         cmap='viridis',
                         s=50)
    
    # Calculate and plot trendline
    z = np.polyfit(data[param], data['pnl'], 1)
    p = np.poly1d(z)
    ax.plot(data[param], p(data[param]), "r--", alpha=0.8, linewidth=2)
    
    # Find the best performing parameter values (top 3)
    param_groups = data.groupby(param)['pnl'].mean().sort_values(ascending=False)
    best_values = param_groups.index[:3].tolist()
    
    # Highlight the best parameter values
    for value in best_values:
        subset = data[data[param] == value]
        avg_pnl = subset['pnl'].mean()
        ax.axvline(x=value, color='green', linestyle='--', alpha=0.5)
        ax.text(value, data['pnl'].min(), f"Best: {value}\nAvg PNL: {int(avg_pnl)}", 
                ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7))
    
    # Calculate correlation coefficient
    correlation = data[param].corr(data['pnl'])
    
    # Add labels and title
    ax.set_xlabel(f"{param.replace('_', ' ').title()}", fontsize=14)
    ax.set_ylabel("PNL", fontsize=14)
    ax.set_title(f"PNL vs {param.replace('_', ' ').title()}\nCorrelation: {correlation:.2f}", fontsize=16)
    
    # Format y-axis with 'k' for thousands
    ax.yaxis.set_major_formatter(FuncFormatter(format_pnl))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add text annotation showing best parameter value
    best_value = param_groups.index[0]
    best_pnl = param_groups.iloc[0]
    ax.annotate(f'Best value: {best_value} (avg PNL: {int(best_pnl)})',
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                ha='left', va='top', fontsize=12)

# Adjust layout
plt.tight_layout()
fig.suptitle('Parameter Impact on PNL', fontsize=20, y=1.02)

# Save and show the figure
plt.savefig(os.path.join(os.path.dirname(__file__), 'pnl_parameter_analysis.png'), dpi=300, bbox_inches='tight')
plt.show()

# Create a summary dataframe of parameter performance
summary = []
for param in params:
    # Calculate correlation
    correlation = data[param].corr(data['pnl'])
    
    # Group by parameter and find best values
    param_groups = data.groupby(param)['pnl'].mean().sort_values(ascending=False)
    
    # Add top 3 values to summary
    for rank, (value, avg_pnl) in enumerate(param_groups.head(3).items(), 1):
        summary.append({
            'Parameter': param.replace('_', ' ').title(),
            'Rank': rank,
            'Value': value,
            'Average PNL': round(avg_pnl, 2),
            'Correlation': round(correlation, 3)
        })

# Create a summary dataframe
summary_df = pd.DataFrame(summary)

# Print and save summary
print("\n===== Parameter Performance Summary =====")
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(os.path.dirname(__file__), 'parameter_performance_summary.csv'), index=False)

# Create a final recommendations section
best_params = {}
for param in params:
    best_value = data.groupby(param)['pnl'].mean().idxmax()
    best_params[param] = best_value

print("\n===== Recommended Parameter Values =====")
for param, value in best_params.items():
    print(f"{param.replace('_', ' ').title()}: {value}")

# Additional analysis: best performing parameter combinations
top_configs = data.sort_values('pnl', ascending=False).head(10)
print("\n===== Top 10 Performing Parameter Combinations =====")
print(top_configs[params + ['pnl']].to_string(index=False))

# Save the top configurations
top_configs[params + ['pnl']].to_csv(os.path.join(os.path.dirname(__file__), 'top_parameter_combinations.csv'), index=False)
