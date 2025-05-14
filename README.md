# Bollinger Band Trading Strategy

This project implements a configurable Bollinger Band trading strategy using historical market data. It includes:

- Signal generation using Bollinger Bands  
- Position sizing and trade execution  
- Return computation with leverage and transaction costs  
- Visualizations for analysis and debugging  

The code uses NumPy and pandas and is fully vectorized. It can test strategies on large datasets quickly once the initial data is fetched.

## Quick Start

### Installation

Install the required dependencies:

```bash
pip install pandas numpy matplotlib tpqoa
```

> Note: `tpqoa` is only needed if using OANDA data directly.

## File Structure

```
.
├── BollingerStrategy.py          # Core logic: indicators, trade signals, returns
├── Visualisation.py              # All plotting functions
├── Examples.py                   # Strategy examples
├── data/
│   └── EUR_GBP_H1_2020_2024.csv  # Sample dataset (optional)
├── README.md                     # Project overview (this file)
```

## Strategy Overview

The strategy uses Bollinger Bands to identify price extremes and executes trades based on configurable parameters:

```python
settings = [(20, 3, 1.5)]  # (SMA_window_size, entry_std_dev, exit_std_dev)
```

- Can operate in mean-reversion or trend-following mode  
- Supports transaction cost and leverage customization  
- Optional time filtering to restrict trading hours

## Usage Example

### With OANDA data

Download the repository and add a raw text file named `oanda.cfg` with your OANDA API credentials.  
**WARNING:** Never share API credentials publicly.

```python
from strategy import get_data, test_strategy

data = get_data("EUR_USD", "2020-01-01", "2020-03-01", "H1")
settings = [(20, 3, 1.5)]
test_strategy(data, settings, transaction_cost=0.7, leverage=1.0)
```

### With pre-saved CSV data

```python
import pandas as pd
from strategy import test_strategy

data = pd.read_csv("data/EUR_GBP_H1_2020_2024.csv", index_col=0, parse_dates=True)
settings = [(20, 3, 1.5)]
test_strategy(data, settings)
```

## Visual Outputs

The following visualizations are automatically generated:

- Price chart with Bollinger Bands and entry/exit signals  
- Cumulative returns (strategy vs. market)  
- Histogram of trade returns  
- Z-score trends for selected indicators  
- Best and worst trades with contextual plots

## Examples

Run `examples.py` to see a working example of the code.

## Disclaimer

This project is for educational purposes only. It demonstrates how to build and evaluate a basic trading strategy using Bollinger Bands. The strategy is not optimized for live trading and is unlikely to be profitable in real-world market conditions without significant refinement. 

## License

This project is licensed under the MIT License.  
You are free to use, modify, and distribute it for personal or commercial purposes.
