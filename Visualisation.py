import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_price_with_bands_and_signals(
    data: pd.DataFrame,
    title: str = "Price with Bollinger Bands and Signals"
) -> None:
    """
    Plots price, Bollinger bands, and trade entry/exit signals.
    
    Args:
        data (pd.DataFrame): The DataFrame containing price and Bollinger band data.
        title (str): The title for the plot.
    
    Returns:
        None
    """
    upper_band = data["upper_entry_threshold"]
    lower_band = data["lower_entry_threshold"]
    
    plt.figure(figsize=(10, 6))
    plt.plot(data["close"], label="Price", color="blue", alpha=1)
    plt.plot(upper_band, label="Upper Band", color="green", alpha=1, lw=0.5)
    plt.plot(lower_band, label="Lower Band", color="red", alpha=1, lw=0.5)
    plt.fill_between(data.index, lower_band, upper_band, color="blue", alpha=0.1)

    # Plot trades
    longs = data[data["position"] == 1]
    shorts = data[data["position"] == -1]

    minn = lower_band.min()
    maxx = upper_band.max()
    y_coord = minn * (1 - 0.1 * (maxx - minn))
    
    plt.scatter(longs.index, [y_coord] * len(longs["close"]), marker="^", color="green", label="Long", alpha=0.8, s=1)
    plt.scatter(shorts.index, [y_coord] * len(shorts["close"]), marker="v", color="red", label="Short", alpha=0.8, s=1)

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cumulative_returns(data: pd.DataFrame) -> None:
    """
    Plots cumulative returns of the strategy vs. the market.
    
    Args:
        data (pd.DataFrame): The DataFrame containing cumulative returns.
    
    Returns:
        None
    """
    plt.figure(figsize=(14, 4))
    plt.title("Cumulative Returns: Strategy vs Market")
    plt.plot(data["cumulative_price"], label="Market", color="blue", alpha=0.6)
    plt.plot(data["cumulative_profit"], label="Strategy", color="purple")
    
    # Plot trades
    longs = data[data["position"] == 1]
    shorts = data[data["position"] == -1]

    minn = min(data["cumulative_price"].min(), data["cumulative_profit"].min())
    maxx = max(data["cumulative_price"].max(), data["cumulative_profit"].max())
    y_coord = minn * (1 - 0.1 * (maxx - minn))
    
    plt.scatter(longs.index, [y_coord] * len(longs["close"]), marker="^", color="green", label="Long", alpha=0.8, s=1)
    plt.scatter(shorts.index, [y_coord] * len(shorts["close"]), marker="v", color="red", label="Short", alpha=0.8, s=1)
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_return_histogram(data: pd.DataFrame) -> None:
    """
    Plots histogram of individual trade returns (strategy_return).
    
    Args:
        data (pd.DataFrame): The DataFrame containing trade profit data.
    
    Returns:
        None
    """
    filtered_data = (np.exp(data["realised_profit"]) - 1)[data["realised_profit"] != 0] * 100
    plt.hist(filtered_data, bins=100, edgecolor="black", lw=0.4)
    plt.xlabel("Return (%)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Returns")
    mean_value = filtered_data.mean()
    plt.axvline(mean_value, color='red', linestyle='--', linewidth=1, label=f"Mean = {round(mean_value, 3)}")
    plt.axvline(0, color='black', linestyle='-', linewidth=1)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_zscore(data: pd.DataFrame, zscore_cols: list, stds: list) -> None:
    """
    Plots Z-score over time for one or more indicators.
    
    Args:
        data (pd.DataFrame): The DataFrame containing Z-score data.
        zscore_cols (list): List of column names to plot.
        stds (list): List of standard deviation values to plot.
    
    Returns:
        None
    """
    plt.figure(figsize=(15, 4))
    for col in zscore_cols:
        plt.plot(data[col], label=col, lw=0.8)
    plt.axhline(0, color="black", linewidth=1.2)
    for std in stds:
        plt.axhline(std, color="red", linestyle="--", linewidth=1.2)
        plt.axhline(-std, color="green", linestyle="--", linewidth=1.2)
    plt.title("Z-Score")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_trade_context(data: pd.DataFrame, trade_index: pd.Timestamp, title: str = None) -> None:
    """
    Plots a trade in context (e.g., best or worst trade) with Bollinger bands and signal lines.
    
    Args:
        data (pd.DataFrame): The DataFrame containing price and signal data.
        trade_index (pd.Timestamp): The index of the trade to focus on.
        title (str, optional): Title for the plot. Defaults to None.
    
    Returns:
        None
    """
    start_index = trade_index - pd.Timedelta(days=3)
    end_index = trade_index + pd.Timedelta(days=3)
    sliced_df = data.loc[start_index:end_index]
    
    trade_times_bullish = sliced_df[(sliced_df['trade'] > 0) & (sliced_df["position"] > 0)].index
    trade_times_bearish = sliced_df[(sliced_df['trade'] < 0) & (sliced_df["position"] < 0)].index
    trade_times_neutral = sliced_df[(sliced_df['trade'] != 0) & (sliced_df["position"] == 0)].index

    sliced_df.plot(
        y=["close", "upper_entry_threshold", "lower_entry_threshold", "upper_exit_threshold", "lower_exit_threshold"],
        color=["blue", "green", "red", "purple", "orange"],
        use_index=True
    )

    plt.fill_between(
        sliced_df.index, sliced_df["upper_exit_threshold"], sliced_df["upper_entry_threshold"],
        color="blue", alpha=0.1
    )
    plt.fill_between(
        sliced_df.index, sliced_df["lower_entry_threshold"], sliced_df["upper_exit_threshold"],
        color="blue", alpha=0.1
    )

    for time in trade_times_bullish:
        if time != trade_index:
            plt.axvline(time, color='green', linestyle='--', linewidth=0.8)
    for time in trade_times_bearish:
        if time != trade_index:
            plt.axvline(time, color='red', linestyle='--', linewidth=0.8)
    for time in trade_times_neutral:
        plt.axvline(time, color='black', linestyle='--', linewidth=0.8)

    if trade_index in trade_times_bullish:
        plt.axvline(trade_index, color='green', linestyle='--', linewidth=2)

    if trade_index in trade_times_bearish:
        plt.axvline(trade_index, color='red', linestyle='--', linewidth=2)
       
    plt.title(title)
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def plot_best_and_worst_trades(data: pd.DataFrame) -> None:
    """
    Plot the best and worst trades based on cumulative profit of a strategy.
    Assumes 'position' changes at entry/exit and 'profit' is calculated.
    The best trade is displayed in bold.
    
    Args:
        data (pd.DataFrame): The DataFrame containing strategy data (including positions and profits).
    
    Returns:
        None
    """
    trades = []
    in_trade = False
    trade_start = None

    # Identify trades
    for i in range(1, len(data)):
        prev_pos = data.iloc[i - 1]["position"]
        curr_pos = data.iloc[i]["position"]
        
        # Entry
        if not in_trade and curr_pos != 0:
            in_trade = True
            trade_start = i

        # Exit
        elif in_trade and curr_pos == 0:
            trade_end = i
            cumulative_return = data.iloc[trade_start:trade_end]["profit"].sum()
            trades.append({
                "start": trade_start,
                "end": trade_end,
                "return": cumulative_return
            })
            in_trade = False

    if not trades:
        print("No trades found.")
        return

    # Sort by return
    sorted_trades = sorted(trades, key=lambda x: x["return"])
    worst = sorted_trades[0]
    best = sorted_trades[-1]
    plot_trade_context(data, data.index[best["start"]], title="Best Trade")
    plot_trade_context(data, data.index[worst["start"]], title="Worst Trade")
