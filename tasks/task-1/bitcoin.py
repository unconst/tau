"""Bitcoin price tracking functionality."""

import os
import threading
import time
import requests
from datetime import datetime

# Calculate workspace root (go up from tasks/task-1/ to workspace root)
WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BITCOIN_FILE = os.path.join(WORKSPACE, "tasks", "task-1", "Bitcoin.md")


def fetch_bitcoin_price() -> float | None:
    """Fetch current Bitcoin price from Coinbase API."""
    try:
        response = requests.get(
            "https://api.coinbase.com/v2/exchange-rates?currency=BTC",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        usd_rate = float(data["data"]["rates"]["USD"])
        return usd_rate
    except Exception as e:
        print(f"Error fetching Bitcoin price: {e}")
        return None


def write_bitcoin_price():
    """Fetch Bitcoin price and append to Bitcoin.md."""
    price = fetch_bitcoin_price()
    if price is None:
        return False
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp} - ${price:,.2f} USD\n"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(BITCOIN_FILE), exist_ok=True)
    
    # Append to Bitcoin.md (create if doesn't exist)
    with open(BITCOIN_FILE, "a") as f:
        f.write(entry)
    
    return True


def run_hourly_scheduler(stop_event=None):
    """Run write_bitcoin_price() every hour in a loop."""
    HOUR_IN_SECONDS = 3600
    
    while True:
        if stop_event and stop_event.is_set():
            break
        
        # Write price immediately on start
        write_bitcoin_price()
        
        # Sleep for 1 hour, checking stop_event periodically
        for _ in range(60):  # Check every minute
            if stop_event and stop_event.is_set():
                return
            time.sleep(60)  # Sleep 1 minute at a time
