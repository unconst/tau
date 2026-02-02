#!/usr/bin/env python3
"""Script to send updated Bitcoin price every minute forever via Telegram."""

import os
import sys
import time
import requests
from datetime import datetime

# Add workspace root to path to import tau modules
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, workspace_root)

from tau.telegram import notify


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
        print(f"Error fetching Bitcoin price: {e}", file=sys.stderr)
        return None


def main():
    """Send Bitcoin price every minute forever."""
    print("Starting Bitcoin price monitor. Sending price updates every minute...")
    notify("Bitcoin price monitor started. You'll receive price updates every minute.")
    
    while True:
        try:
            price = fetch_bitcoin_price()
            if price is not None:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                message = f"Bitcoin price ({timestamp}): ${price:,.2f} USD"
                notify(message)
                print(f"Sent: {message}")
            else:
                error_msg = f"Failed to fetch Bitcoin price at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                print(error_msg, file=sys.stderr)
                # Don't spam Telegram with errors, just log them
            
            # Wait 60 seconds before next update
            time.sleep(60)
            
        except KeyboardInterrupt:
            print("\nStopping Bitcoin price monitor...")
            notify("Bitcoin price monitor stopped.")
            break
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            print(error_msg, file=sys.stderr)
            # Wait before retrying to avoid rapid error loops
            time.sleep(60)


if __name__ == "__main__":
    main()
