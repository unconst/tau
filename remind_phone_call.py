#!/usr/bin/env python3
"""Send a reminder message at 10:20 EST about a phone call."""

import sys
import os
import time
from datetime import datetime, timedelta

# Add tau to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tau.tools.send_message import main as send_message

def get_target_time():
    """Calculate the target time (10:20 EST) in UTC."""
    # EST is UTC-5
    EST_OFFSET_HOURS = 5
    
    now_utc = datetime.utcnow()
    
    # Target time in EST: 10:20
    target_hour_est = 10
    target_minute_est = 20
    
    # Convert to UTC: 10:20 EST = 15:20 UTC
    target_hour_utc = target_hour_est + EST_OFFSET_HOURS
    target_minute_utc = target_minute_est
    
    # Create target datetime for today
    target_utc = datetime(now_utc.year, now_utc.month, now_utc.day, 
                         target_hour_utc, target_minute_utc)
    
    # If target time has already passed today, schedule for tomorrow
    if target_utc <= now_utc:
        target_utc += timedelta(days=1)
    
    return target_utc

def main():
    target_time = get_target_time()
    now = datetime.utcnow()
    wait_seconds = (target_time - now).total_seconds()
    
    print(f"Current time (UTC): {now}")
    print(f"Target time (10:20 EST / {target_time.hour:02d}:{target_time.minute:02d} UTC): {target_time}")
    print(f"Waiting {wait_seconds:.0f} seconds ({wait_seconds/60:.1f} minutes)...")
    
    if wait_seconds > 0:
        time.sleep(wait_seconds)
    
    # Send the reminder
    message = "📞 Reminder: You have a phone call"
    print(f"Sending reminder: {message}")
    
    # Call send_message tool
    from tau.telegram import notify
    notify(message)

if __name__ == "__main__":
    main()
