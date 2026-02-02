#!/usr/bin/env python3
"""Script to ask GPT-4 'what is the capital of Texas' and send the answer via Telegram."""

import os
import sys
import subprocess

# Add workspace root to path to import tau modules
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, workspace_root)

from openai import OpenAI
from tau.telegram import notify


def main():
    """Ask GPT-4 and send answer via Telegram."""
    # Get OpenAI API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        error_msg = "Error: OPENAI_API_KEY environment variable not set"
        print(error_msg, file=sys.stderr)
        notify(error_msg)
        sys.exit(1)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    try:
        # Ask GPT-4 the question
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "what is the capital of Texas"}
            ],
            max_tokens=100
        )
        
        # Extract the answer
        answer = response.choices[0].message.content.strip()
        
        # Send answer via Telegram
        message = f"GPT-4 says: {answer}"
        notify(message)
        print(f"Question asked and answer sent: {answer}")
        
    except Exception as e:
        error_msg = f"Error calling OpenAI API: {str(e)}"
        print(error_msg, file=sys.stderr)
        notify(error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
