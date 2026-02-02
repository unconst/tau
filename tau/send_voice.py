#!/usr/bin/env python3
"""Script to send a voice message via Telegram using OpenAI TTS."""

import os
import sys
import tempfile

# Add parent directory to path to import tau
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from tau.telegram import bot, get_chat_id, append_chat_history

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY not set")
    sys.exit(1)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def generate_tts(text: str, output_path: str = None) -> str:
    """Generate text-to-speech audio using OpenAI TTS API."""
    if not output_path:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            output_path = tmp_file.name
    
    try:
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
        
        return output_path
    except Exception as e:
        raise Exception(f"Failed to generate TTS: {str(e)}")


def send_voice_message(chat_id: int, text: str):
    """Generate TTS audio and send it as a voice message via Telegram."""
    voice_path = None
    try:
        voice_path = generate_tts(text)
        
        with open(voice_path, 'rb') as voice_file:
            bot.send_voice(chat_id, voice_file)
        
        append_chat_history("assistant", f"[voice message]: {text}")
    except Exception as e:
        raise Exception(f"Failed to send voice message: {str(e)}")
    finally:
        if voice_path and os.path.exists(voice_path):
            try:
                os.unlink(voice_path)
            except Exception:
                pass

def main():
    """Send a voice message."""
    chat_id = get_chat_id()
    if not chat_id:
        print("Error: No chat ID found. Please send a message to the bot first.")
        return
    
    # Default message if none provided
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello! This is a voice message generated using OpenAI's text-to-speech API."
    
    try:
        send_voice_message(chat_id, text)
        print(f"✅ Voice message sent: {text}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
