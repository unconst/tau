#!/usr/bin/env python3
"""Send a voice message via Telegram using Chutes Kokoro TTS.

Agents can call this script to send voice messages to the user on Telegram.

Usage:
    python -m tau.tools.send_voice "Your message here"
    
The script converts text to speech using the Chutes Kokoro TTS API and sends
it as a voice message via Telegram.
"""

import os
import sys
import tempfile

import requests
from tau.telegram import bot, get_chat_id, append_chat_history

CHUTES_API_TOKEN = os.getenv("CHUTES_API_TOKEN")
KOKORO_URL = "https://chutes-kokoro.chutes.ai/speak"


def generate_tts(text: str, output_path: str = None, speed: float = 1.0) -> str:
    """Generate text-to-speech audio using Chutes Kokoro TTS API.

    Returns the path to a temporary WAV file.
    """
    if not CHUTES_API_TOKEN:
        raise Exception("CHUTES_API_TOKEN not set")

    if not output_path:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            output_path = tmp_file.name

    try:
        resp = requests.post(
            KOKORO_URL,
            headers={
                "Authorization": f"Bearer {CHUTES_API_TOKEN}",
                "Content-Type": "application/json",
            },
            json={"text": text, "speed": speed},
            timeout=60,
        )
        resp.raise_for_status()

        with open(output_path, 'wb') as f:
            f.write(resp.content)

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
        sys.exit(1)

    if len(sys.argv) < 2:
        print('Usage: python -m tau.tools.send_voice "your message"', file=sys.stderr)
        sys.exit(1)

    text = " ".join(sys.argv[1:])

    try:
        send_voice_message(chat_id, text)
        print(f"Voice message sent: {text}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
