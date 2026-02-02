#!/usr/bin/env python3
"""Generate images using OpenAI DALL-E API and send via Telegram."""

import os
import sys
import tempfile
import urllib.request
from pathlib import Path

# Add workspace to path to import tau modules
WORKSPACE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(WORKSPACE))

from openai import OpenAI
from tau.telegram import bot, get_chat_id, append_chat_history

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY environment variable not set")

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def generate_image(prompt: str, model: str = "dall-e-3", size: str = "1024x1024", quality: str = "standard") -> str:
    """Generate an image using OpenAI DALL-E API.
    
    Args:
        prompt: Text description of the image to generate
        model: DALL-E model to use ("dall-e-2" or "dall-e-3")
        size: Image size ("256x256", "512x512", "1024x1024" for DALL-E-2, 
              "1024x1024", "1792x1024", "1024x1792" for DALL-E-3)
        quality: Quality setting for DALL-E-3 ("standard" or "hd")
    
    Returns:
        URL of the generated image
    """
    try:
        response = openai_client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality if model == "dall-e-3" else None,
            n=1
        )
        return response.data[0].url
    except Exception as e:
        raise Exception(f"Failed to generate image: {str(e)}")


def download_image(image_url: str) -> str:
    """Download an image from a URL and save to temporary file.
    
    Args:
        image_url: URL of the image to download
    
    Returns:
        Path to the downloaded image file
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            with urllib.request.urlopen(image_url, timeout=30) as response:
                tmp_file.write(response.read())
            return tmp_file.name
    except Exception as e:
        raise Exception(f"Failed to download image: {str(e)}")


def send_image_via_telegram(chat_id: int, image_path: str, caption: str = None):
    """Send an image via Telegram.
    
    Args:
        chat_id: Telegram chat ID to send the image to
        image_path: Path to the image file
        caption: Optional caption for the image
    """
    try:
        with open(image_path, 'rb') as photo:
            bot.send_photo(chat_id, photo, caption=caption)
        append_chat_history("assistant", f"[image]: {caption or 'Generated image'}")
    except Exception as e:
        raise Exception(f"Failed to send image via Telegram: {str(e)}")


def generate_and_send_image(prompt: str, chat_id: int = None, caption: str = None):
    """Generate an image and send it via Telegram.
    
    Args:
        prompt: Text description of the image to generate
        chat_id: Telegram chat ID (if None, will get from saved chat_id.txt)
        caption: Optional caption for the image
    """
    if chat_id is None:
        chat_id = get_chat_id()
        if not chat_id:
            raise Exception("No chat ID available. Please start a conversation with the bot first.")
    
    image_path = None
    try:
        # Generate image
        print(f"Generating image with prompt: {prompt}")
        image_url = generate_image(prompt)
        print(f"Image generated: {image_url}")
        
        # Download image
        print("Downloading image...")
        image_path = download_image(image_url)
        print(f"Image downloaded to: {image_path}")
        
        # Send via Telegram
        print(f"Sending image to chat {chat_id}...")
        send_image_via_telegram(chat_id, image_path, caption or prompt)
        print("Image sent successfully!")
        
    except Exception as e:
        raise Exception(f"Error generating and sending image: {str(e)}")
    finally:
        # Clean up temporary file
        if image_path and os.path.exists(image_path):
            try:
                os.unlink(image_path)
            except Exception:
                pass


def main():
    """Main function to generate and send a horse snake image."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate images using DALL-E and send via Telegram")
    parser.add_argument("prompt", nargs="?", default="a horse snake, mythical creature combining horse and snake features, detailed, artistic",
                       help="Image generation prompt (default: horse snake)")
    parser.add_argument("--caption", help="Caption for the Telegram message")
    
    args = parser.parse_args()
    
    try:
        generate_and_send_image(args.prompt, caption=args.caption)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
