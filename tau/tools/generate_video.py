#!/usr/bin/env python3
"""Generate video using Chutes WAN 2.2 Image-to-Video model.

Agents can call this tool to generate videos from images + prompts.

Usage:
    python -m tau.tools.generate_video --prompt "A cat walking" --image /path/to/image.jpg
    python -m tau.tools.generate_video --prompt "Ocean waves" --image "https://example.com/image.jpg"
    python -m tau.tools.generate_video --prompt "A cat walking" --image base64_string

The script calls the Chutes WAN 2.2 I2V (image-to-video) API and sends
the resulting video via Telegram.  The API returns raw MP4 bytes on success.
"""

import base64
import os
import sys
import tempfile
import time
import logging

import requests
from tau.telegram import bot, get_chat_id, append_chat_history

logger = logging.getLogger(__name__)

CHUTES_API_TOKEN = os.getenv("CHUTES_API_TOKEN")
WAN_I2V_URL = "https://chutes-wan-2-2-i2v-14b-fast.chutes.ai/generate"


def image_to_base64(image_source: str) -> str:
    """Convert an image source to a base64 string.

    Accepts:
    - A local file path
    - A URL (http/https)
    - A raw base64 string (returned as-is)
    """
    # Already base64?
    if not os.path.exists(image_source) and not image_source.startswith(("http://", "https://")):
        return image_source

    # URL — download first
    if image_source.startswith(("http://", "https://")):
        resp = requests.get(image_source, timeout=30)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode("utf-8")

    # Local file
    with open(image_source, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_video(
    prompt: str,
    image: str,
    *,
    frames: int = 81,
    fps: int = 16,
    resolution: str = "480p",
    fast: bool = True,
    timeout: int = 300,
) -> bytes:
    """Call the Chutes WAN 2.2 I2V API and return raw video bytes.

    Args:
        prompt: Text description of the desired video motion/content.
        image: Image source — file path, URL, or base64 string.
        frames: Number of frames to generate (default 81).
        fps: Frames per second (default 16).
        resolution: Output resolution (default "480p").
        fast: Use fast mode (default True).
        timeout: Request timeout in seconds.

    Returns:
        Raw video bytes (mp4).

    Raises:
        Exception on API errors.
    """
    if not CHUTES_API_TOKEN:
        raise Exception("CHUTES_API_TOKEN not set — video generation unavailable")

    # Convert image to base64 if needed
    image_b64 = image_to_base64(image)

    body = {
        "prompt": prompt,
        "image": image_b64,
        "frames": frames,
        "fps": fps,
        "resolution": resolution,
        "fast": fast,
    }

    headers = {
        "Authorization": f"Bearer {CHUTES_API_TOKEN}",
        "Content-Type": "application/json",
    }

    logger.info(f"Calling WAN I2V API: prompt='{prompt[:50]}...', frames={frames}, fps={fps}, res={resolution}")

    # Retry with backoff on 429 (capacity) errors
    max_retries = 4
    for attempt in range(max_retries + 1):
        resp = requests.post(
            WAN_I2V_URL,
            headers=headers,
            json=body,
            timeout=timeout,
        )
        if resp.status_code == 429 and attempt < max_retries:
            wait = 15 * (attempt + 1)  # 15s, 30s, 45s, 60s
            logger.info(f"Rate limited (429), retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
            continue
        break

    resp.raise_for_status()

    # API returns raw MP4 bytes on 200
    if len(resp.content) < 1000:
        # Suspiciously small — might be a JSON error body
        try:
            data = resp.json()
            raise Exception(f"API returned JSON instead of video: {data}")
        except (ValueError, Exception):
            pass

    return resp.content


def send_video_message(
    chat_id: int,
    prompt: str,
    image: str,
    *,
    frames: int = 81,
    reply_to_message_id: int | None = None,
) -> str:
    """Generate a video and send it to Telegram.

    Args:
        chat_id: Telegram chat to send to.
        prompt: Video generation prompt.
        image: Image source (path, URL, or base64).
        frames: Number of frames.
        reply_to_message_id: Message to reply to.

    Returns:
        Status message.
    """
    video_path = None
    try:
        video_bytes = generate_video(prompt, image, frames=frames)

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            video_path = tmp.name

        # Send via Telegram
        with open(video_path, "rb") as video_file:
            bot.send_video(
                chat_id,
                video_file,
                caption=f"🎬 {prompt[:200]}",
                reply_to_message_id=reply_to_message_id,
            )

        append_chat_history("assistant", f"[video generated]: {prompt}")
        return f"Video sent: {prompt[:100]}"

    except Exception as e:
        raise Exception(f"Failed to generate/send video: {str(e)}")
    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except Exception:
                pass


def main():
    """CLI entry point for video generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate video from image + prompt")
    parser.add_argument("--prompt", required=True, help="Video generation prompt")
    parser.add_argument("--image", required=True, help="Image path, URL, or base64")
    parser.add_argument("--frames", type=int, default=81, help="Number of frames (default 81)")
    parser.add_argument("--output", help="Save video to file instead of sending to Telegram")
    args = parser.parse_args()

    if args.output:
        # Save to file
        try:
            video_bytes = generate_video(args.prompt, args.image, frames=args.frames)
            with open(args.output, "wb") as f:
                f.write(video_bytes)
            print(f"Video saved to {args.output} ({len(video_bytes)} bytes)")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Send to Telegram
        chat_id = get_chat_id()
        if not chat_id:
            print("Error: No chat ID found. Send a message to the bot first.")
            sys.exit(1)

        try:
            result = send_video_message(chat_id, args.prompt, args.image, frames=args.frames)
            print(result)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
