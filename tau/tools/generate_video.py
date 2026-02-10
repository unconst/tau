#!/usr/bin/env python3
"""Generate video using Chutes WAN 2.2 Image-to-Video model.

Agents can call this tool to generate videos from images + prompts.

Usage:
    python -m tau.tools.generate_video --prompt "A cat walking" --image /path/to/image.jpg
    python -m tau.tools.generate_video --prompt "Ocean waves" --image "https://example.com/image.jpg"
    python -m tau.tools.generate_video --prompt "A cat walking" --image base64_string

The script calls the Chutes WAN 2.2 I2V (image-to-video) API and sends
the resulting video via Telegram.
"""

import base64
import io
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

# Default negative prompt (Chinese + English quality boosters from the Chutes example)
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def image_to_base64(image_source: str) -> str:
    """Convert an image source to a base64 string.

    Accepts:
    - A local file path
    - A URL (http/https)
    - A raw base64 string (returned as-is)
    """
    # Already base64?
    if not os.path.exists(image_source) and not image_source.startswith(("http://", "https://")):
        # Assume it's already base64
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
    guidance_scale: float = 1.0,
    negative_prompt: str | None = None,
    fast: bool = True,
    seed: int | None = None,
    timeout: int = 300,
) -> bytes:
    """Call the Chutes WAN 2.2 I2V API and return raw video bytes.

    Args:
        prompt: Text description of the desired video motion/content.
        image: Image source — file path, URL, or base64 string.
        frames: Number of frames to generate (default 81 ≈ 3.4s at 24fps).
        guidance_scale: CFG scale (default 1.0).
        negative_prompt: Things to avoid. Uses a sensible default if None.
        fast: Use fast mode (default True).
        seed: Random seed for reproducibility. None = random.
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
        "fast": fast,
        "seed": seed,
        "image": image_b64,
        "frames": frames,
        "prompt": prompt,
        "guidance_scale": guidance_scale,
        "negative_prompt": negative_prompt or DEFAULT_NEGATIVE_PROMPT,
    }

    headers = {
        "Authorization": f"Bearer {CHUTES_API_TOKEN}",
        "Content-Type": "application/json",
    }

    logger.info(f"Calling WAN I2V API: prompt='{prompt[:50]}...', frames={frames}")

    resp = requests.post(
        WAN_I2V_URL,
        headers=headers,
        json=body,
        timeout=timeout,
    )
    resp.raise_for_status()

    # The API may return:
    # 1. Raw video bytes (content-type contains "video")
    # 2. JSON with a video URL or base64 video
    content_type = resp.headers.get("content-type", "")

    if "video" in content_type or "octet-stream" in content_type:
        # Raw binary video
        return resp.content

    # Try JSON response
    try:
        data = resp.json()
    except Exception:
        # If it's not JSON and not video content-type, treat as raw bytes
        if len(resp.content) > 1000:
            return resp.content
        raise Exception(f"Unexpected response: {resp.text[:200]}")

    # Extract video from JSON — try common field names
    for key in ("video", "output", "result", "data", "url", "video_url"):
        value = data.get(key)
        if value is None:
            continue

        # If it's a URL, download the video
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            video_resp = requests.get(value, timeout=120)
            video_resp.raise_for_status()
            return video_resp.content

        # If it's base64, decode it
        if isinstance(value, str) and len(value) > 100:
            try:
                return base64.b64decode(value)
            except Exception:
                pass

    raise Exception(f"Could not extract video from API response: {list(data.keys())}")


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
