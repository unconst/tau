#!/usr/bin/env python3
"""Search for available skills/tools in the Eve (Eden.art) ecosystem.

Agents can call this script to discover creative tools for image, video, audio
generation and other capabilities.

Usage:
    python -m tau.tools.search_skills                    # List all available skills
    python -m tau.tools.search_skills "image"            # Search for image-related skills
    python -m tau.tools.search_skills "video generation" # Search for video tools
    python -m tau.tools.search_skills --category audio   # Filter by category
    python -m tau.tools.search_skills --details flux     # Get detailed info about a skill
"""

import sys
import json
from typing import Optional

# Skill catalog - sourced from Eve (edenartlab/eve) tool constants
# This is a static snapshot for offline access; can be updated from Eve repo
SKILLS_CATALOG = {
    # Image Generation
    "flux_schnell": {
        "name": "Flux Schnell",
        "category": "image",
        "description": "Fast Flux image generation model for quick iterations",
        "output_type": "image",
        "parameters": ["prompt", "aspect_ratio", "lora"],
    },
    "flux_dev": {
        "name": "Flux Dev",
        "category": "image", 
        "description": "High-quality Flux image generation with better detail",
        "output_type": "image",
        "parameters": ["prompt", "aspect_ratio", "lora", "guidance_scale"],
    },
    "flux_dev_lora": {
        "name": "Flux Dev LoRA",
        "category": "image",
        "description": "Flux Dev with LoRA model support for custom styles",
        "output_type": "image",
        "parameters": ["prompt", "lora", "lora_strength"],
    },
    "flux_kontext": {
        "name": "Flux Kontext",
        "category": "image",
        "description": "Context-aware Flux generation for coherent image series",
        "output_type": "image",
        "parameters": ["prompt", "context_images"],
    },
    "txt2img": {
        "name": "Text to Image (SDXL)",
        "category": "image",
        "description": "Stable Diffusion XL text-to-image generation",
        "output_type": "image",
        "parameters": ["prompt", "negative_prompt", "width", "height", "steps"],
    },
    "openai_image_generate": {
        "name": "DALL-E Image Generation",
        "category": "image",
        "description": "OpenAI DALL-E image generation",
        "output_type": "image",
        "parameters": ["prompt", "size", "quality"],
    },
    "openai_image_edit": {
        "name": "DALL-E Image Edit",
        "category": "image",
        "description": "OpenAI DALL-E image editing with masks",
        "output_type": "image",
        "parameters": ["image", "mask", "prompt"],
    },
    "seedream3": {
        "name": "Seedream 3",
        "category": "image",
        "description": "Seedream model version 3 for creative image generation",
        "output_type": "image",
        "parameters": ["prompt"],
    },
    "seedream45": {
        "name": "Seedream 4.5",
        "category": "image",
        "description": "Latest Seedream model with improved quality",
        "output_type": "image",
        "parameters": ["prompt"],
    },
    "outpaint": {
        "name": "Outpaint",
        "category": "image",
        "description": "Expand images beyond their original boundaries",
        "output_type": "image",
        "parameters": ["image", "direction", "prompt"],
    },
    "style_transfer": {
        "name": "Style Transfer",
        "category": "image",
        "description": "Apply artistic styles from one image to another",
        "output_type": "image",
        "parameters": ["content_image", "style_image"],
    },
    
    # Video Generation
    "runway": {
        "name": "Runway Gen-1",
        "category": "video",
        "description": "Runway video generation model",
        "output_type": "video",
        "parameters": ["prompt", "image"],
    },
    "runway2": {
        "name": "Runway Gen-2",
        "category": "video",
        "description": "Runway Gen-2 improved video generation",
        "output_type": "video",
        "parameters": ["prompt", "image", "motion_strength"],
    },
    "runway3": {
        "name": "Runway Gen-3",
        "category": "video",
        "description": "Latest Runway video generation with best quality",
        "output_type": "video",
        "parameters": ["prompt", "image"],
    },
    "kling": {
        "name": "Kling",
        "category": "video",
        "description": "Kling video generation model",
        "output_type": "video",
        "parameters": ["prompt", "image"],
    },
    "kling_pro": {
        "name": "Kling Pro",
        "category": "video",
        "description": "Kling Pro with extended duration and quality",
        "output_type": "video",
        "parameters": ["prompt", "image", "duration"],
    },
    "kling_v25": {
        "name": "Kling v2.5",
        "category": "video",
        "description": "Latest Kling model version 2.5",
        "output_type": "video",
        "parameters": ["prompt", "image"],
    },
    "hedra": {
        "name": "Hedra",
        "category": "video",
        "description": "Hedra video/avatar generation",
        "output_type": "video",
        "parameters": ["prompt", "audio"],
    },
    "seedance1": {
        "name": "Seedance",
        "category": "video",
        "description": "Seedance video generation for dance/motion",
        "output_type": "video",
        "parameters": ["prompt", "image"],
    },
    "vid2vid_sdxl": {
        "name": "Video to Video (SDXL)",
        "category": "video",
        "description": "Transform existing videos with SDXL styling",
        "output_type": "video",
        "parameters": ["video", "prompt", "strength"],
    },
    "video_FX": {
        "name": "Video FX",
        "category": "video",
        "description": "Apply effects and filters to videos",
        "output_type": "video",
        "parameters": ["video", "effect"],
    },
    "texture_flow": {
        "name": "Texture Flow",
        "category": "video",
        "description": "Animated texture generation for VJ/live visuals",
        "output_type": "video",
        "parameters": ["prompt", "motion_type"],
    },
    
    # Audio Generation
    "elevenlabs_speech": {
        "name": "ElevenLabs Speech",
        "category": "audio",
        "description": "High-quality text-to-speech with ElevenLabs",
        "output_type": "audio",
        "parameters": ["text", "voice", "model"],
    },
    "elevenlabs_music": {
        "name": "ElevenLabs Music",
        "category": "audio",
        "description": "AI music generation with ElevenLabs",
        "output_type": "audio",
        "parameters": ["prompt", "duration"],
    },
    "elevenlabs_fx": {
        "name": "ElevenLabs Sound FX",
        "category": "audio",
        "description": "Sound effects generation with ElevenLabs",
        "output_type": "audio",
        "parameters": ["prompt", "duration"],
    },
    "elevenlabs_dialogue": {
        "name": "ElevenLabs Dialogue",
        "category": "audio",
        "description": "Multi-voice dialogue generation",
        "output_type": "audio",
        "parameters": ["segments", "voices"],
    },
    "vibevoice": {
        "name": "VibeVoice",
        "category": "audio",
        "description": "Voice cloning and synthesis",
        "output_type": "audio",
        "parameters": ["text", "voice_sample"],
    },
    "transcription": {
        "name": "Audio Transcription",
        "category": "audio",
        "description": "Transcribe audio to text",
        "output_type": "string",
        "parameters": ["audio"],
    },
    
    # Media Editing
    "media_editor": {
        "name": "Media Editor",
        "category": "editing",
        "description": "General-purpose media editing tool",
        "output_type": "image",
        "parameters": ["media", "operation", "parameters"],
    },
    
    # Social Media
    "tweet": {
        "name": "Tweet",
        "category": "social",
        "description": "Post a tweet to Twitter/X",
        "output_type": "string",
        "parameters": ["text", "media"],
    },
    "twitter_search": {
        "name": "Twitter Search",
        "category": "social",
        "description": "Search tweets on Twitter/X",
        "output_type": "array",
        "parameters": ["query", "count"],
    },
    "twitter_mentions": {
        "name": "Twitter Mentions",
        "category": "social",
        "description": "Get mentions on Twitter/X",
        "output_type": "array",
        "parameters": ["count"],
    },
    "discord_post": {
        "name": "Discord Post",
        "category": "social",
        "description": "Post a message to Discord",
        "output_type": "string",
        "parameters": ["channel_id", "message", "media"],
    },
    "discord_search": {
        "name": "Discord Search",
        "category": "social",
        "description": "Search messages in Discord",
        "output_type": "array",
        "parameters": ["query", "channel_id"],
    },
    "telegram_post": {
        "name": "Telegram Post",
        "category": "social",
        "description": "Post a message to Telegram",
        "output_type": "string",
        "parameters": ["channel_id", "message", "media"],
    },
    "farcaster_cast": {
        "name": "Farcaster Cast",
        "category": "social",
        "description": "Post a cast to Farcaster",
        "output_type": "string",
        "parameters": ["text", "media"],
    },
    "farcaster_search": {
        "name": "Farcaster Search",
        "category": "social",
        "description": "Search casts on Farcaster",
        "output_type": "array",
        "parameters": ["query"],
    },
    "instagram_post": {
        "name": "Instagram Post",
        "category": "social",
        "description": "Post to Instagram",
        "output_type": "string",
        "parameters": ["media", "caption"],
    },
    "tiktok_post": {
        "name": "TikTok Post",
        "category": "social",
        "description": "Post a video to TikTok",
        "output_type": "string",
        "parameters": ["video", "caption"],
    },
    
    # Utilities
    "weather": {
        "name": "Weather",
        "category": "utility",
        "description": "Get current weather information",
        "output_type": "string",
        "parameters": ["location"],
    },
    "websearch": {
        "name": "Web Search",
        "category": "utility",
        "description": "Search the web for information",
        "output_type": "array",
        "parameters": ["query"],
    },
    "google_calendar_query": {
        "name": "Google Calendar Query",
        "category": "utility",
        "description": "Query Google Calendar events",
        "output_type": "array",
        "parameters": ["start_date", "end_date"],
    },
    "google_calendar_edit": {
        "name": "Google Calendar Edit",
        "category": "utility",
        "description": "Create or edit calendar events",
        "output_type": "string",
        "parameters": ["title", "start", "end", "description"],
    },
    "email_send": {
        "name": "Email Send",
        "category": "utility",
        "description": "Send an email",
        "output_type": "string",
        "parameters": ["to", "subject", "body"],
    },
}

CATEGORIES = {
    "image": "Image Generation",
    "video": "Video Generation", 
    "audio": "Audio Generation",
    "editing": "Media Editing",
    "social": "Social Media",
    "utility": "Utilities",
}


def search_skills(query: Optional[str] = None, category: Optional[str] = None) -> list:
    """Search skills by query string or category."""
    results = []
    
    for key, skill in SKILLS_CATALOG.items():
        # Filter by category if specified
        if category and skill["category"] != category:
            continue
        
        # Filter by query if specified
        if query:
            query_lower = query.lower()
            searchable = f"{key} {skill['name']} {skill['description']} {skill['category']}".lower()
            if query_lower not in searchable:
                continue
        
        results.append({
            "key": key,
            **skill
        })
    
    return results


def get_skill_details(key: str) -> Optional[dict]:
    """Get detailed information about a specific skill."""
    if key in SKILLS_CATALOG:
        return {"key": key, **SKILLS_CATALOG[key]}
    
    # Try partial match
    for skill_key, skill in SKILLS_CATALOG.items():
        if key.lower() in skill_key.lower():
            return {"key": skill_key, **skill}
    
    return None


def format_skill_list(skills: list, verbose: bool = False) -> str:
    """Format skill list for display."""
    if not skills:
        return "No skills found matching your criteria."
    
    lines = [f"Found {len(skills)} skill(s):\n"]
    
    # Group by category
    by_category = {}
    for skill in skills:
        cat = skill["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(skill)
    
    for cat, cat_skills in sorted(by_category.items()):
        cat_name = CATEGORIES.get(cat, cat.title())
        lines.append(f"\n## {cat_name}")
        
        for skill in sorted(cat_skills, key=lambda x: x["key"]):
            if verbose:
                lines.append(f"\n### {skill['key']}")
                lines.append(f"Name: {skill['name']}")
                lines.append(f"Description: {skill['description']}")
                lines.append(f"Output: {skill['output_type']}")
                lines.append(f"Parameters: {', '.join(skill['parameters'])}")
            else:
                lines.append(f"- {skill['key']}: {skill['description']}")
    
    return "\n".join(lines)


def format_skill_details(skill: dict) -> str:
    """Format detailed skill information."""
    lines = [
        f"# {skill['name']} ({skill['key']})",
        f"",
        f"**Category:** {CATEGORIES.get(skill['category'], skill['category'])}",
        f"**Output Type:** {skill['output_type']}",
        f"",
        f"## Description",
        skill['description'],
        f"",
        f"## Parameters",
    ]
    
    for param in skill['parameters']:
        lines.append(f"- `{param}`")
    
    lines.extend([
        f"",
        f"## Usage",
        f"```python",
        f"from eve.tool import Tool",
        f"",
        f"tool = Tool.load('{skill['key']}')",
        f"result = tool.run({{",
    ])
    
    for param in skill['parameters'][:2]:  # Show first 2 params as example
        lines.append(f'    "{param}": "...",')
    
    lines.extend([
        f"}})",
        f"```",
    ])
    
    return "\n".join(lines)


def main():
    """Main entry point for skill search."""
    args = sys.argv[1:]
    
    # Parse arguments
    query = None
    category = None
    details_key = None
    verbose = False
    
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--category" and i + 1 < len(args):
            category = args[i + 1]
            i += 2
        elif arg == "--details" and i + 1 < len(args):
            details_key = args[i + 1]
            i += 2
        elif arg in ("--verbose", "-v"):
            verbose = True
            i += 1
        elif arg in ("--help", "-h"):
            print(__doc__)
            sys.exit(0)
        elif not arg.startswith("-"):
            query = arg if not query else f"{query} {arg}"
            i += 1
        else:
            i += 1
    
    # Get details for a specific skill
    if details_key:
        skill = get_skill_details(details_key)
        if skill:
            print(format_skill_details(skill))
        else:
            print(f"Skill '{details_key}' not found.")
            sys.exit(1)
        return
    
    # Search skills
    skills = search_skills(query, category)
    print(format_skill_list(skills, verbose))
    
    # Also output JSON to stderr for programmatic use
    if skills:
        print(f"\n---\nJSON output:", file=sys.stderr)
        print(json.dumps(skills, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
