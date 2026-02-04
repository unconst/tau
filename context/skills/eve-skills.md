# Eve Skills System — Integration Guide

This document describes how to access and use the Eve (Eden.art) skills/tools system programmatically.

## Overview

Eve is Eden.art's autonomous creative agent framework. It provides a large collection of tools/skills for:
- Image generation (Flux, SDXL, Stable Diffusion, etc.)
- Video generation (Runway, Kling, etc.)
- Audio generation (ElevenLabs speech/music, MMAudio, etc.)
- Social media posting (Twitter, Discord, Telegram, Farcaster, etc.)
- Media editing and utilities

Repository: https://github.com/edenartlab/eve

## Installation

```bash
# Add to dependencies
pip install git+https://github.com/edenartlab/eve.git

# Or with uv
uv pip install git+https://github.com/edenartlab/eve.git
```

## Core Concepts

### Tool Structure

Each tool in Eve is defined by:
- `key`: Unique identifier (e.g., "flux_schnell", "runway", "elevenlabs_speech")
- `name`: Display name
- `description`: What the tool does
- `tip`: Additional usage hints
- `output_type`: Type of output (image, video, audio, string, etc.)
- `parameters`: Input parameters with types, defaults, constraints
- `handler`: Execution backend (local, modal, comfyui, replicate, gcp, fal, mcp)

### Tool Categories

Tools are organized into sets:
- `ALL_TOOLS`: Core creative tools
- `SOCIAL_MEDIA_TOOLS`: Posting to platforms
- `RETRIEVAL_TOOLS`: Memory/search tools
- `GIGABRAIN_TOOLS`: Advanced reasoning tools
- `HOME_ASSISTANT_TOOLS`: Smart home control

## Programmatic Access

### Loading Tools from MongoDB

```python
from eve.tool import get_tools_from_mongo

# Get all active tools
tools = get_tools_from_mongo()

# Get specific tools
tools = get_tools_from_mongo(tools=["flux_schnell", "runway", "elevenlabs_speech"])

# Include inactive tools
tools = get_tools_from_mongo(include_inactive=True)
```

### Loading Tools from YAML Files

```python
from eve.tool import get_tools_from_api_files, Tool

# Get all tools from api.yaml files
tools = get_tools_from_api_files()

# Load a specific tool from file
tool = Tool.from_yaml("path/to/api.yaml")
```

### Getting Tool Schemas (for LLM integration)

```python
# Get OpenAI function calling schema
openai_schema = tool.openai_schema(exclude_hidden=True)

# Get Anthropic schema
anthropic_schema = tool.anthropic_schema(exclude_hidden=True)

# Get Gemini schema
gemini_schema = tool.gemini_schema(exclude_hidden=True)
```

### Running Tools

```python
import asyncio
from eve.tool import Tool

# Load tool
tool = Tool.load("flux_schnell")

# Run synchronously
result = tool.run({
    "prompt": "a beautiful sunset over mountains",
    "aspect_ratio": "16:9"
})

# Run asynchronously
async def generate():
    result = await tool.async_run({
        "prompt": "a beautiful sunset over mountains",
        "aspect_ratio": "16:9"
    })
    return result
```

### Creating Custom Tools

```python
from pydantic import BaseModel, Field
from eve.tool import Tool

class MyToolInput(BaseModel):
    """Description of my custom tool."""
    text: str = Field(..., description="Input text")
    count: int = Field(default=1, description="Number of outputs")

# Create tool from Pydantic model
my_tool = Tool.from_pydantic(
    model=MyToolInput,
    key="my_custom_tool",
    name="My Custom Tool",
    description="Does something custom",
    output_type="string",
    handler="local"
)

# Register with a handler function
from eve.tool import Tool
from eve.tools.tool_handlers import handlers

async def my_handler(context):
    args = context.args
    return {"output": f"Processed: {args['text']}"}

Tool.register_new(MyToolInput, my_handler)
```

## Available Tools (Key Categories)

### Image Generation
- `flux_schnell`: Fast Flux image generation
- `flux_dev`: High-quality Flux generation
- `flux_dev_lora`: Flux with LoRA support
- `flux_kontext`: Context-aware Flux
- `txt2img`: Text to image (SDXL)
- `openai_image_generate`: DALL-E image generation
- `openai_image_edit`: DALL-E image editing
- `seedream3`, `seedream4`, `seedream45`: Seedream models

### Video Generation
- `runway`, `runway2`, `runway3`: Runway video models
- `kling`, `kling_pro`, `kling_v25`: Kling video models
- `hedra`: Hedra video generation
- `seedance1`: Seedance video
- `vid2vid_sdxl`: Video to video
- `video_FX`: Video effects
- `texture_flow`: Texture animation

### Audio Generation
- `elevenlabs_speech`: Text to speech
- `elevenlabs_music`: Music generation
- `elevenlabs_fx`: Sound effects
- `elevenlabs_dialogue`: Multi-voice dialogue
- `vibevoice`: Voice cloning
- `transcription`: Audio transcription

### Media Editing
- `media_editor`: General media editing
- `outpaint`: Image outpainting
- `style_transfer`: Style transfer

### Social Media
- `tweet`, `twitter_search`, `twitter_mentions`: Twitter
- `discord_post`, `discord_search`: Discord
- `telegram_post`: Telegram
- `farcaster_cast`, `farcaster_search`: Farcaster
- `instagram_post`: Instagram
- `tiktok_post`: TikTok

### Utilities
- `weather`: Weather information
- `websearch`: Web search
- `google_calendar_*`: Calendar operations
- `email_send`, `gmail_send`: Email

## Tool Discovery

To discover available tools programmatically:

```python
from eve.tool import get_tools_from_mongo
from eve.tool_constants import ALL_TOOLS, TOOL_SETS

# List all default tools
print(ALL_TOOLS)

# List tool sets
for set_name, tools in TOOL_SETS.items():
    print(f"{set_name}: {tools}")

# Get tool details
tools = get_tools_from_mongo()
for key, tool in tools.items():
    print(f"{key}: {tool.description}")
    print(f"  Output: {tool.output_type}")
    print(f"  Handler: {tool.handler}")
```

## Environment Requirements

Eve requires various API keys depending on which tools you use:
- `OPENAI_API_KEY`: For OpenAI tools
- `ANTHROPIC_API_KEY`: For Claude
- `ELEVENLABS_API_KEY`: For ElevenLabs
- `FAL_KEY`: For Fal.ai tools
- `REPLICATE_API_TOKEN`: For Replicate tools
- MongoDB connection for tool/agent storage

## Integration with Tau

Tau can use Eve skills by:
1. Installing eve as a dependency
2. Loading tools with `get_tools_from_mongo()` or `Tool.load(key)`
3. Getting LLM-compatible schemas with `tool.openai_schema()`
4. Running tools with `tool.run(args)` or `await tool.async_run(args)`

The skills expand Tau's creative capabilities significantly, enabling:
- Image/video/audio generation
- Social media automation
- Media editing and processing
