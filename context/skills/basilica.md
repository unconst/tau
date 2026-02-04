# Basilica SDK — Integration Guide

This document describes how to use Basilica for on-demand GPU compute and LLM inference.

## Overview

Basilica provides on-demand compute for AI. Deploy containerized applications, serve LLMs, or access raw GPU nodes. Built on the Bittensor network with per-minute billing.

Docs: https://docs.basilica.ai
GitHub: https://github.com/one-covenant/basilica

## Installation

```bash
# Install with pip
pip install basilica-sdk

# Or with uv (recommended)
uv pip install basilica-sdk
```

## Authentication

Set your API key as an environment variable:

```bash
export BASILICA_API_TOKEN="your_api_key_here"
```

Or pass it directly to the client:

```python
from basilica import BasilicaClient

client = BasilicaClient(api_key="your_api_key_here")
```

Authentication priority:
1. `api_key` parameter passed to `BasilicaClient()`
2. `BASILICA_API_TOKEN` environment variable
3. Stored credentials from `basilica login` CLI

## Client Initialization

```python
from basilica import BasilicaClient

# Auto-detect from BASILICA_API_TOKEN
client = BasilicaClient()

# Explicit configuration
client = BasilicaClient(
    base_url="https://api.basilica.ai",
    api_key="basilica_..."
)
```

## Core Features

### Deploy Applications

Two ways to deploy:

#### 1. Using the Decorator

```python
import basilica

@basilica.deployment(
    name="hello-world",
    port=8000,
    pip_packages=["fastapi", "uvicorn"],
)
def serve():
    from fastapi import FastAPI
    import uvicorn

    app = FastAPI()

    @app.get("/")
    def index():
        return {"message": "Hello from Basilica!"}

    uvicorn.run(app, host="0.0.0.0", port=8000)

deployment = serve()
print(f"URL: {deployment.url}")
```

#### 2. Using client.deploy()

```python
from basilica import BasilicaClient

client = BasilicaClient()

# From file
deployment = client.deploy(
    name="my-app",
    source="app.py",
    port=8000,
    pip_packages=["fastapi", "uvicorn"],
)

# From inline code
deployment = client.deploy(
    name="hello",
    source="""
from http.server import HTTPServer, BaseHTTPRequestHandler

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Hello World!')

HTTPServer(('', 8000), Handler).serve_forever()
""",
    port=8000,
)

# From image only
deployment = client.deploy(
    name="nginx",
    image="nginx:alpine",
    port=80
)
```

### Deployment Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Deployment name (DNS-safe) |
| `source` | `str` | None | File path or inline code |
| `image` | `str` | `"python:3.11-slim"` | Container image |
| `port` | `int` | `8000` | Port your app listens on |
| `cpu` | `str` | `"500m"` | CPU allocation |
| `memory` | `str` | `"512Mi"` | Memory allocation |
| `env` | `Dict[str, str]` | `None` | Environment variables |
| `pip_packages` | `List[str]` | `None` | Pip dependencies |
| `replicas` | `int` | `1` | Number of instances |
| `ttl_seconds` | `int` | `None` | Auto-delete timer |
| `timeout` | `int` | `300` | Deploy timeout (seconds) |
| `storage` | `bool` or `str` | `None` | Enable storage at /data or custom path |
| `gpu_count` | `int` | `None` | Number of GPUs |
| `gpu_models` | `List[str]` | `None` | Acceptable GPU models |

### Deployment Object

```python
deployment = client.deploy(name="my-app", source="app.py")

# Properties
deployment.url        # Public URL
deployment.name       # Deployment name
deployment.state      # Current state
deployment.created_at # Creation timestamp

# Methods
deployment.status()   # Get detailed status
deployment.logs()     # Get container logs
deployment.logs(tail=50)  # Last 50 lines
deployment.delete()   # Delete the deployment
deployment.refresh()  # Refresh cached state
```

### Management

```python
# Get deployment by name
deployment = client.get("my-app")

# List all deployments
for d in client.list():
    print(f"{d.name}: {d.state}")

# Delete
deployment.delete()
# Or: client.delete("my-app")
```

## GPU Deployments

Access H100s, A100s, and more:

```python
deployment = client.deploy(
    name="gpu-app",
    source="app.py",
    image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    gpu_count=1,
    memory="8Gi",
)
```

### GPU Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `gpu_count` | `int` | Number of GPUs (1-8) |
| `gpu_models` | `List[str]` | Acceptable GPU models (e.g., `["H100", "A100"]`) |
| `min_cuda_version` | `str` | Minimum CUDA version |
| `min_gpu_memory_gb` | `int` | Minimum GPU VRAM in GB |

### Multi-GPU Example

```python
deployment = client.deploy(
    name="multi-gpu",
    source="app.py",
    image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    gpu_count=4,
    memory="32Gi",
)
```

## LLM Inference

Deploy LLMs with OpenAI-compatible APIs using vLLM or SGLang.

### vLLM (Production Workloads)

```python
from basilica import BasilicaClient

client = BasilicaClient()

deployment = client.deploy_vllm(
    name="qwen-api",
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu_count=1,
)

print(f"OpenAI-compatible API: {deployment.url}/v1")
```

### SGLang (Structured Generation)

```python
deployment = client.deploy_sglang(
    name="qwen-sglang",
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu_count=1,
)
```

### Using the LLM API

```python
from openai import OpenAI

client = OpenAI(
    base_url=f"{deployment.url}/v1",
    api_key="not-needed",  # Basilica handles auth
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### vLLM/SGLang Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Deployment name |
| `model` | `str` | required | Hugging Face model ID |
| `gpu_count` | `int` | `1` | Number of GPUs |
| `tensor_parallel_size` | `int` | `gpu_count` | Tensor parallelism |
| `max_model_len` / `context_length` | `int` | model default | Max context length |
| `quantization` | `str` | `None` | `"awq"`, `"gptq"`, etc. |
| `env` | `Dict[str, str]` | `None` | Environment variables (e.g., `HF_TOKEN`) |
| `timeout` | `int` | `600` | Deployment timeout |

### Gated Models (Llama, etc.)

```python
deployment = client.deploy_vllm(
    name="llama-api",
    model="meta-llama/Llama-3.1-8B-Instruct",
    env={"HF_TOKEN": "hf_..."},  # Hugging Face token
)
```

## Persistent Storage

Data survives container restarts, backed by Cloudflare R2.

### Enable Storage

```python
# Default mount at /data
deployment = client.deploy(
    name="app",
    source="app.py",
    storage=True,
)

# Custom mount path
deployment = client.deploy(
    name="app",
    source="app.py",
    storage="/cache",
)
```

### Named Volumes

```python
import basilica

cache = basilica.Volume.from_name("model-cache", create_if_missing=True)

@basilica.deployment(
    name="app",
    volumes={"/models": cache}
)
def serve():
    from pathlib import Path
    models_dir = Path("/models")
    # Use persistent storage...
```

### Multiple Volumes

```python
data = basilica.Volume.from_name("app-data", create_if_missing=True)
cache = basilica.Volume.from_name("model-cache", create_if_missing=True)

@basilica.deployment(
    name="ml-app",
    volumes={
        "/data": data,
        "/cache": cache,
    }
)
def serve():
    pass
```

## Error Handling

```python
from basilica import (
    BasilicaError,
    DeploymentTimeout,
    DeploymentFailed,
    DeploymentNotFound,
    ValidationError,
)

try:
    deployment = client.deploy("my-app", source="app.py")
except DeploymentTimeout as e:
    print(f"Deployment timed out after {e.timeout_seconds}s")
except DeploymentFailed as e:
    print(f"Deployment failed: {e.reason}")
except DeploymentNotFound as e:
    print(f"Deployment {e.instance_name} not found")
except ValidationError as e:
    print(f"Invalid parameter {e.field}: {e.value}")
except BasilicaError as e:
    print(f"Error: {e.message}")
```

## Common Images

| Use Case | Image |
|----------|-------|
| PyTorch | `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` |
| TensorFlow | `tensorflow/tensorflow:2.14.0-gpu` |
| vLLM | `vllm/vllm-openai:latest` |
| SGLang | `lmsysorg/sglang:latest` |
| NVIDIA Base | `nvidia/cuda:12.1-runtime-ubuntu22.04` |

## Example: FastAPI with GPU

```python
import basilica

@basilica.deployment(
    name="pytorch-api",
    image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    port=8000,
    gpu_count=1,
    memory="8Gi",
    pip_packages=["fastapi", "uvicorn"],
)
def serve():
    import torch
    from fastapi import FastAPI
    import uvicorn

    app = FastAPI()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @app.get("/")
    def info():
        return {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }

    uvicorn.run(app, host="0.0.0.0", port=8000)

deployment = serve()
print(f"API: {deployment.url}")
```

## Cleanup

Always delete deployments when done to stop billing:

```python
deployment.delete()
# Or via CLI: basilica deploy down <deployment-id>
```

## Environment Variables

```bash
# Required for authentication
export BASILICA_API_TOKEN="basilica_..."

# Optional custom endpoint
export BASILICA_API_URL="https://api.basilica.ai"
```

## Integration with Tau

Tau can use Basilica for:
1. Deploying GPU-powered inference services
2. Running LLM inference with vLLM/SGLang
3. Hosting web applications with persistent storage
4. Running batch processing jobs

Example integration:

```python
from basilica import BasilicaClient
import os

# Initialize client (uses BASILICA_API_TOKEN from env)
client = BasilicaClient()

# Deploy an LLM for inference
deployment = client.deploy_vllm(
    name="tau-llm",
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu_count=1,
)

# Use with OpenAI client
from openai import OpenAI

llm_client = OpenAI(
    base_url=f"{deployment.url}/v1",
    api_key="not-needed",
)

# Make requests...
```
