# Lium GPU Pod Management — Integration Guide

This document describes how to use Lium for GPU pod management on Bittensor Subnet 51.

## Overview

Lium is a GPU marketplace on Bittensor Subnet 51. Rent GPU pods (H100, A100, RTX4090, etc.) for compute tasks with per-hour billing. Pods provide full SSH access, Docker templates, and persistent volumes.

Docs: https://docs.lium.io/cli/overview
Platform: https://lium.io

## Installation

The Lium CLI must be installed:

```bash
# Install lium-cli
pip install lium-cli

# Or with uv (recommended)
uv pip install lium-cli

# Verify installation
lium --version
```

## Authentication

Set your API key as an environment variable:

```bash
export LIUM_API_KEY="your_api_key_here"
```

Get your API key:
1. Register at https://lium.io/register
2. Navigate to account settings
3. Generate an API key in the API section

The tool automatically uses `LIUM_API_KEY` from the environment.

## Tool Usage

```bash
python -m tau.tools.lium COMMAND [OPTIONS]
```

## Environment Variables

- `LIUM_API_KEY`: Required API key (get from https://lium.io/register)
- `LIUM_SSH_KEY`: Optional path to SSH key (defaults to ~/.ssh/id_ed25519)

## Commands

### List Available GPUs (`ls`)

Browse available GPU nodes with real-time pricing:

```bash
python -m tau.tools.lium ls                    # All available nodes
python -m tau.tools.lium ls H100               # Only H100 GPUs
python -m tau.tools.lium ls A100               # Only A100 GPUs
python -m tau.tools.lium ls --max-price 2.5    # Under $2.50/hour
python -m tau.tools.lium ls --format json      # JSON output
python -m tau.tools.lium ls --region US        # Filter by region
python -m tau.tools.lium ls --min-memory 80    # Minimum 80GB GPU memory
```

**Options:**
- `GPU_TYPE`: Filter by GPU type (H100, A100, RTX4090, H200, A6000, etc.)
- `--region REGION`: Filter by region
- `--min-memory GB`: Minimum GPU memory
- `--max-price USD`: Maximum price per hour
- `--format FORMAT`: Output format (table, json, csv)

**Output:** Shows executor ID, GPU type, price/hour, availability, and ★ (Pareto-optimal) markers.

### List Active Pods (`ps`)

View your running pods and their status:

```bash
python -m tau.tools.lium ps                    # Running pods
python -m tau.tools.lium ps --all              # Include stopped pods
python -m tau.tools.lium ps --format json      # JSON output
python -m tau.tools.lium ps --sort cost        # Sort by cost
```

**Options:**
- `--all, -a`: Show all pods including stopped
- `--format FORMAT`: Output format (table, json, csv)
- `--sort FIELD`: Sort by field (name, status, cost, uptime)

**Output:** Shows pod name, executor, status, uptime, and cost.

### Create a Pod (`up`)

Create a new GPU pod. The tool automatically adds `--yes` to skip confirmation prompts.

```bash
# Basic usage
python -m tau.tools.lium up 1                  # Create on executor #1 from ls
python -m tau.tools.lium up                    # Interactive selection

# Filtering executors
python -m tau.tools.lium up --gpu H100         # Auto-select H100 executor
python -m tau.tools.lium up --gpu A100 --name my-training
python -m tau.tools.lium up --gpu H200 --country US
python -m tau.tools.lium up --gpu A6000 -c 2   # 2 GPUs, A6000 type
python -m tau.tools.lium up --ports 5          # Require ≥5 ports, allocate 5

# Volume management
python -m tau.tools.lium up 1 --volume id:brave-fox-3a        # Attach existing volume
python -m tau.tools.lium up 1 --volume new:name=my-data       # Create new volume
python -m tau.tools.lium up 1 --volume new:name=data,desc="Training data"

# Advanced
python -m tau.tools.lium up 1 --template_id abc123            # Specify template
```

**Arguments:**
- `EXECUTOR_ID`: Executor UUID, HUID, or index from last `ls`. If not provided, shows interactive selection.

**Options:**
- `--name, -n NAME`: Custom pod name (auto-generated if not specified)
- `--template_id, -t ID`: Specify template ID
- `--volume, -v SPEC`: Volume specification (see examples above)
- `--gpu TYPE`: Filter executors by GPU type (H200, A6000, etc.)
- `--count, -c NUM`: Filter by number of GPUs per pod
- `--country CODE`: Filter executors by ISO country code (e.g., US, FR)
- `--ports, -p NUM`: Filter executors with minimum NUM available ports AND allocate NUM ports

**Note:** The tool automatically adds `--yes` flag to skip confirmation prompts for automation.

### Remove Pods (`rm`)

Stop and remove pods. The tool automatically adds `--force` to skip confirmation prompts.

```bash
python -m tau.tools.lium rm my-pod             # Remove specific pod
python -m tau.tools.lium rm pod1 pod2 pod3     # Remove multiple pods
python -m tau.tools.lium rm all                # Remove all pods
python -m tau.tools.lium rm 1                  # Remove pod #1 from ps
python -m tau.tools.lium rm 1,2,3              # Remove multiple by index
```

**Arguments:**
- `POD`: Pod name(s), index from `ps`, comma-separated list, or "all"

**Options:**
- `--keep-volumes`: Preserve pod volumes when removing

**Note:** The tool automatically adds `--force` flag to skip confirmation prompts for automation.

### SSH Access (`ssh`)

Connect to a pod via SSH:

```bash
python -m tau.tools.lium ssh my-pod            # Interactive SSH
python -m tau.tools.lium ssh 1                 # SSH to pod #1 from ps
python -m tau.tools.lium ssh my-pod --command "nvidia-smi"  # Run command and exit
```

**Arguments:**
- `POD`: Pod name or index from `ps`

**Options:**
- `--command CMD`: Execute command and exit (non-interactive)
- `--port PORT`: SSH port (default: 22)
- `--key PATH`: Use specific SSH key

### Execute Commands (`exec`)

Run commands on pods without SSH:

```bash
python -m tau.tools.lium exec my-pod "nvidia-smi"
python -m tau.tools.lium exec my-pod "python train.py"
python -m tau.tools.lium exec 1 "ls -la"       # Pod #1 from ps
python -m tau.tools.lium exec all "pip install numpy"  # All pods
python -m tau.tools.lium exec 1,2,3 "apt update"  # Multiple pods
python -m tau.tools.lium exec my-pod "nvidia-smi" --output gpu_info.txt
```

**Arguments:**
- `POD`: Pod name, index, comma-separated list, or "all"
- `COMMAND`: Command to execute

**Options:**
- `--timeout SECONDS`: Command timeout
- `--output FILE`: Save output to file

### File Transfer (`scp` and `rsync`)

Copy files and sync directories to pods:

**SCP (single files/directories):**
```bash
python -m tau.tools.lium scp my-pod ./script.py           # Copy to /root/
python -m tau.tools.lium scp my-pod ./data.csv /root/datasets/  # Specific path
python -m tau.tools.lium scp 1 ./config.json              # Pod #1
python -m tau.tools.lium scp all ./requirements.txt        # All pods
python -m tau.tools.lium scp 1,2,3 ./model.py             # Multiple pods
python -m tau.tools.lium scp my-pod ./folder -r           # Copy directory recursively
```

**RSYNC (directory synchronization):**
```bash
python -m tau.tools.lium rsync my-pod ./project           # Sync directory
python -m tau.tools.lium rsync 1 ./data /root/datasets/   # Specific path
python -m tau.tools.lium rsync all ./configs --delete     # Mirror exactly
python -m tau.tools.lium rsync my-pod ./code --exclude "*.pyc"  # Exclude files
python -m tau.tools.lium rsync my-pod ./project --dry-run # Preview changes
```

**SCP Options:**
- `--recursive, -r`: Copy directories recursively
- `--preserve, -p`: Preserve file attributes

**RSYNC Options:**
- `--delete`: Delete files not in source
- `--exclude PATTERN`: Exclude files matching pattern
- `--dry-run`: Show what would be synced

### Templates (`templates`)

Browse available Docker templates for common workloads:

```bash
python -m tau.tools.lium templates             # List all templates
python -m tau.tools.lium templates pytorch     # Search for pytorch
python -m tau.tools.lium templates --category ml  # ML templates only
python -m tau.tools.lium templates --format json  # JSON output
```

**Arguments:**
- `SEARCH`: Search term to filter templates

**Options:**
- `--category CAT`: Filter by category (ml, web, database, etc.)
- `--format FORMAT`: Output format

### Configuration (`config`)

Manage CLI configuration:

```bash
python -m tau.tools.lium config show           # Show all config
python -m tau.tools.lium config get api.api_key  # Get specific value
python -m tau.tools.lium config set ssh.key_path ~/.ssh/new_key
```

### Funding (`fund`)

Fund account with TAO from Bittensor wallet:

```bash
python -m tau.tools.lium fund                  # Interactive mode
python -m tau.tools.lium fund -w default -a 10.0  # Fund 10 TAO
```

## Typical Workflow

1. **Browse GPUs**: `python -m tau.tools.lium ls H100` to find suitable hardware
2. **Create pod**: `python -m tau.tools.lium up --gpu H100 --name training`
3. **Copy code**: `python -m tau.tools.lium scp training ./train.py`
4. **Run training**: `python -m tau.tools.lium exec training "python train.py"`
5. **Monitor**: `python -m tau.tools.lium ssh training` or `python -m tau.tools.lium exec training "nvidia-smi"`
6. **Cleanup**: `python -m tau.tools.lium rm training` when done

## Using Pod Indices

You can use numbers from `ps` and `ls` output:

```bash
python -m tau.tools.lium ps                    # Note pod numbers
python -m tau.tools.lium ssh 1                 # SSH to pod #1
python -m tau.tools.lium exec 2 "ls"           # Execute on pod #2
python -m tau.tools.lium rm 3                  # Remove pod #3

python -m tau.tools.lium ls                    # Note executor numbers
python -m tau.tools.lium up 1                  # Create on executor #1
```

## Batch Operations

Many commands support batch operations:

```bash
# Execute on multiple pods
python -m tau.tools.lium exec 1,2,3 "apt update"
python -m tau.tools.lium exec all "nvidia-smi"

# Copy to multiple pods
python -m tau.tools.lium scp 1,2,3 ./config.json
python -m tau.tools.lium scp all ./requirements.txt

# Sync to all pods
python -m tau.tools.lium rsync all ./project
```

## Cost Tips

- Look for ★ (Pareto-optimal) nodes in `ls` output - these offer best price/performance
- Always remove pods when done: `python -m tau.tools.lium rm all`
- Use `ps` to monitor spending and uptime
- Filter by `--max-price` when browsing to stay within budget

## Integration with Tau

Tau can use Lium for:
1. GPU-accelerated training and inference
2. Running compute-intensive tasks on-demand
3. Testing models on different GPU types
4. Batch processing across multiple pods

**Example workflow:**
```bash
# Find suitable GPU
python -m tau.tools.lium ls H100 --max-price 3.0

# Create pod
python -m tau.tools.lium up --gpu H100 --name tau-training

# Copy training script
python -m tau.tools.lium scp tau-training ./train.py

# Run training
python -m tau.tools.lium exec tau-training "python train.py"

# Monitor progress
python -m tau.tools.lium exec tau-training "nvidia-smi"

# Cleanup
python -m tau.tools.lium rm tau-training
```

## Error Handling

The tool handles common errors:
- Missing API key: Warns and continues (lium CLI may have config)
- Missing CLI: Exits with installation instructions
- Timeout: Commands timeout after 5 minutes
- Pod not found: Returns error from lium CLI

## Exit Codes

- `0`: Success
- `1`: General error
- `2`: Configuration error
- `3`: API error
- `4`: SSH error
- `5`: Pod not found
- `6`: Permission denied
