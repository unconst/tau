#!/usr/bin/env python3
"""Lium GPU Pod Management Tool.

Manage GPU pods on the Lium network (Bittensor Subnet 51).

Usage:
    python -m tau.tools.lium ls                      # List available GPU nodes
    python -m tau.tools.lium ls H100                 # List only H100 GPUs
    python -m tau.tools.lium ls --max-price 2.5      # Under $2.50/hour
    python -m tau.tools.lium ps                      # List your active pods
    python -m tau.tools.lium up 1                    # Create pod on executor #1
    python -m tau.tools.lium up --gpu H100           # Create pod with H100 GPU
    python -m tau.tools.lium up --gpu A100 --name my-pod  # Create named pod
    python -m tau.tools.lium rm my-pod               # Remove a pod
    python -m tau.tools.lium rm all                  # Remove all pods
    python -m tau.tools.lium ssh my-pod              # SSH into pod
    python -m tau.tools.lium exec my-pod "nvidia-smi"  # Execute command on pod
    python -m tau.tools.lium templates               # List available templates
    python -m tau.tools.lium templates pytorch       # Search templates

Environment:
    LIUM_API_KEY: Your Lium API key (required)
    LIUM_SSH_KEY: Path to SSH key (optional, defaults to ~/.ssh/id_ed25519)

Notes:
    - The lium-cli package must be installed: pip install lium-cli
    - Get your API key from https://lium.io/register
"""

import os
import sys
import subprocess
import shutil
from typing import Optional, List


def check_lium_installed() -> bool:
    """Check if lium CLI is installed."""
    return shutil.which("lium") is not None


def check_api_key() -> Optional[str]:
    """Check if API key is configured."""
    return os.environ.get("LIUM_API_KEY")


def run_lium_command(args: List[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run a lium CLI command.
    
    Args:
        args: Command arguments (without 'lium' prefix)
        capture_output: If True, capture stdout/stderr
        
    Returns:
        CompletedProcess result
    """
    cmd = ["lium"] + args
    
    env = os.environ.copy()
    
    # Ensure API key is in environment
    api_key = check_api_key()
    if api_key:
        env["LIUM_API_KEY"] = api_key
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            env=env,
            timeout=300  # 5 minute timeout for most operations
        )
        return result
    except subprocess.TimeoutExpired:
        print("Error: Command timed out after 5 minutes", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: lium CLI not found. Install with: pip install lium-cli", file=sys.stderr)
        sys.exit(1)


def cmd_ls(args: List[str]) -> None:
    """List available GPU nodes."""
    run_lium_command(["ls"] + args)


def cmd_ps(args: List[str]) -> None:
    """List active pods."""
    run_lium_command(["ps"] + args)


def cmd_up(args: List[str]) -> None:
    """Create a new pod."""
    # For pod creation, we need to handle interactive mode
    # Add --yes to skip confirmation when automated
    if "--yes" not in args and "-y" not in args:
        args = args + ["--yes"]
    run_lium_command(["up"] + args)


def cmd_rm(args: List[str]) -> None:
    """Remove/stop pods."""
    # Add force flag to skip confirmation
    if "--force" not in args and "-f" not in args:
        args = args + ["--force"]
    run_lium_command(["rm"] + args)


def cmd_ssh(args: List[str]) -> None:
    """SSH into a pod."""
    run_lium_command(["ssh"] + args)


def cmd_exec(args: List[str]) -> None:
    """Execute command on pod."""
    run_lium_command(["exec"] + args)


def cmd_scp(args: List[str]) -> None:
    """Copy files to pods."""
    run_lium_command(["scp"] + args)


def cmd_rsync(args: List[str]) -> None:
    """Sync directories to pods."""
    run_lium_command(["rsync"] + args)


def cmd_templates(args: List[str]) -> None:
    """List available templates."""
    run_lium_command(["templates"] + args)


def cmd_config(args: List[str]) -> None:
    """Manage configuration."""
    run_lium_command(["config"] + args)


def cmd_fund(args: List[str]) -> None:
    """Fund account with TAO."""
    run_lium_command(["fund"] + args)


def print_usage() -> None:
    """Print usage information."""
    print(__doc__)
    print("\nCommands:")
    print("  ls [GPU_TYPE] [OPTIONS]     List available GPU nodes")
    print("  ps [OPTIONS]                List your active pods")
    print("  up [EXECUTOR] [OPTIONS]     Create a new pod")
    print("  rm POD [OPTIONS]            Remove/stop a pod")
    print("  ssh POD [OPTIONS]           SSH into a pod")
    print("  exec POD COMMAND            Execute command on pod")
    print("  scp POD FILE [PATH]         Copy files to pod")
    print("  rsync POD DIR [PATH]        Sync directory to pod")
    print("  templates [SEARCH]          List available templates")
    print("  config SUBCOMMAND           Manage configuration")
    print("  fund [OPTIONS]              Fund account with TAO")
    print("\nExamples:")
    print("  python -m tau.tools.lium ls H100 --max-price 3.0")
    print("  python -m tau.tools.lium up --gpu A100 --name training-pod")
    print("  python -m tau.tools.lium exec training-pod 'nvidia-smi'")
    print("  python -m tau.tools.lium rm training-pod")


def main() -> None:
    """Main entry point for Lium tool."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)
    
    command = sys.argv[1]
    args = sys.argv[2:]
    
    if command in ("--help", "-h", "help"):
        print_usage()
        sys.exit(0)
    
    # Check prerequisites
    if not check_lium_installed():
        print("Error: lium CLI not installed.", file=sys.stderr)
        print("Install with: pip install lium-cli", file=sys.stderr)
        print("Then configure with: lium init", file=sys.stderr)
        sys.exit(1)
    
    api_key = check_api_key()
    if not api_key and command not in ("config", "--help", "-h", "help"):
        print("Warning: LIUM_API_KEY not set in environment.", file=sys.stderr)
        print("Set it with: export LIUM_API_KEY=your-api-key", file=sys.stderr)
        print("Or configure with: lium init", file=sys.stderr)
        # Continue anyway - lium might have it in config file
    
    # Dispatch to command handlers
    commands = {
        "ls": cmd_ls,
        "ps": cmd_ps,
        "up": cmd_up,
        "rm": cmd_rm,
        "ssh": cmd_ssh,
        "exec": cmd_exec,
        "scp": cmd_scp,
        "rsync": cmd_rsync,
        "templates": cmd_templates,
        "config": cmd_config,
        "fund": cmd_fund,
    }
    
    if command in commands:
        commands[command](args)
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
