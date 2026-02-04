## Targon Rentals — GPU/CPU Container Rentals

This file documents how to create and access dedicated GPU/CPU machines (Rentals) on Targon.

### Source

- `https://docs.targon.com/guides/rentals`

---

## What is a Rental?

A Rental is a dedicated, persistent container that provides the experience of a traditional VPS. It runs as a lightweight, fast-starting container with:

- Stateful environment
- Full root access
- SSH connectivity

Ideal for: long-running tasks, interactive development environments, workloads requiring dedicated machine control.

---

## Creating a Rental

### Via Dashboard

1. Go to [Targon Dashboard](https://targon.com/dashboard)
2. Navigate to **Rentals** section
3. Click **Create Rental**

### Configuration Options

**Server Type:**
- GPU or CPU

**Configuration Types:**
- **Custom**: Specify Docker image, env vars, ports, commands, arguments
- **Private Template**: Use saved custom configurations
- **Public Template**: Pre-configured templates with popular software

### Custom Configuration Details

| Setting | Description |
|---------|-------------|
| **Image** | Docker image (Docker Hub, GHCR, Quay.io, or private) |
| **Environment Variables** | Key-value pairs for your application |
| **Service Ports** | Expose container ports |
| **Commands** | Program(s) to run on startup (e.g., `python`) |
| **Arguments** | Inputs/options for commands (e.g., `my_script.py`) |

### Additional Settings

- **Storage**: Attach pre-existing or new volumes
- **Default Shell**: `/bin/bash`, `/bin/zsh`, etc.
- **SSH Key**: Required for access (add new or use existing)
- **Rental Name**: Descriptive identifier

### Deploy

Click **Create Rental** — available in seconds.

---

## Connecting via SSH

### Get the SSH Command

1. Go to your Rental's detail page in the Targon dashboard
2. Copy the provided SSH command

### SSH Command Format

```bash
ssh <rental-id>@ssh.deployments.targon.com
```

Example:

```bash
ssh rentals-abc123@ssh.deployments.targon.com
```

### Using a Non-Default SSH Key

```bash
ssh -i ~/.ssh/your_key rentals-abc123@ssh.deployments.targon.com
```

---

## Connecting with VS Code

### Prerequisites

Your rental's image must have SFTP software installed for VS Code Remote - SSH to work.

### Installing SFTP (Ubuntu)

If using a base Ubuntu image:

```bash
apt-get update && apt-get install -y openssh-sftp-server
```

### Connect

Use the **Remote - SSH** extension with the same SSH host string from the dashboard:

```
rentals-abc123@ssh.deployments.targon.com
```

---

## Quick Reference

| Task | Command/Action |
|------|----------------|
| Create rental | Dashboard → Rentals → Create Rental |
| SSH connect | `ssh <rental-id>@ssh.deployments.targon.com` |
| VS Code connect | Remote-SSH extension with same host |
| Install SFTP | `apt-get update && apt-get install -y openssh-sftp-server` |
