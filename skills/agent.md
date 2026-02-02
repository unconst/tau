## Cursor CLI `agent` command — docs extract

This file consolidates Cursor documentation about using the `agent` CLI command (interactive, non-interactive/headless, modes, commands/flags, configuration/permissions, MCP, Shell Mode, and CI/GitHub Actions workflows).

### Sources (Cursor docs)

- `https://cursor.com/docs/cli/overview`
- `https://cursor.com/docs/cli/installation`
- `https://cursor.com/docs/cli/using`
- `https://cursor.com/docs/cli/shell-mode`
- `https://cursor.com/docs/cli/mcp`
- `https://cursor.com/docs/cli/reference/parameters`
- `https://cursor.com/docs/cli/reference/slash-commands`
- `https://cursor.com/docs/cli/reference/authentication`
- `https://cursor.com/docs/cli/reference/permissions`
- `https://cursor.com/docs/cli/reference/configuration`
- `https://cursor.com/docs/cli/reference/output-format`
- `https://cursor.com/docs/cli/headless`
- `https://cursor.com/docs/cli/github-actions`
- `https://cursor.com/docs/cli/cookbook/code-review`
- `https://cursor.com/docs/cli/cookbook/update-docs`
- `https://cursor.com/docs/cli/cookbook/fix-ci`
- `https://cursor.com/docs/cli/cookbook/secret-audit`
- `https://cursor.com/docs/cli/cookbook/translate-keys`

---

## Install / verify / update

### Install

macOS / Linux / WSL:

```bash
curl https://cursor.com/install -fsS | bash
```

Windows (PowerShell):

```powershell
irm 'https://cursor.com/install?win32=true' | iex
```

### Verify installation

```bash
agent --version
```

### Post-install PATH setup

Bash:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Zsh:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Update Cursor CLI

```bash
agent update
# or
agent upgrade
```

---

## Authentication

Cursor CLI supports:

- **Browser-based login (recommended)**
- **API keys**

### Browser authentication (recommended)

```bash
agent login
agent status
agent logout
```

Docs notes:

- `agent login` opens your default browser to authenticate with your Cursor account; credentials are stored locally afterward.
- `agent status` shows whether you’re authenticated, account info, and endpoint config.

### API key authentication (automation / CI)

Generate a key in the Cursor dashboard under **Integrations > User API Keys**, then set it via:

Environment variable (recommended):

```bash
export CURSOR_API_KEY=your_api_key_here
agent "implement user authentication"
```

CLI flag:

```bash
agent --api-key your_api_key_here "implement user authentication"
```

### Troubleshooting flags mentioned in docs

- `--insecure`: for SSL certificate errors (development environments)
- `--endpoint`: specify a custom API endpoint

---

## Quick start (interactive)

Start an interactive agent session:

```bash
agent
```

Start interactive session with an initial prompt:

```bash
agent "refactor the auth module to use JWT tokens"
```

Docs note (prompting): stating intent clearly is recommended, e.g.:

- `"do not write any code"` to help ensure no edits occur when you’re planning.

---

## Modes

The CLI supports the same modes as the editor.

| Mode | What it does | How to switch |
| --- | --- | --- |
| **Agent** | Full access to tools for complex coding tasks | default |
| **Plan** | Clarifying questions + plan before coding | Shift+Tab, `/plan`, or `--mode=plan` |
| **Ask** | Read-only exploration (no file edits) | `/ask` or `--mode=ask` |

---

## Interactive usage details

### Navigation

- Use **ArrowUp** to access previous messages (cycle through history).

### Input shortcuts

- **Shift+Tab**: rotate between modes (Agent / Plan / Ask)
- **Shift+Enter**: insert newline instead of submitting (multi-line prompts)
- **Ctrl+D**: exit CLI (double-press to exit)
- **Ctrl+J or +Enter**: alternative newline insertion

Terminal note from docs:

- **Shift+Enter** works in iTerm2, Ghostty, Kitty, Warp, and Zed.
- Run `/setup-terminal` to auto-configure **Option+Enter** for Apple Terminal, Alacritty, or VS Code.

### Reviewing changes

- **Ctrl+R**: review changes
- Press **i**: add follow-up instructions (during review)
- Use **ArrowUp/ArrowDown**: scroll
- Use **ArrowLeft/ArrowRight**: switch files

### Selecting context

- Use `@` to include files/folders in context.
- Use `/compress` to summarize the conversation and free context space.

### Command approval (interactive)

Before running terminal commands, the CLI asks you to approve **(y)** or reject **(n)** execution.

---

## Slash commands (interactive)

| Command | Description |
| --- | --- |
| `/plan` | Switch to Plan mode |
| `/ask` | Switch to Ask mode |
| `/model <model>` | Set or list models |
| `/auto-run [state]` | Toggle auto-run (default) or set `[on|off|status]` |
| `/new-chat` | Start a new chat session |
| `/vim` | Toggle Vim keys |
| `/help [command]` | Show help |
| `/feedback <message>` | Share feedback with the team |
| `/resume <chat>` | Resume a previous chat by folder name |
| `/usage` | View Cursor streaks and usage stats |
| `/about` | Show environment and CLI setup details |
| `/copy-req-id` | Copy last request ID |
| `/logout` | Sign out from Cursor |
| `/quit` | Exit |
| `/setup-terminal` | Auto-configure terminal keybindings |
| `/mcp list` | Browse/enable/configure MCP servers |
| `/mcp enable <name>` | Enable an MCP server |
| `/mcp disable <name>` | Disable an MCP server |
| `/rules` | Create/edit rules |
| `/commands` | Create/edit commands |
| `/compress` | Summarize conversation to free context |

---

## CLI commands and global flags

### When you run `agent` with no subcommand

The CLI starts in **interactive chat mode** by default.

You can provide an initial prompt as an argument:

```bash
agent "your prompt here"
```

### Global options (can be used with any command)

| Option | Description |
| --- | --- |
| `-v, --version` | Output the version number |
| `-a, --api-key <key>` | API key for authentication (or `CURSOR_API_KEY` env var) |
| `-p, --print` | Print responses to console (non-interactive). Has access to tools, including write and bash. |
| `--output-format <format>` | Output format (only works with `--print`): `text`, `json`, or `stream-json` (default: `text`) |
| `--stream-partial-output` | Stream partial output as individual text deltas (only works with `--print` + `stream-json`) |
| `-b, --background` | Start in background mode (open composer picker on launch) |
| `--fullscreen` | Enable fullscreen mode |
| `--resume [chatId]` | Resume a chat session |
| `-m, --model <model>` | Model to use |
| `--mode <mode>` | Set mode: `agent` (default), `plan`, or `ask` |
| `--list-models` | List all available models |
| `-f, --force` | Force allow commands unless explicitly denied |
| `-h, --help` | Display help |

### Commands

| Command | Description | Usage |
| --- | --- | --- |
| `login` | Authenticate with Cursor | `agent login` |
| `logout` | Sign out and clear stored auth | `agent logout` |
| `status` | Check auth status | `agent status` |
| `models` | List all available models | `agent models` |
| `mcp` | Manage MCP servers | `agent mcp` |
| `update` / `upgrade` | Update Cursor Agent | `agent update` / `agent upgrade` |
| `ls` | List previous conversations | `agent ls` |
| `resume` | Resume latest conversation | `agent resume` |
| `help [command]` | Help | `agent help [command]` |

### Sessions / history / resume

```bash
agent ls
agent resume
agent --resume="chat-id-here"
```

Docs notes:

- Continue from an existing thread with `--resume [thread id]`.
- You can also use the `/resume` slash command (interactive).

---

## Cloud Agent handoff

You can send your message to a Cloud Agent by prefixing it with `&`:

```bash
& refactor the auth module and add comprehensive tests
```

Pick up Cloud Agent tasks at `https://cursor.com/agents`.

---

## Non-interactive / print mode (scripts, CI, automation)

### Basics

Use `-p` / `--print`:

```bash
agent -p "find and fix performance issues" --model "gpt-5.2"
```

Control formatting with `--output-format`:

```bash
agent -p "review these changes for security issues" --output-format text
```

Docs note: `--output-format` is only valid when printing (`--print`) **or when print mode is inferred** (non-TTY stdout or piped stdin).

### Enabling file changes in print mode

Docs guidance (Headless CLI): combine `--print` with `--force` to allow file modifications in scripts:

```bash
agent -p --force "Refactor this code to use modern ES6+ syntax"

# Without --force, changes are only proposed, not applied
agent -p "Add JSDoc comments to this file"
```

Batch example from docs:

```bash
find src/ -name "*.js" | while read file; do
  agent -p --force "Add comprehensive JSDoc comments to $file"
done
```

Docs note: `--force` allows the agent to make direct file changes without confirmation.

### Working with images / media in headless mode

Docs: include file paths in your prompt; the agent will read files via tool calls (supports images, videos, and other formats).

Examples:

```bash
agent -p "Analyze this image and describe what you see: ./screenshot.png"
agent -p "Compare these two images and identify differences: ./before.png ./after.png"
agent -p "Review src/app.ts and designs/homepage.png. Suggest improvements to match the design."
```

---

## Output formats (`--print` + `--output-format`)

Docs: `--output-format` supports:

- `text` (default)
- `json`
- `stream-json` (NDJSON)

### `text` output format

Only prints the final assistant message (after the last tool call). No intermediate progress updates.

Example output shown in docs:

```text
The command to move this branch onto main is `git rebase --onto main HEAD~3`.
```

### `json` output format

Emits a single JSON object on success (newline-terminated). No deltas/tool events.

On failure: non-zero exit code + error message on stderr; no well-formed JSON is emitted.

Success response shape (docs):

```json
{
  "type": "result",
  "subtype": "success",
  "is_error": false,
  "duration_ms": 1234,
  "duration_api_ms": 1234,
  "result": "<full assistant text>",
  "session_id": "<uuid>",
  "request_id": "<optional request id>"
}
```

### `stream-json` output format (NDJSON)

Docs: emits newline-delimited JSON, one JSON object per line, representing events during execution. Aggregates text deltas and outputs **one line per assistant message** (the complete message between tool calls).

The stream ends with a terminal `result` event on success.

On failure: non-zero exit code + error on stderr; the stream may end early without a terminal event.

#### Streaming partial output

Use `--stream-partial-output` with `--output-format stream-json` for incremental text deltas (multiple `assistant` events per message). Reconstruct the full response by concatenating `message.content[].text`.

#### Event types (docs)

System init:

```json
{
  "type": "system",
  "subtype": "init",
  "apiKeySource": "env|flag|login",
  "cwd": "/absolute/path",
  "session_id": "<uuid>",
  "model": "<model display name>",
  "permissionMode": "default"
}
```

User message:

```json
{
  "type": "user",
  "message": {
    "role": "user",
    "content": [{ "type": "text", "text": "<prompt>" }]
  },
  "session_id": "<uuid>"
}
```

Assistant message:

```json
{
  "type": "assistant",
  "message": {
    "role": "assistant",
    "content": [{ "type": "text", "text": "<complete message text>" }]
  },
  "session_id": "<uuid>"
}
```

Tool call started:

```json
{
  "type": "tool_call",
  "subtype": "started",
  "call_id": "<string id>",
  "tool_call": {
    "readToolCall": {
      "args": { "path": "file.txt" }
    }
  },
  "session_id": "<uuid>"
}
```

Tool call completed (read example):

```json
{
  "type": "tool_call",
  "subtype": "completed",
  "call_id": "<string id>",
  "tool_call": {
    "readToolCall": {
      "args": { "path": "file.txt" },
      "result": {
        "success": {
          "content": "file contents...",
          "isEmpty": false,
          "exceededLimit": false,
          "totalLines": 54,
          "totalChars": 1254
        }
      }
    }
  },
  "session_id": "<uuid>"
}
```

Terminal result (success):

```json
{
  "type": "result",
  "subtype": "success",
  "duration_ms": 1234,
  "duration_api_ms": 1234,
  "is_error": false,
  "result": "<full assistant text>",
  "session_id": "<uuid>",
  "request_id": "<optional request id>"
}
```

#### Output-format implementation notes (docs)

- Each event is one line terminated by `\n`
- `thinking` events are suppressed in print mode and will not appear in output
- Fields may be added over time; consumers should ignore unknown fields
- `json` waits for completion before outputting
- `stream-json` outputs complete agent messages
- Tool call IDs can correlate start/completion events
- Session IDs remain consistent throughout a single `agent` execution

---

## Shell Mode

Docs: Shell Mode runs shell commands directly from the CLI without leaving your conversation (quick, non-interactive commands, safety checks, output in conversation).

### Command execution

Commands run in your login shell (`$SHELL`) with the CLI’s working directory + environment. To run in another directory, chain commands:

```bash
cd subdir && npm test
```

### Output behavior

- Large outputs are truncated automatically.
- Long-running processes time out to maintain performance.

### Limitations

- Commands time out after 30 seconds
- Long-running processes / servers / interactive prompts are not supported
- Use short, non-interactive commands

### Permissions

- Commands are checked against your permissions and team settings.
- Admin policies may block certain commands.
- Commands with redirection cannot be allowlisted inline.

### Troubleshooting tips (docs)

- If a command hangs: Ctrl+C and add non-interactive flags
- When prompted for permissions: approve once or add to allowlist with Tab
- For truncated output: Ctrl+O to expand
- To run in different directories: use `cd && ...` since changes don’t persist
- Shell Mode supports zsh and bash from `$SHELL`

### FAQ (docs)

- **Does `cd` persist across runs?** No. Each command runs independently. Use `cd && ...` each time.
- **Can I change the timeout?** No. Commands are limited to 30 seconds and not configurable.
- **Where are permissions configured?** Managed by CLI and team configuration; use the decision banner to add allowlists.
- **How do I exit Shell Mode?** Press Escape on empty input, Backspace/Delete on empty input, or Ctrl+C to clear and exit.

---

## MCP (Model Context Protocol)

Docs: Cursor CLI supports MCP servers and uses the same configuration as the editor.

### `agent mcp` subcommands

List servers (interactive menu):

```bash
agent mcp list
```

Docs says the MCP list UI shows:

- Server names and identifiers
- Connection status (connected/disconnected)
- Configuration source (project or global)
- Transport method (stdio, HTTP, SSE)

List tools for a server:

```bash
agent mcp list-tools <identifier>
```

Login to a server:

```bash
agent mcp login <identifier>
```

Enable/disable:

```bash
agent mcp enable <identifier>
agent mcp disable <identifier>
```

Docs notes:

- `/mcp list` provides the same interactive menu in interactive mode.
- MCP server names with spaces are supported in `/mcp` commands.
- After `agent mcp login`, the agent can use authenticated MCP tools immediately.
- Config precedence matches editor: **project → global → nested**, discovered from parent directories.

Example from docs:

```bash
agent mcp list
agent mcp list-tools playwright
agent -p "Navigate to google.com and take a screenshot of the search page"
```

---

## Rules (CLI)

Docs: CLI agent supports the same rules system as the editor:

- Put rules in `.cursor/rules/`
- The CLI also reads `AGENTS.md` and `CLAUDE.md` at the project root (if present) and applies them as rules alongside `.cursor/rules/`.

---

## CLI configuration (`cli-config.json` / project `cli.json`)

Docs: configure the Agent CLI using `cli-config.json`.

### File location

| Type | Platform | Path |
| --- | --- | --- |
| Global | macOS/Linux | `~/.cursor/cli-config.json` |
| Global | Windows | `$env:USERPROFILE\.cursor\cli-config.json` |
| Project | All | `<project>/.cursor/cli.json` |

Docs note: only **permissions** can be configured at the project level; other settings must be global.

### Environment variable overrides

- `CURSOR_CONFIG_DIR`: custom directory path
- `XDG_CONFIG_HOME` (Linux/BSD): uses `$XDG_CONFIG_HOME/cursor/cli-config.json`

### Schema

Required fields:

- `version` (current: `1`)
- `editor.vimMode` (default: `false`)
- `permissions.allow` (string[])
- `permissions.deny` (string[])

Optional fields:

- `model` (object)
- `hasChangedDefaultModel` (boolean)
- `network.useHttp1ForAgent` (boolean; default `false`)
- `attribution.attributeCommitsToAgent` (boolean; default `true`) — add `Co-authored-by: Cursor` trailer to agent commits
- `attribution.attributePRsToAgent` (boolean; default `true`) — add “Made with Cursor” footer to agent PRs

### Examples

Minimal config:

```json
{
  "version": 1,
  "editor": { "vimMode": false },
  "permissions": { "allow": ["Shell(ls)"], "deny": [] }
}
```

Enable Vim mode:

```json
{
  "version": 1,
  "editor": { "vimMode": true },
  "permissions": { "allow": ["Shell(ls)"], "deny": [] }
}
```

Configure permissions:

```json
{
  "version": 1,
  "editor": { "vimMode": false },
  "permissions": {
    "allow": ["Shell(ls)", "Shell(echo)"],
    "deny": ["Shell(rm)"]
  }
}
```

### Troubleshooting notes (docs)

Config errors: move file aside and restart:

```bash
mv ~/.cursor/cli-config.json ~/.cursor/cli-config.json.bad
```

Other notes:

- Pure JSON only (no comments)
- CLI self-repairs missing fields
- Corrupted files are backed up as `.bad` and recreated
- Some fields are CLI-managed and may be overwritten

### Model selection

Use `/model`:

```bash
/model auto
/model gpt-5.2
/model sonnet-4.5-thinking
```

### Proxy configuration

Set environment variables:

```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
export NODE_USE_ENV_PROXY=1
```

If SSL inspection is present (MITM), trust your CA cert:

```bash
export NODE_EXTRA_CA_CERTS=/path/to/corporate-ca-cert.pem
```

### HTTP/1.1 fallback for proxies

Docs: some enterprise proxies (e.g. Zscaler) don’t support HTTP/2 bidirectional streaming. Configure HTTP/1.1 + SSE:

```json
{
  "version": 1,
  "editor": { "vimMode": false },
  "permissions": { "allow": [], "deny": [] },
  "network": { "useHttp1ForAgent": true }
}
```

---

## Permissions

Docs: configure permissions using tokens in CLI config.

Where to set:

- Global: `~/.cursor/cli-config.json`
- Project-specific: `<project>/.cursor/cli.json`

### Permission token types

Shell commands:

- **Format:** `Shell(commandBase)`
- Docs says `commandBase` is the first token in the command line.

Examples:

- `Shell(ls)` — allow `ls`
- `Shell(git)` — allow any `git` subcommand
- `Shell(npm)` — allow npm
- `Shell(rm)` — deny destructive removal

File reads:

- **Format:** `Read(pathOrGlob)` (supports globs)
- Examples: `Read(src/**/*.ts)`, `Read(**/*.md)`, `Read(.env*)`, `Read(/etc/passwd)`

File writes:

- **Format:** `Write(pathOrGlob)` (supports globs)
- Docs note: when using in print mode, `--force` is required to write files.
- Examples: `Write(src/**)`, `Write(package.json)`, `Write(**/*.key)`, `Write(**/.env*)`

### Example permissions block

```json
{
  "permissions": {
    "allow": ["Shell(ls)", "Shell(git)", "Read(src/**/*.ts)", "Write(package.json)"],
    "deny": ["Shell(rm)", "Read(.env*)", "Write(**/*.key)"]
  }
}
```

### Pattern matching notes (docs)

- Globs use `**`, `*`, `?`
- Relative paths are scoped to the current workspace
- Absolute paths can target files outside the project
- Deny rules take precedence over allow rules

---

## GitHub Actions / CI usage

### Basic GitHub Actions integration (docs)

```yaml
- name: Install Cursor CLI
  run: |
    curl https://cursor.com/install -fsS | bash
    echo "$HOME/.cursor/bin" >> $GITHUB_PATH

- name: Run Cursor Agent
  env:
    CURSOR_API_KEY: ${{ secrets.CURSOR_API_KEY }}
  run: |
    agent -p "Your prompt here" --model gpt-5.2
```

Windows runners note: `irm 'https://cursor.com/install?win32=true' | iex`

### CI autonomy levels (docs)

Full autonomy: allow the agent to do git operations, API calls, PR operations, etc.

Restricted autonomy: restrict agent to file modifications; keep git pushes/PR comments deterministic in separate workflow steps.

Example restricted pattern (docs):

```yaml
- name: Generate docs updates (restricted)
  run: |
    agent -p "IMPORTANT: Do NOT create branches, commit, push, or post PR comments.
    Only modify files in the working directory. A later workflow step handles publishing."

- name: Publish docs branch (deterministic)
  run: |
    git checkout -B "docs/${{ github.head_ref }}"
    git add -A
    git commit -m "docs: update for PR"
    git push origin "docs/${{ github.head_ref }}"

- name: Post PR comment (deterministic)
  run: |
    gh pr comment ${{ github.event.pull_request.number }} --body "Docs updated"
```

### Permission-based restrictions (docs example)

```json
{
  "permissions": {
    "allow": ["Read(**/*.md)", "Write(docs/**/*)", "Shell(grep)", "Shell(find)"],
    "deny": ["Shell(git)", "Shell(gh)", "Write(.env*)", "Write(package.json)"]
  }
}
```

### GitHub secrets (docs)

Set `CURSOR_API_KEY` as a GitHub secret (examples using `gh`):

```bash
gh secret set CURSOR_API_KEY --repo OWNER/REPO --body "$CURSOR_API_KEY"
gh secret set CURSOR_API_KEY --org ORG --visibility all --body "$CURSOR_API_KEY"
```

And in workflows:

```yaml
env:
  CURSOR_API_KEY: ${{ secrets.CURSOR_API_KEY }}
```

---

## Cookbook workflows (GitHub Actions)

The docs provide multiple end-to-end GitHub Actions workflows that run `agent` in print mode (typically `agent -p` or `agent --print`) and often use `--force`, `--model`, and `--output-format=text`.

### Code Review workflow (docs)

The cookbook includes an example workflow `.github/workflows/cursor-code-review.yml` that runs on pull requests and uses `agent --force --model ... --output-format=text --print "<long prompt>"`.

Example command pattern from docs:

```bash
agent --force --model "$MODEL" --output-format=text --print '...prompt...'
```

It also describes an optional “blocking review” mechanism where the prompt writes `CRITICAL_ISSUES_FOUND=true/false` to `$GITHUB_ENV`, and a subsequent step fails the workflow if blocking is enabled and critical issues are found.

The cookbook also shows setting up agent permissions via `.cursor/cli.json` to prevent operations like pushing code or creating PRs:

```json
{
  "permissions": {
    "deny": ["Shell(git push)", "Shell(gh pr create)", "Write(**)"]
  }
}
```

### Update Docs workflow (docs)

The cookbook includes a workflow `.github/workflows/update-docs.yml` that runs on PRs and uses `agent -p "...prompt..." --force --model "$MODEL" --output-format=text` to:

- compute incremental diffs
- update docs
- maintain a persistent docs branch for the PR head
- push changes
- post or update a PR comment (but not create/edit PRs directly), including a compare link

### Fix CI Failures workflow (docs)

The cookbook includes a workflow `.github/workflows/fix-ci.yml` that runs on failed workflow runs and uses `agent -p "...prompt..." --force --model "$MODEL" --output-format=text` to:

- locate the PR associated with a failed run
- maintain a persistent fix branch
- attempt minimal edits to resolve the CI failure
- push changes
- post/update a PR comment with a compare link

### Secret Audit workflow (docs)

The cookbook includes a scheduled/manual workflow `.github/workflows/secret-audit.yml` that uses `agent -p "...prompt..." --force --model "$MODEL" --output-format=text` to:

- scan for secrets exposure
- detect risky workflow patterns
- propose minimal hardening edits (including pinning actions, reducing permissions, adding guardrails)
- maintain a persistent branch
- push changes
- optionally comment on the most recently updated open PR with a compare link

### Translate Keys workflow (docs)

The cookbook includes a workflow `.github/workflows/translate-keys.yml` that uses `agent -p "...prompt..." --force --model "$MODEL" --output-format=text` to:

- detect i18n keys changed in a PR
- fill only missing locales without overwriting existing translations
- validate JSON formatting/schemas
- maintain a persistent translate branch
- post/update a PR comment with a compare link

