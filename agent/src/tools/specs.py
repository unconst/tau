"""Tool specifications for SuperAgent - defines JSON schemas for all tools."""

from __future__ import annotations

from typing import Any

# Shell command tool
SHELL_COMMAND_SPEC: dict[str, Any] = {
    "name": "shell_command",
    "description": """Runs a shell command and returns its output.
Always set the `workdir` param when using this tool. Do not use `cd` unless absolutely necessary.
Use `rg` (ripgrep) for searching text or files as it's much faster than grep.""",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "workdir": {
                "type": "string",
                "description": "The working directory to execute the command in",
            },
            "timeout_ms": {
                "type": "number",
                "description": "The timeout for the command in milliseconds",
            },
            "persist_approval": {
                "type": "boolean",
                "description": "When true, persist approval as an exec-policy allow prefix for similar commands.",
            },
            "abort_on_denied": {
                "type": "boolean",
                "description": "When true, abort the current turn if policy denies this command.",
            },
        },
        "required": ["command"],
    },
}

# Read file tool
READ_FILE_SPEC: dict[str, Any] = {
    "name": "read_file",
    "description": """Reads a local file with hashline-tagged lines.
Each line is returned as 'line_number:hash|content' where hash is a 2-char hex tag.
Use the line_number:hash references with hashline_edit to make precise edits.
Supports reading specific ranges with offset and limit parameters.""",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or relative path to the file",
            },
            "offset": {
                "type": "number",
                "description": "The line number to start reading from (1-indexed, default: 1)",
            },
            "limit": {
                "type": "number",
                "description": "The maximum number of lines to return (default: 500). Use offset+limit to read specific sections of large files.",
            },
        },
        "required": ["file_path"],
    },
}

# List directory tool
LIST_DIR_SPEC: dict[str, Any] = {
    "name": "list_dir",
    "description": """Lists entries in a local directory with type indicators.
Directories are marked with '/', symlinks with '@'.
Supports recursive listing with configurable depth.""",
    "parameters": {
        "type": "object",
        "properties": {
            "dir_path": {
                "type": "string",
                "description": "Absolute or relative path to the directory to list",
            },
            "offset": {
                "type": "number",
                "description": "The entry number to start listing from (1-indexed, default: 1)",
            },
            "limit": {
                "type": "number",
                "description": "The maximum number of entries to return (default: 50)",
            },
            "depth": {
                "type": "number",
                "description": "The maximum directory depth to traverse (default: 2)",
            },
        },
        "required": ["dir_path"],
    },
}

# Grep files tool
GREP_FILES_SPEC: dict[str, Any] = {
    "name": "grep_files",
    "description": """Finds files whose contents match the pattern.
Uses ripgrep (rg) for fast searching.
Returns matching lines with surrounding context, showing filepath:line_number:content.
The limit parameter controls total match results (not files).
If no matches are found, returns a clear 'No matches found' message.
Do NOT retry the same pattern — if you get no matches, verify the search path exists with list_dir or glob_files first.""",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regular expression pattern to search for",
            },
            "include": {
                "type": "string",
                "description": "Optional glob to filter which files are searched (e.g., '*.py', '*.{ts,tsx}')",
            },
            "path": {
                "type": "string",
                "description": "Directory or file path to search. Defaults to working directory.",
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of match results to return (default: 50)",
            },
            "context_lines": {
                "type": "number",
                "description": "Number of surrounding context lines to show per match (default: 2, max: 5)",
            },
        },
        "required": ["pattern"],
    },
}

# Apply patch tool
APPLY_PATCH_SPEC: dict[str, Any] = {
    "name": "apply_patch",
    "description": """Applies file patches to create, update, or delete files.

Patch format:
*** Begin Patch
*** Add File: <path>
+line to add
*** Update File: <path>
@@ context line
-old line
+new line
*** Delete File: <path>
*** End Patch

Rules:
- Use @@ with context to identify where to make changes
- Prefix new lines with + (even for new files)
- Prefix removed lines with -
- Use 3 lines of context before and after changes
- File paths must be relative, never absolute""",
    "parameters": {
        "type": "object",
        "properties": {
            "patch": {
                "type": "string",
                "description": "The patch content following the format described above",
            },
        },
        "required": ["patch"],
    },
}

# View image tool
VIEW_IMAGE_SPEC: dict[str, Any] = {
    "name": "view_image",
    "description": """View a local image from the filesystem.
Only use this if given a full filepath by the user, and the image isn't already attached.
Supported formats: PNG, JPEG, GIF, WebP, BMP.
The image will be loaded and attached to the conversation for analysis.""",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Local filesystem path to the image file",
            },
        },
        "required": ["path"],
    },
}

# Write file tool
WRITE_FILE_SPEC: dict[str, Any] = {
    "name": "write_file",
    "description": """Write content to a file.
Creates the file if it doesn't exist, or overwrites if it does.
Parent directories are created automatically.""",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
        },
        "required": ["file_path", "content"],
    },
}

# Update plan tool
UPDATE_PLAN_SPEC: dict[str, Any] = {
    "name": "update_plan",
    "description": """Updates the task plan to track progress.
Use this to show the user your planned steps and mark them as completed.
Each step should be 5-7 words maximum.""",
    "parameters": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Short description of the step (5-7 words)",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                            "description": "Current status of the step",
                        },
                    },
                    "required": ["description", "status"],
                },
                "description": "List of plan steps with their status",
            },
            "explanation": {
                "type": "string",
                "description": "Optional explanation of why the plan changed",
            },
        },
        "required": ["steps"],
    },
}

# String replace tool
STR_REPLACE_SPEC: dict[str, Any] = {
    "name": "str_replace",
    "description": """Performs exact string replacement in a file.
Use this for targeted edits — more reliable than apply_patch for single changes.
The old_string must match the file content exactly (including whitespace and indentation).
The edit will FAIL if old_string is not unique in the file unless replace_all is true.""",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to modify",
            },
            "old_string": {
                "type": "string",
                "description": "The exact text to find and replace (must match file content exactly)",
            },
            "new_string": {
                "type": "string",
                "description": "The replacement text",
            },
            "replace_all": {
                "type": "boolean",
                "description": "If true, replace all occurrences. Default false (requires unique match).",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    },
}

# Hashline edit tool
HASHLINE_EDIT_SPEC: dict[str, Any] = {
    "name": "hashline_edit",
    "description": """Edit a file using line:hash references from read_file/grep output.
Each line in read_file output has format 'line_number:hash|content'.
Reference these tags to make precise edits without reproducing old content.

Operations:
- replace: Replace line(s). Provide start (and optional end for ranges) + content.
- insert: Insert new lines after the referenced line. Provide start + content.
- delete: Delete line(s). Provide start (and optional end for ranges).

Example: To replace lines 5:a3 through 7:0e with new code:
  {"op": "replace", "start": "5:a3", "end": "7:0e", "content": "new code here"}

If the file changed since you read it, hashes won't match and the edit is rejected.
Multiple operations are applied bottom-up to preserve line numbers.""",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to edit",
            },
            "operations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": ["replace", "insert", "delete"],
                            "description": "Operation type",
                        },
                        "start": {
                            "type": "string",
                            "description": "Line reference as 'line_number:hash' (e.g. '5:a3')",
                        },
                        "end": {
                            "type": "string",
                            "description": "End line reference for range operations (e.g. '10:f1'). Omit for single-line ops.",
                        },
                        "content": {
                            "type": "string",
                            "description": "New content (required for replace and insert, omit for delete)",
                        },
                    },
                    "required": ["op", "start"],
                },
                "description": "List of edit operations to apply",
            },
        },
        "required": ["file_path", "operations"],
    },
}

# Glob files tool
GLOB_FILES_SPEC: dict[str, Any] = {
    "name": "glob_files",
    "description": """Find files matching a glob pattern.
Returns matching file paths sorted by modification time (most recent first).
Patterns not starting with '**/' are automatically prepended with '**/' for recursive search.
Examples: '*.py', '*.{ts,tsx}', 'test_*.py', '**/components/**/*.tsx'""",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match files (e.g. '*.py', '*.{ts,tsx}')",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (defaults to working directory)",
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of files to return (default: 100)",
            },
        },
        "required": ["pattern"],
    },
}

# Lint tool
LINT_SPEC: dict[str, Any] = {
    "name": "lint",
    "description": """Run linter on specified files and return diagnostics.
Auto-detects the linter (ruff for Python, eslint for JS/TS) or you can specify one.
Use after editing files to check for errors you may have introduced.""",
    "parameters": {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of file paths to lint",
            },
            "linter": {
                "type": "string",
                "description": "Linter to use (auto-detected if not specified). Options: ruff, eslint, flake8",
            },
        },
        "required": ["files"],
    },
}

# Ask user tool — pauses the agent loop and requests user input
ASK_USER_SPEC: dict[str, Any] = {
    "name": "ask_user",
    "description": """Ask the user a clarifying question before proceeding.
Use this when the task is ambiguous and you need user input to make the right decision.
The agent loop will pause until the user responds via Telegram.
Only use this for genuinely ambiguous situations — do not ask unnecessary questions.""",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of predefined choices for the user (renders as buttons)",
            },
        },
        "required": ["question"],
    },
}


# ---------------------------------------------------------------------------
# Lazy TOOL_SPECS — built on first access to avoid circular imports
# (subagent.py and web_search.py would pull in core/llm modules at import
# time, creating a registry -> specs -> subagent -> core -> executor ->
# registry cycle).
# ---------------------------------------------------------------------------

_TOOL_SPECS: dict[str, dict[str, Any]] | None = None


def _build_tool_specs() -> dict[str, dict[str, Any]]:
    """Build the full tool specs dict, importing external specs lazily."""
    from src.tools.subagent import SUBAGENT_SPEC, COMPARISON_SPEC
    from src.tools.web_search import WEB_SEARCH_SPEC

    return {
        "shell_command": SHELL_COMMAND_SPEC,
        "read_file": READ_FILE_SPEC,
        "write_file": WRITE_FILE_SPEC,
        "list_dir": LIST_DIR_SPEC,
        "grep_files": GREP_FILES_SPEC,
        "apply_patch": APPLY_PATCH_SPEC,
        "view_image": VIEW_IMAGE_SPEC,
        "update_plan": UPDATE_PLAN_SPEC,
        "web_search": WEB_SEARCH_SPEC,
        "spawn_subagent": SUBAGENT_SPEC,
        "spawn_comparison": COMPARISON_SPEC,
        "str_replace": STR_REPLACE_SPEC,
        "hashline_edit": HASHLINE_EDIT_SPEC,
        "glob_files": GLOB_FILES_SPEC,
        "lint": LINT_SPEC,
        "ask_user": ASK_USER_SPEC,
    }


def _get_tool_specs() -> dict[str, dict[str, Any]]:
    """Return the (lazily-initialized) tool specs dict."""
    global _TOOL_SPECS
    if _TOOL_SPECS is None:
        _TOOL_SPECS = _build_tool_specs()
    return _TOOL_SPECS


# Keep a public name for backward compatibility — but callers should prefer
# get_all_tools() / get_tool_spec() which go through the lazy accessor.
TOOL_SPECS: dict[str, dict[str, Any]] = {}  # populated on first get_all_tools() call


def get_all_tools() -> list[dict[str, Any]]:
    """Get all tool specifications as a list.

    Returns:
        List of tool specification dicts
    """
    return list(_get_tool_specs().values())


def get_tool_spec(name: str) -> dict[str, Any] | None:
    """Get a specific tool specification.

    Args:
        name: Name of the tool

    Returns:
        Tool specification dict or None if not found
    """
    return _get_tool_specs().get(name)


MUTATING_TOOLS: set[str] = {
    "write_file",
    "apply_patch",
    "str_replace",
    "hashline_edit",
    "shell_command",
}

READ_ONLY_TOOLS: set[str] = {
    "read_file",
    "list_dir",
    "grep_files",
    "glob_files",
    "view_image",
    "web_search",
    "lint",
    "update_plan",
    "spawn_subagent",
    "spawn_comparison",
    "ask_user",
}


def tool_is_mutating(name: str, arguments: dict[str, Any] | None = None) -> bool:
    """Return True if the tool can mutate workspace/system state."""
    if name in MUTATING_TOOLS:
        return True
    if name == "spawn_subagent":
        subagent_type = (arguments or {}).get("type", "explore")
        return subagent_type == "execute"
    return False


PARALLEL_SAFE_TOOLS: set[str] = {
    "read_file",
    "list_dir",
    "grep_files",
    "glob_files",
    "view_image",
    "web_search",
    "lint",
    "update_plan",
    "spawn_subagent",
}


def tool_supports_parallel(name: str) -> bool:
    """Return True when a tool can safely run in parallel."""
    return name in PARALLEL_SAFE_TOOLS


def _matches_type(value: Any, schema_type: str) -> bool:
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "object":
        return isinstance(value, dict)
    return True


def _validate_schema(value: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    schema_type = schema.get("type")
    if schema_type and not _matches_type(value, schema_type):
        errors.append(f"{path} must be of type {schema_type}")
        return

    enum_values = schema.get("enum")
    if enum_values is not None and value not in enum_values:
        errors.append(f"{path} must be one of {enum_values}")
        return

    if schema_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(value, dict):
            return
        for key in required:
            if key not in value:
                errors.append(f"{path}.{key} is required")
        for key, key_value in value.items():
            key_schema = properties.get(key)
            if key_schema is None:
                continue
            _validate_schema(key_value, key_schema, f"{path}.{key}", errors)
    elif schema_type == "array":
        if not isinstance(value, list):
            return
        item_schema = schema.get("items")
        if not item_schema:
            return
        for idx, item in enumerate(value):
            _validate_schema(item, item_schema, f"{path}[{idx}]", errors)


def validate_tool_arguments(name: str, arguments: dict[str, Any]) -> list[str]:
    """Validate arguments against tool schema; returns a list of errors."""
    spec = get_tool_spec(name)
    if not spec:
        return [f"Unknown tool: {name}"]
    params = spec.get("parameters", {})
    errors: list[str] = []
    _validate_schema(arguments, params, name, errors)
    return errors
