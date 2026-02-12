"""System prompt management and templating.

This module provides a flexible system for building and rendering system prompts
with support for sections, variables, presets, and capability contexts.

Based on: cli/fabric-core/src/context/system_prompt.rs
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# =============================================================================
# Context Strings
# =============================================================================

CODE_EXECUTION_CONTEXT = """## Code Execution
You have access to execute shell commands and code. Use this capability responsibly:
- Prefer non-destructive operations when possible
- Make reasonable decisions and proceed autonomously without asking for confirmation
- Handle errors gracefully and retry with different approaches if needed"""

FILE_OPERATIONS_CONTEXT = """## File Operations
You can read, write, and modify files. Guidelines:
- Read files to understand context before making changes
- Make targeted edits rather than rewriting entire files
- Create backups when making significant changes
- Respect file permissions and ownership"""

WEB_SEARCH_CONTEXT = """## Web Search
You can search the web for information. Guidelines:
- Use specific, targeted searches
- Cite sources when providing information
- Verify information from multiple sources when possible
- Be clear about the recency of information"""

CODING_ASSISTANT_BASE = """You are an expert software engineer who helps users with coding tasks.

## Capabilities
- Write, review, and debug code
- Execute shell commands to test and verify changes
- Read and modify files in the project
- Search for patterns and understand codebases

## Guidelines
- Write clean, maintainable code
- Follow project conventions and style
- Explain your reasoning and approach
- Test changes when possible
- Be concise but thorough"""

CODE_REVIEWER_BASE = """Review code for:
- Correctness and bugs
- Performance issues
- Security vulnerabilities
- Code style and maintainability
- Test coverage

Provide specific, actionable feedback with examples."""


# =============================================================================
# Token Estimation
# =============================================================================


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses a simple heuristic based on character count.
    More accurate estimation would require a tokenizer.

    Args:
        text: Text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    # Simple heuristic: ~4 characters per token + 1
    return (len(text) // 4) + 1


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PromptSection:
    """A section of the system prompt.

    Attributes:
        name: Section name (used as header).
        content: Section content.
        enabled: Whether this section is enabled.
        priority: Priority (higher = earlier in prompt).
    """

    name: str
    content: str
    enabled: bool = True
    priority: int = 0

    def with_priority(self, priority: int) -> PromptSection:
        """Set priority and return self for chaining.

        Args:
            priority: Priority value (higher = earlier).

        Returns:
            Self for method chaining.
        """
        self.priority = priority
        return self

    def set_enabled(self, enabled: bool) -> PromptSection:
        """Set enabled state and return self for chaining.

        Args:
            enabled: Whether section is enabled.

        Returns:
            Self for method chaining.
        """
        self.enabled = enabled
        return self


@dataclass
class SystemPrompt:
    """System prompt configuration.

    Supports base prompts, sections, variables, capability contexts,
    custom instructions, and personas.

    Attributes:
        base: Base prompt text.
        sections: Sections to include.
        variables: Variables for templating.
        code_execution: Enable code execution context.
        file_operations: Enable file operation context.
        web_search: Enable web search context.
        custom_instructions: Custom instructions.
        persona: Persona/role.
    """

    base: Optional[str] = None
    sections: List[PromptSection] = field(default_factory=list)
    variables: Dict[str, str] = field(default_factory=dict)
    code_execution: bool = False
    file_operations: bool = False
    web_search: bool = False
    custom_instructions: Optional[str] = None
    persona: Optional[str] = None
    _token_count: int = 0

    @classmethod
    def new(cls) -> SystemPrompt:
        """Create a new system prompt.

        Returns:
            New SystemPrompt instance.
        """
        return cls()

    @classmethod
    def with_base(cls, base: str) -> SystemPrompt:
        """Create with base text.

        Args:
            base: Base prompt text.

        Returns:
            New SystemPrompt with base set.
        """
        prompt = cls(base=base)
        prompt._recalculate_tokens()
        return prompt

    def set_base(self, base: str) -> None:
        """Set base prompt.

        Args:
            base: Base prompt text.
        """
        self.base = base
        self._recalculate_tokens()

    def add_section(self, section: PromptSection) -> None:
        """Add a section.

        Args:
            section: Section to add.
        """
        self.sections.append(section)
        self._recalculate_tokens()

    def remove_section(self, name: str) -> None:
        """Remove a section by name.

        Args:
            name: Name of section to remove.
        """
        self.sections = [s for s in self.sections if s.name != name]
        self._recalculate_tokens()

    def set_variable(self, key: str, value: str) -> None:
        """Set a variable.

        Args:
            key: Variable name.
            value: Variable value.
        """
        self.variables[key] = value
        self._recalculate_tokens()

    def set_persona(self, persona: str) -> None:
        """Set persona.

        Args:
            persona: Persona/role description.
        """
        self.persona = persona
        self._recalculate_tokens()

    def set_custom_instructions(self, instructions: str) -> None:
        """Set custom instructions.

        Args:
            instructions: Custom instructions text.
        """
        self.custom_instructions = instructions
        self._recalculate_tokens()

    def enable_code_execution(self) -> None:
        """Enable code execution context."""
        self.code_execution = True
        self._recalculate_tokens()

    def enable_file_operations(self) -> None:
        """Enable file operations context."""
        self.file_operations = True
        self._recalculate_tokens()

    def enable_web_search(self) -> None:
        """Enable web search context."""
        self.web_search = True
        self._recalculate_tokens()

    def token_count(self) -> int:
        """Get token count estimate.

        Returns:
            Estimated token count.
        """
        return self._token_count

    def render(self) -> Optional[str]:
        """Render the full system prompt.

        Combines persona, base, sections (sorted by priority),
        capability contexts, and custom instructions.

        Returns:
            Rendered prompt string, or None if empty.
        """
        parts: List[str] = []

        # Persona
        if self.persona:
            parts.append(self.persona)

        # Base prompt
        if self.base:
            rendered = self._render_template(self.base)
            parts.append(rendered)

        # Sections (sorted by priority, higher first)
        sorted_sections = sorted(self.sections, key=lambda s: -s.priority)
        for section in sorted_sections:
            if section.enabled:
                content = self._render_template(section.content)
                if section.name:
                    parts.append(f"## {section.name}\n{content}")
                else:
                    parts.append(content)

        # Capability contexts
        if self.code_execution:
            parts.append(CODE_EXECUTION_CONTEXT)
        if self.file_operations:
            parts.append(FILE_OPERATIONS_CONTEXT)
        if self.web_search:
            parts.append(WEB_SEARCH_CONTEXT)

        # Custom instructions
        if self.custom_instructions:
            parts.append(f"## Custom Instructions\n{self.custom_instructions}")

        if not parts:
            return None

        return "\n\n".join(parts)

    def _render_template(self, template: str) -> str:
        """Render template with variables.

        Supports both {{key}} and ${key} syntax.

        Args:
            template: Template string.

        Returns:
            Rendered string with variables substituted.
        """
        result = template
        for key, value in self.variables.items():
            # Support {{key}} syntax
            result = result.replace(f"{{{{{key}}}}}", value)
            # Support ${key} syntax
            result = result.replace(f"${{{key}}}", value)
        return result

    def _recalculate_tokens(self) -> None:
        """Recalculate token count estimate."""
        rendered = self.render()
        if rendered:
            self._token_count = estimate_tokens(rendered)
        else:
            self._token_count = 0


# =============================================================================
# Builder Pattern
# =============================================================================


class SystemPromptBuilder:
    """Builder for system prompts.

    Provides a fluent interface for constructing SystemPrompt instances.

    Example:
        prompt = (SystemPromptBuilder()
            .persona("You are a helpful assistant.")
            .base("Help the user with their tasks.")
            .variable("name", "Alice")
            .code_execution()
            .build())
    """

    def __init__(self) -> None:
        """Create a new builder."""
        self._prompt = SystemPrompt()

    def base(self, base: str) -> SystemPromptBuilder:
        """Set base prompt.

        Args:
            base: Base prompt text.

        Returns:
            Self for method chaining.
        """
        self._prompt.base = base
        return self

    def persona(self, persona: str) -> SystemPromptBuilder:
        """Set persona.

        Args:
            persona: Persona/role description.

        Returns:
            Self for method chaining.
        """
        self._prompt.persona = persona
        return self

    def section(
        self, name: str, content: str, priority: int = 0, enabled: bool = True
    ) -> SystemPromptBuilder:
        """Add a section.

        Args:
            name: Section name (used as header).
            content: Section content.
            priority: Priority (higher = earlier in prompt).
            enabled: Whether section is enabled.

        Returns:
            Self for method chaining.
        """
        self._prompt.sections.append(
            PromptSection(name=name, content=content, priority=priority, enabled=enabled)
        )
        return self

    def variable(self, key: str, value: str) -> SystemPromptBuilder:
        """Add a variable.

        Args:
            key: Variable name.
            value: Variable value.

        Returns:
            Self for method chaining.
        """
        self._prompt.variables[key] = value
        return self

    def custom_instructions(self, instructions: str) -> SystemPromptBuilder:
        """Set custom instructions.

        Args:
            instructions: Custom instructions text.

        Returns:
            Self for method chaining.
        """
        self._prompt.custom_instructions = instructions
        return self

    def code_execution(self) -> SystemPromptBuilder:
        """Enable code execution context.

        Returns:
            Self for method chaining.
        """
        self._prompt.code_execution = True
        return self

    def file_operations(self) -> SystemPromptBuilder:
        """Enable file operations context.

        Returns:
            Self for method chaining.
        """
        self._prompt.file_operations = True
        return self

    def web_search(self) -> SystemPromptBuilder:
        """Enable web search context.

        Returns:
            Self for method chaining.
        """
        self._prompt.web_search = True
        return self

    def build(self) -> SystemPrompt:
        """Build the system prompt.

        Returns:
            Configured SystemPrompt instance.
        """
        self._prompt._recalculate_tokens()
        return self._prompt


# =============================================================================
# Presets
# =============================================================================


class Presets:
    """Predefined system prompts for common use cases."""

    @staticmethod
    def coding_assistant() -> SystemPrompt:
        """Default coding assistant prompt.

        Returns:
            SystemPrompt configured for coding assistance.
        """
        return (
            SystemPromptBuilder()
            .persona("You are Fabric, an expert AI coding assistant.")
            .base(CODING_ASSISTANT_BASE)
            .code_execution()
            .file_operations()
            .build()
        )

    @staticmethod
    def research_assistant() -> SystemPrompt:
        """Research assistant prompt.

        Returns:
            SystemPrompt configured for research assistance.
        """
        return (
            SystemPromptBuilder()
            .persona("You are a helpful research assistant with access to web search.")
            .base("Help the user find and analyze information. Cite sources when possible.")
            .web_search()
            .build()
        )

    @staticmethod
    def code_reviewer() -> SystemPrompt:
        """Code review prompt.

        Returns:
            SystemPrompt configured for code review.
        """
        return (
            SystemPromptBuilder()
            .persona("You are an expert code reviewer.")
            .base(CODE_REVIEWER_BASE)
            .file_operations()
            .build()
        )

    @staticmethod
    def minimal() -> SystemPrompt:
        """Minimal assistant prompt.

        Returns:
            SystemPrompt with minimal configuration.
        """
        return SystemPromptBuilder().base("You are a helpful assistant. Be concise.").build()


# =============================================================================
# Legacy API
# =============================================================================

# Legacy constant for backward compatibility
SYSTEM_PROMPT = """You are a coding agent running in SuperAgent, an autonomous terminal-based coding assistant.

You are precise, safe, and helpful. You run in fully autonomous mode — all commands execute without user approval.

# How you work

## Personality
Concise, direct, and friendly. Prioritize actionable guidance. Avoid verbose explanations unless asked.

## Task execution
Keep going until the task is completely resolved. Only yield when you are sure the problem is solved. Do NOT guess or make up answers.

Before tool calls, send a 1-sentence preamble describing what you're about to do.

## Parallel tool calls
Maximize throughput by issuing multiple tool calls in a single response whenever possible:
- When exploring: read several files, grep for multiple patterns, or list multiple directories at once
- When verifying: run lint, tests, and re-read edited files together
- Only read-only tools (`read_file`, `list_dir`, `grep_files`, `glob_files`, `web_search`, `view_image`, `lint`) can run in parallel
- Mutating tools (`write_file`, `str_replace`, `hashline_edit`, `apply_patch`, `shell_command`) must be issued one at a time
- Prefer batching 3-6 read-only calls per turn instead of one-at-a-time

GOOD — 1 turn with 2 parallel reads:
  read_file(file_path="src/foo.py", offset=100, limit=30)
  read_file(file_path="src/foo.py", offset=500, limit=30)

BAD — 2 separate turns for the same information:
  Turn 1: read_file(file_path="src/foo.py", offset=100, limit=30)
  Turn 2: read_file(file_path="src/foo.py", offset=500, limit=30)

Each turn costs a full LLM round-trip. Wasting turns on single reads when you could batch them is the #1 cause of slow task completion. ALWAYS batch independent reads.

## Planning discipline
Use `update_plan` to decompose tasks and track progress.
- One step `in_progress` at a time; keep descriptions short and actionable
- Always include a verification step (re-read output file, run syntax check, etc.)
- NEVER issue `update_plan` as the sole tool call in a turn — always pair it with a productive action (read, edit, grep, etc.)
- Batch multiple step transitions into one `update_plan` call instead of updating after every step
- Only update the plan when: (a) creating the initial plan, (b) the plan structure changes, or (c) finishing

GOOD — plan update paired with a read:
  update_plan(steps=[...])
  read_file(file_path="src/foo.py")

BAD — plan update alone wastes an entire turn:
  update_plan(steps=[...])
  (nothing else)

## Optimal workflow
Follow the read-then-edit pattern to minimize turns:
1. Read all relevant files in parallel — omit offset/limit for files under 500 lines to get the full file in one call. Narrow reads (30-50 lines) waste turns on re-reads when you need surrounding context later.
2. Plan your edits mentally based on what you read
3. Apply all edits in as few tool calls as possible (use batched operations). When making multiple edits to the same file, combine them into ONE hashline_edit call.
4. Verify once at the end — then STOP. Do not re-verify, re-read edited files, or create additional checks after a passing verification.

For a typical refactoring task, aim for 3-5 turns total, NOT 15-20.

## Coding guidelines
- `read_file` and `grep_files` return lines tagged as `line_number:hash|content`
- Prefer `hashline_edit` for surgical edits — reference the `line:hash` tags directly
  - Example: `{"op": "replace", "start": "5:a3", "end": "7:0e", "content": "new code"}`
  - If hashes don't match (file changed), re-read the file and retry
  - When making multiple edits to the same file, combine them into a SINGLE `hashline_edit` call with multiple operations in the `operations` array — do NOT issue separate calls for each edit
- Fall back to `str_replace` or `apply_patch` when hashline_edit is not a good fit
- Fix root causes, not symptoms; avoid unneeded complexity
- Keep changes minimal, consistent with existing codebase style
- Do not fix unrelated bugs or broken tests
- A successful edit means the text replacement was applied, NOT that the result is logically correct — always verify correctness (re-read the output file, run `python -c "import ast; ast.parse(open('file.py').read())"` for Python, etc.)
- Do not `git commit` unless explicitly requested
- Use `rg` (ripgrep) for searching — much faster than `grep`
- NEVER use destructive git commands (`reset --hard`, `checkout --`) unless requested
- Never revert changes you didn't make

## Editing constraints
- Default to ASCII; prefer `hashline_edit` for targeted edits, `apply_patch` for multi-file patches
- Dirty worktree: ignore unrelated changes, don't revert them

## Validation
- NEVER create standalone test/verification scripts. Use `shell_command` with inline Python instead:
  `python3 -c "from module import func; assert func(x) == y; print('OK')"`
  A test file costs 3-5 tool calls (write, run, debug, fix, re-run). An inline command costs 1.
- For syntax checks: `python3 -c "import ast; ast.parse(open('file.py').read())"`
- Verify ONCE. After a passing check, the task is done — do not re-read edited files, re-run checks, or create additional validation.
- If verification fails, fix it in ONE turn and re-verify. Do NOT spiral into multi-turn debugging — after 2 diagnostic commands without a fix, either apply your best fix or accept the result.
- Do not add tests to codebases with none

## Responses
- Be concise (aim for <10 lines). Reference file paths instead of dumping contents.
- Wrap commands, paths, and identifiers in backticks
- For code changes: lead with what changed and why, suggest next steps if natural
"""


def get_system_prompt(
    cwd: Optional[Path] = None,
    shell: Optional[str] = None,
    environment_context: Optional[Dict[str, object]] = None,
    model: Optional[str] = None,
) -> str:
    """Get the full system prompt with environment context.

    Uses the SYSTEM_PROMPT constant which includes autonomous behavior
    and mandatory verification plan instructions.

    Args:
        cwd: Current working directory.
        shell: Shell being used.
        environment_context: Optional execution policy/sandbox context.
        model: Model identifier being used for this session.

    Returns:
        Complete system prompt string.
    """
    # Use the SYSTEM_PROMPT constant directly (includes all autonomous behavior instructions)
    cwd_str = str(cwd) if cwd else "/app"
    shell_str = shell or "/bin/sh"

    # Add environment section
    env_lines = [
        f"- Working directory: {cwd_str}",
        f"- Platform: {platform.system()}",
        f"- Shell: {shell_str}",
    ]
    if model:
        env_lines.append(f"- Model: {model}")
    if environment_context:
        env_lines.extend(
            [
                f"- approval_policy: {environment_context.get('approval_policy', 'unknown')}",
                f"- sandbox_mode: {environment_context.get('sandbox_mode', 'unknown')}",
                f"- network_access: {environment_context.get('network_access', 'unknown')}",
                f"- readonly: {environment_context.get('readonly', False)}",
                f"- readable_roots: {environment_context.get('readable_roots', [])}",
                f"- writable_roots: {environment_context.get('writable_roots', [])}",
            ]
        )

    return f"{SYSTEM_PROMPT}\n\n# Environment\n" + "\n".join(env_lines)


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Core classes
    "PromptSection",
    "SystemPrompt",
    "SystemPromptBuilder",
    "Presets",
    # Context strings
    "CODE_EXECUTION_CONTEXT",
    "FILE_OPERATIONS_CONTEXT",
    "WEB_SEARCH_CONTEXT",
    "CODING_ASSISTANT_BASE",
    "CODE_REVIEWER_BASE",
    # Utilities
    "estimate_tokens",
    # Legacy API
    "SYSTEM_PROMPT",
    "get_system_prompt",
]
