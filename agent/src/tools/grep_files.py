"""Grep files tool for SuperAgent."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

from src.tools.base import BaseTool, ToolResult


class GrepFilesTool(BaseTool):
    """Tool for searching file contents using patterns."""

    name = "grep_files"
    description = "Finds files whose contents match the pattern."

    # Default limits
    DEFAULT_LIMIT = 100
    MAX_LIMIT = 2000
    TIMEOUT_SECONDS = 30

    def execute(
        self,
        pattern: str,
        include: Optional[str] = None,
        path: Optional[str] = None,
        limit: int = DEFAULT_LIMIT,
        context_lines: int = 2,
    ) -> ToolResult:
        """Search for files matching a pattern.

        Args:
            pattern: Regex pattern to search for
            include: Glob pattern to filter files
            path: Directory to search in
            limit: Maximum number of results
            context_lines: Number of surrounding context lines to show per match

        Returns:
            ToolResult with matching lines and context
        """
        # Resolve search path
        search_path = self.resolve_path(path) if path else self.cwd

        if not search_path.exists():
            return ToolResult.fail(f"Path not found: {search_path}")

        # Cap limit
        limit = min(limit, self.MAX_LIMIT)
        # Clamp context_lines to 0-5
        context_lines = max(0, min(context_lines, 5))

        # Try ripgrep first (fastest)
        rg_result = self._search_with_ripgrep(pattern, include, search_path, limit, context_lines)
        if rg_result is not None:
            return rg_result

        # Fallback to Python implementation
        return self._search_with_python(pattern, include, search_path, limit, context_lines)

    def _search_with_ripgrep(
        self,
        pattern: str,
        include: Optional[str],
        search_path: Path,
        limit: int,
        context_lines: int = 2,
    ) -> Optional[ToolResult]:
        """Search using ripgrep (rg).

        Returns None if ripgrep is not available.
        """
        cmd = ["rg", "-n", "--color=never", f"-C {context_lines}"]

        if include:
            # Convert glob to rg glob format
            cmd.extend(["--glob", include])

        cmd.extend([pattern, str(search_path)])
        # Add max-count to limit total matches
        cmd.extend(["--max-count", str(limit)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT_SECONDS,
            )

            if result.returncode == 0:
                # ripgrep output is already formatted as "filepath:linenumber:content"
                # Split into lines but keep only up to limit matches
                lines = [line for line in result.stdout.strip().split("\n") if line]
                lines = lines[:limit]

                if not lines:
                    return ToolResult.ok("No matches found.")

                output = "\n".join(lines)
                if len(lines) > limit:
                    output += f"\n\n[... {len(lines) - limit} more matches ...]"

                return ToolResult.ok(output)

            elif result.returncode == 1:
                # No matches
                return ToolResult.ok("No matches found.")

            elif result.returncode == 2:
                # Error - might be bad pattern
                return ToolResult.fail(f"Search error: {result.stderr.strip()}")

            return None  # Try fallback

        except FileNotFoundError:
            return None  # rg not installed, use fallback
        except subprocess.TimeoutExpired:
            return ToolResult.fail(f"Search timed out after {self.TIMEOUT_SECONDS}s")
        except Exception:
            return None  # Use fallback

    def _search_with_python(
        self,
        pattern: str,
        include: Optional[str],
        search_path: Path,
        limit: int,
        context_lines: int = 2,
    ) -> ToolResult:
        """Fallback Python implementation for searching."""
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return ToolResult.fail(f"Invalid regex pattern: {e}")

        matches: list[str] = []
        errors: list[str] = []
        match_count = 0

        # Convert include glob to regex if provided
        include_regex = None
        if include:
            # Simple glob to regex conversion
            include_pattern = include.replace(".", r"\.").replace("*", ".*").replace("?", ".")
            if "{" in include_pattern:
                # Handle {a,b,c} patterns
                include_pattern = re.sub(
                    r"\{([^}]+)\}",
                    lambda m: f"({'|'.join(m.group(1).split(','))})",
                    include_pattern,
                )
            try:
                include_regex = re.compile(f"^{include_pattern}$", re.IGNORECASE)
            except re.error:
                pass

        def should_include(file_path: Path) -> bool:
            if include_regex is None:
                return True
            return include_regex.match(file_path.name) is not None

        def search_file(file_path: Path) -> None:
            nonlocal match_count
            if match_count >= limit:
                return
            
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                lines = content.splitlines()
                
                for i, line in enumerate(lines):
                    if match_count >= limit:
                        return
                    
                    if regex.search(line):
                        # Get context lines
                        start_line = max(0, i - context_lines)
                        end_line = min(len(lines), i + context_lines + 1)
                        
                        # Add context lines
                        for j in range(start_line, end_line):
                            line_num = j + 1
                            line_content = lines[j]
                            prefix = "  " if j != i else "> "
                            matches.append(f"{file_path}:{line_num}:{prefix}{line_content}")
                        
                        # Add separator between matches
                        if j < end_line - 1:
                            matches.append("--")
                        
                        match_count += 1
                        
            except (PermissionError, OSError):
                pass

        def search_dir(dir_path: Path) -> None:
            if match_count >= limit:
                return

            try:
                for item in dir_path.iterdir():
                    if match_count >= limit:
                        return

                    if item.is_file() and should_include(item):
                        search_file(item)

                    elif item.is_dir() and not item.is_symlink():
                        # Skip hidden directories
                        if not item.name.startswith("."):
                            search_dir(item)

            except PermissionError:
                errors.append(f"Permission denied: {dir_path}")

        if search_path.is_file():
            search_file(search_path)
        else:
            search_dir(search_path)

        if not matches:
            return ToolResult.ok("No matches found.")

        output = "\n".join(matches)
        if match_count >= limit:
            output += f"\n\n[... reached limit of {limit} matches ...]"

        if errors:
            output += "\n\nWarnings:\n" + "\n".join(errors[:5])

        return ToolResult.ok(output, data={"count": match_count})
