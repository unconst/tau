"""Web search tool for SuperAgent.

Uses ``curl`` + a public search API or falls back to a simple ``curl``-based
approach so that no extra Python dependencies are needed.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Dict

from src.tools.base import ToolResult

# Tool specification
WEB_SEARCH_SPEC: Dict[str, Any] = {
    "name": "web_search",
    "description": (
        "Search the web for current information. Returns a summary of the "
        "top results.  Use when you need up-to-date facts, documentation, "
        "or information not available in local files."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
        },
        "required": ["query"],
    },
}


def web_search(args: Dict[str, Any]) -> ToolResult:
    """Execute a web search via shell ``curl`` against DuckDuckGo Lite."""
    query = args.get("query", "").strip()
    if not query:
        return ToolResult.fail("No query provided")

    # Use DuckDuckGo's HTML Lite endpoint via curl, then extract text lines.
    # This avoids any Python dependency on requests/httpx for the search itself.
    try:
        curl_cmd = [
            "curl",
            "-sS",
            "-L",
            "--max-time", "15",
            "-A", "Mozilla/5.0 (compatible; TauAgent/1.0)",
            "-d", f"q={query}",
            "https://lite.duckduckgo.com/lite/",
        ]

        result = subprocess.run(
            curl_cmd,
            capture_output=True,
            text=True,
            timeout=20,
        )

        if result.returncode != 0:
            return ToolResult.fail(f"curl failed: {result.stderr.strip()}")

        html = result.stdout

        # Rough extraction: pull text from <a> and <td> tags
        import re
        # Extract result links
        links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*class="result-link"[^>]*>([^<]+)</a>', html)
        # Extract snippet text from td class="result-snippet"
        snippets = re.findall(r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>', html, re.DOTALL)

        lines: list[str] = []
        for i, ((url, title), snippet) in enumerate(
            zip(links[:5], snippets[:5]), start=1
        ):
            clean_snippet = re.sub(r"<[^>]+>", "", snippet).strip()
            lines.append(f"{i}. {title.strip()}\n   {url}\n   {clean_snippet}\n")

        if not lines:
            # Fallback: strip all HTML and return the first 3000 chars
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                return ToolResult.ok(f"Raw results for '{query}':\n{text[:3000]}")
            return ToolResult.ok(f"No results found for '{query}'.")

        return ToolResult.ok(f"Search results for '{query}':\n\n" + "\n".join(lines))

    except subprocess.TimeoutExpired:
        return ToolResult.fail("Web search timed out")
    except Exception as e:
        return ToolResult.fail(f"Web search error: {e}")
