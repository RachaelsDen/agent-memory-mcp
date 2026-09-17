"""Credential screen shared by every write path (DESIGN §11; Issue #9).

The fixed pattern list and the field-naming rejection live here so every
entry point applies the SAME screen: capture, the reason fields of
dispute / demote / promote / corroborate / contradict, write_lesson's
evidence reasons, and the digest renderer's defense-in-depth pass over
its own output. The error names the FIELD only; the matched content is
never echoed back.
"""

import re

try:  # mcp>=2 renamed FastMCP to MCPServer; keep both import spellings working
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # fallback targets mcp 1.x, which is not installed here
    from mcp.server.fastmcp.exceptions import ToolError  # pyright: ignore[reportMissingImports]

SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"eyJhbGciOi[A-Za-z0-9._-]{20,}"),
)


def contains_secret(text: str) -> bool:
    """True when any fixed credential pattern matches the text."""
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def screen_secrets(**fields: str | list[str] | None) -> None:
    """Reject any text field (or tag entry) carrying a credential pattern."""
    for name, value in fields.items():
        texts: list[str] = [value] if isinstance(value, str) else (value or [])
        if any(contains_secret(text) for text in texts):
            raise ToolError(
                f"field {name!r} appears to contain a secret; refusing to store it; redact and retry"
            )


def screen_document(text: str) -> None:
    """Defense in depth for rendered artifacts: reject the exact text about
    to hit disk (digest content) without echoing what matched."""
    if contains_secret(text):
        raise ToolError(
            "field 'digest content' appears to contain a secret; refusing to "
            "write it; redact and retry"
        )
