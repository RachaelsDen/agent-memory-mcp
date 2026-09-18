"""Session-scoped namespace state and THE namespace resolution (Issue #4).

One stdio server process serves exactly one client session, so module-level
state IS session state: it dies with the process, nothing persists to the
DB, and no schema is involved. The state is (a) a session override set via
the memory_set_namespace tool and (b) the client name observed on the
initialize handshake's clientInfo, which derives the agent half of the
default namespace when no Settings source explicitly set MEMORY_NAMESPACE.

Every pipeline resolves through resolve_namespace — precedence:

    tool param  >  session-set  >  MEMORY_NAMESPACE explicitly configured
    via any Settings source (env var — the --namespace CLI flag writes env
    — or direct Settings construction)  >  derived default

Explicitness is detected with pydantic v2's ``model_fields_set``, so every
source pydantic-settings honors counts; only the compiled-in default
(``default@local``) falls through to the derived tier. The derived default
is ``<sanitized clientInfo name>@local`` with a ``default@local`` fallback
when no usable name was observed (CLI subcommands, clients that skip
clientInfo). ``global`` stays legal as an explicit per-tool param and as
memory_promote's target, never as a session default — that mirrors
write_lesson's promotion-only guard.
"""

import re

try:  # mcp>=2 renamed FastMCP to MCPServer; keep both import spellings working
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # fallback targets mcp 1.x, which is not installed here
    from mcp.server.fastmcp.exceptions import ToolError  # pyright: ignore[reportMissingImports]

from agent_memory.config import Settings

_session_namespace: str | None = None
_client_name: str | None = None

_UNNAMESPACED = re.compile(r"[^a-z0-9._-]+")


def _sanitize_agent_name(name: str) -> str:
    """clientInfo name -> namespace-safe agent token; '' when unusable.

    'Claude Desktop' -> 'claude-desktop'. '@' is excluded so a client name
    can never smuggle a second namespace separator into the derived default.
    """
    return _UNNAMESPACED.sub("-", name.strip().lower()).strip("-.")


def note_client_name(name: str) -> None:
    """Record the initialize handshake's clientInfo name; first name wins."""
    global _client_name
    sanitized = _sanitize_agent_name(name)
    if sanitized and _client_name is None:
        _client_name = sanitized


def set_session_namespace(namespace: str) -> str:
    """Validate and store the session-scoped default; returns it verbatim."""
    global _session_namespace
    if not namespace.strip():
        raise ToolError(
            f"session namespace must be a non-empty string, got {namespace!r}"
        )
    if namespace == "global":
        raise ToolError(
            "namespace 'global' is promotion-only: global lessons are created "
            "by memory_promote (copy with provenance, DESIGN §11); a session "
            "default of 'global' would route every bare capture into a direct "
            "global write — set an agent@project namespace instead"
        )
    _session_namespace = namespace
    return namespace


def resolve_namespace(settings: Settings, param: str | None) -> str:
    """param > session-set > configured > derived default (module docstring)."""
    if param is not None:
        return param
    if _session_namespace is not None:
        return _session_namespace
    if "MEMORY_NAMESPACE" in settings.model_fields_set:
        return settings.MEMORY_NAMESPACE
    return f"{_client_name or 'default'}@local"
