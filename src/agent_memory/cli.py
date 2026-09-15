"""Console entrypoint: dispatches subcommands; default runs the MCP server."""

import argparse
import importlib
import sys
from collections.abc import Callable

from agent_memory import db
from agent_memory.config import get_settings


def _cmd_migrate() -> None:
    settings = get_settings()
    for name in db.migrate(settings.PGVECTOR_DIM):
        print(f"applied: {name}")


def _cmd_serve() -> None:
    # Server module arrives in task 6; until then this import fails and the
    # CLI surfaces a one-line error instead of a traceback.
    server = importlib.import_module("agent_memory.server")
    serve: Callable[[], None] = server.serve
    serve()


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent-memory", description="Agent memory MCP server")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("migrate", help="apply pending database migrations").set_defaults(
        handler=_cmd_migrate
    )
    # Later tasks add: digest, stats, consolidate-scan.
    handler: Callable[[], None] | None = parser.parse_args().handler
    try:
        if handler is None:
            _cmd_serve()
        else:
            handler()
    except Exception as exc:  # CLI boundary: concise message, nonzero exit, no traceback
        print(f"agent-memory: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
