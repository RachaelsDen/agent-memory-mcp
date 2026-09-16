"""Console entrypoint: dispatches subcommands; default runs the MCP server."""

import argparse
import importlib
import json
import os
import sys
from collections.abc import Callable

from agent_memory import db
from agent_memory.config import get_settings
from agent_memory.consolidation_tools import consolidate_scan
from agent_memory.digest import digest
from agent_memory.oversight import stats


def _cmd_migrate() -> None:
    settings = get_settings()
    for name in db.migrate(settings.PGVECTOR_DIM):
        print(f"applied: {name}")


def _cmd_serve() -> None:
    # Lazy import keeps CLI-only subcommands from loading the MCP stack; the
    # module shares one path with `python -m agent_memory` and the console script.
    server = importlib.import_module("agent_memory.server")
    serve: Callable[[], None] = server.serve
    serve()


def _cmd_digest() -> None:
    print(digest(get_settings())["path"])


def _cmd_stats() -> None:
    print(json.dumps(stats(get_settings())))


def _cmd_consolidate_scan() -> None:
    print(
        json.dumps(
            consolidate_scan(
                get_settings(), pool="fresh", min_cluster_size=2, namespace=None
            )
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agent-memory", description="Agent memory MCP server"
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="override MEMORY_NAMESPACE for this process "
        "(precedence: env < --namespace < per-tool param)",
    )
    sub = parser.add_subparsers(dest="command")
    # A no-subcommand parse leaves `handler` unset unless defaulted here.
    parser.set_defaults(handler=None)
    sub.add_parser("migrate", help="apply pending database migrations").set_defaults(
        handler=_cmd_migrate
    )
    sub.add_parser(
        "digest", help="render the namespace audit digest; prints the file path"
    ).set_defaults(handler=_cmd_digest)
    sub.add_parser(
        "stats", help="print namespace health stats as JSON"
    ).set_defaults(handler=_cmd_stats)
    sub.add_parser(
        "consolidate-scan",
        help="print consolidation scan clusters as JSON (cron entrypoint)",
    ).set_defaults(handler=_cmd_consolidate_scan)
    args = parser.parse_args()
    if args.namespace is not None:
        # Precedence env < --namespace < per-tool param: pin the flag into the
        # process env and drop the settings cache BEFORE any consumer reads it.
        os.environ["MEMORY_NAMESPACE"] = args.namespace
        get_settings.cache_clear()
    handler: Callable[[], None] | None = args.handler
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
