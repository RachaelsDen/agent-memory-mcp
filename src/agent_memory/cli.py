"""Console entrypoint: dispatches subcommands; default runs the MCP server."""

import argparse
import importlib
import json
import os
import sys
from argparse import Namespace
from collections.abc import Callable

from agent_memory import db
from agent_memory.config import get_settings
from agent_memory.consolidation_tools import consolidate_scan
from agent_memory.digest import digest
from agent_memory.oversight import all_namespaces, stats


def _tsv_escape(field: str) -> str:
    """Backslash-escape TSV-breaking characters so one namespace prints one
    3-field line: namespaces and paths are free strings, so a raw tab would
    add a field and a raw newline would split the line. Consumers unescape
    \\\\, \\t, \\n, \\r (README, CLI reference)."""
    return (
        field.replace("\\", "\\\\")
        .replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def _cmd_migrate(args: Namespace) -> None:
    settings = get_settings()
    for name in db.migrate(settings.PGVECTOR_DIM):
        print(f"applied: {name}")


def _cmd_serve() -> None:
    # Lazy import keeps CLI-only subcommands from loading the MCP stack; the
    # module shares one path with `python -m agent_memory` and the console script.
    server = importlib.import_module("agent_memory.server")
    serve: Callable[[], None] = server.serve
    serve()


def _cmd_digest(args: Namespace) -> None:
    if not args.all_namespaces:
        print(digest(get_settings())["path"])
        return
    failed = False
    for namespace in all_namespaces():
        # One namespace's unwritable digest must not starve the others: report
        # the line, keep looping, and let the final nonzero exit carry the loss.
        try:
            payload = digest(get_settings(), namespace=namespace)
        except Exception as exc:
            message = f"ERROR: {type(exc).__name__}: {exc}"
            print(f"{_tsv_escape(namespace)}\t{_tsv_escape(message)}")
            failed = True
            continue
        print(
            f"{_tsv_escape(namespace)}\t{_tsv_escape(str(payload['path']))}"
            f"\t{_tsv_escape(str(payload['flagged_count']))}"
        )
    if failed:
        raise SystemExit(1)


def _cmd_stats(args: Namespace) -> None:
    print(json.dumps(stats(get_settings())))


def _cmd_consolidate_scan(args: Namespace) -> None:
    if not args.all_namespaces:
        print(
            json.dumps(
                consolidate_scan(
                    get_settings(), pool="fresh", min_cluster_size=2, namespace=None
                )
            )
        )
        return
    # Stream one payload per line instead of materializing every namespace's
    # payload first: memory stays bounded to a single namespace's scan even
    # when stdout is /dev/null, and the document stays one valid JSON doc.
    namespaces = all_namespaces()
    if not namespaces:
        print('{"namespaces": []}')
        return
    print('{"namespaces": [')
    last = len(namespaces) - 1
    for position, namespace in enumerate(namespaces):
        payload = consolidate_scan(
            get_settings(), pool="fresh", min_cluster_size=2, namespace=namespace
        )
        print(json.dumps(payload) + ("," if position < last else ""))
    print("]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agent-memory", description="Agent memory MCP server"
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="override MEMORY_NAMESPACE for this process "
        "(precedence: env < --namespace < session-set < per-tool param)",
    )
    sub = parser.add_subparsers(dest="command")
    # A no-subcommand parse leaves `handler` unset unless defaulted here.
    parser.set_defaults(handler=None)
    sub.add_parser("migrate", help="apply pending database migrations").set_defaults(
        handler=_cmd_migrate
    )
    digest_parser = sub.add_parser(
        "digest", help="render the namespace audit digest; prints the file path"
    )
    digest_parser.add_argument(
        "--all-namespaces",
        action="store_true",
        help="digest every existing namespace; one '<ns>\\t<path>\\t<flagged_count>' "
        "stdout line per namespace (exit nonzero if any fails); each field is "
        "backslash-escaped (\\\\, \\t, \\n, \\r) — unescape when consuming",
    )
    digest_parser.set_defaults(handler=_cmd_digest)
    sub.add_parser(
        "stats", help="print namespace health stats as JSON"
    ).set_defaults(handler=_cmd_stats)
    scan_parser = sub.add_parser(
        "consolidate-scan",
        help="print consolidation scan clusters as JSON (cron entrypoint)",
    )
    scan_parser.add_argument(
        "--all-namespaces",
        action="store_true",
        help="scan every existing namespace; prints one JSON object: "
        "'{\"namespaces\": [per-namespace scan payloads]}'",
    )
    scan_parser.set_defaults(handler=_cmd_consolidate_scan)
    args = parser.parse_args()
    # Subparsers without the flag do not carry the attribute; getattr keeps
    # the check safe for every subcommand.
    if args.namespace is not None and getattr(args, "all_namespaces", False):
        parser.error("--namespace and --all-namespaces are mutually exclusive")
    if args.namespace is not None:
        # Precedence env < --namespace < per-tool param: pin the flag into the
        # process env and drop the settings cache BEFORE any consumer reads it.
        os.environ["MEMORY_NAMESPACE"] = args.namespace
        get_settings.cache_clear()
    handler: Callable[[Namespace], None] | None = args.handler
    try:
        if handler is None:
            _cmd_serve()
        else:
            handler(args)
    except Exception as exc:  # CLI boundary: concise message, nonzero exit, no traceback
        print(f"agent-memory: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
