# Agent Memory MCP

A persistent-memory MCP server for coding agents: episode capture with secret
screening, hybrid keyword+vector retrieval with spreading activation,
exposure/outcome feedback, agent-in-the-loop consolidation with
diversity-priced confidence, manual cross-namespace promotion with tombstone
demotion, and a markdown digest audit layer. 13 MCP tools over stdio, backed
by Postgres + pgvector. The full design, rationale, and scope lines live in
[DESIGN.md](DESIGN.md).

## The 13 tools

| Tool | Purpose |
| --- | --- |
| `memory_capture_episode` | Record one episode at the moment of surprise |
| `memory_probe` | Recall memories relevant to what the agent is about to do |
| `memory_search` | Explicit recall ("what do we know about X") |
| `memory_report_usage` | Report verdicts on records a retrieval showed |
| `memory_consolidate_scan` | Scan for consolidation candidate clusters (read-only) |
| `memory_write_lesson` | Store a drafted lesson with its evidence edges |
| `memory_corroborate` | Record an episode as further support for a lesson |
| `memory_contradict` | Record an episode as contradicting a lesson |
| `memory_promote` | Graduate a lesson to a broader namespace (manual) |
| `memory_demote` | Retire a promoted copy with a tombstone (never a delete) |
| `memory_dispute` | Flag a lesson as wrong (human audit layer) |
| `memory_digest` | Render the namespace audit digest to `DIGEST_DIR` |
| `memory_stats` | Namespace health check / curiosity pass |

Every tool takes a trailing optional `namespace` parameter (see
[Namespaces](#namespaces) below).

## Quickstart

```bash
git clone https://github.com/RachaelsDen/agent-memory-mcp
cd agent-memory-mcp
docker compose up -d --wait          # Postgres 16 + pgvector on :55432
uv sync                              # install into .venv
uv run agent-memory migrate          # apply pending migrations (prints applied: 001_init.sql)
```

Then point an MCP host at the server (next section). The server checks for
pending migrations again on every startup, so upgrading just means dropping
new migration files and restarting.

## Host configuration

All snippets run the installed `agent-memory` console script through `uv`. Replace
`/path/to/agent-memory-mcp` in the snippets below with the absolute path of your clone.
Namespaces are exact scopes (`agent@this-project` convention; no prefix hierarchy). The
intended setup is a per-project configuration pinning that project's namespace; a global
config that hardcodes one project's namespace scopes every session across all repositories to it.

### Claude Desktop

`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/agent-memory-mcp",
        "agent-memory"
      ],
      "env": {
        "DATABASE_URL": "postgresql://agent_memory:agent_memory@localhost:55432/agent_memory",
        "MEMORY_NAMESPACE": "claude@this-project"
      }
    }
  }
}
```

Because Claude Desktop has a single global configuration, to work across multiple
projects duplicate the block with distinct server keys (e.g., `agent-memory-projA`)
and namespaces, or omit `MEMORY_NAMESPACE` (defaulting to `default@local`) and pass
the `namespace` tool parameter per call.

### opencode

Repository-root `opencode.json` (project-level configuration overriding the user-level config):

```json
{
  "mcp": {
    "agent-memory": {
      "type": "local",
      "command": [
        "uv", "run",
        "--directory", "/path/to/agent-memory-mcp",
        "agent-memory"
      ],
      "environment": {
        "DATABASE_URL": "postgresql://agent_memory:agent_memory@localhost:55432/agent_memory",
        "MEMORY_NAMESPACE": "opencode@this-project"
      },
      "enabled": true
    }
  }
}
```

In a user-level global config (`~/.config/opencode/opencode.json`), omit `MEMORY_NAMESPACE`
rather than pinning a project—but note that omitting `MEMORY_NAMESPACE` globally requires a
per-repo override in every project you use. Projects without an override will share `default@local`
(treat that as a consciously-accepted shared scope, or pin per-repo configs instead).

### Generic MCP client

Any stdio MCP client: spawn the command below with the environment you want.

```bash
uv run --directory /path/to/agent-memory-mcp agent-memory
```

With no arguments the process serves MCP over stdio; tool errors come back as
MCP error results, never process exits.

## Namespaces

Memory is scoped by namespace (`agent@project` by convention; `global` holds
promoted lessons every namespace can see). Namespaces are exact-match strings
with no prefix hierarchy (`opencode` and `opencode@proj` are disjoint scopes).
Lessons move between namespaces only through promotion; `global` is the one
namespace every probe sees automatically, while promoted copies in other
namespaces are visible when you query those namespaces directly. Evidence edges
are agent-directed and may cite episodes from any namespace—the citing lesson's
provenance then carries those episodes' excerpts wherever it is retrieved.
The effective namespace resolves as:

```
MEMORY_NAMESPACE (env)  <  --namespace (CLI flag)  <  namespace (per-tool param)
```

The global `--namespace` flag overrides `MEMORY_NAMESPACE` for the whole
process, so `agent-memory --namespace me@proj stats` reports on that
namespace even if the env var says another, and
`python -m agent_memory --namespace me@proj` serves tools whose default
namespace is `me@proj`. A per-tool `namespace` argument always wins over both.

## CLI reference

```
agent-memory [--namespace NAMESPACE] {migrate,digest,stats,consolidate-scan}
agent-memory                    # no subcommand -> MCP server over stdio
agent-memory migrate            # apply pending migrations; prints each applied file
agent-memory digest             # render the audit digest; prints the file path
agent-memory stats              # namespace health stats as JSON on stdout
agent-memory consolidate-scan   # consolidation scan clusters as JSON on stdout
```

`python -m agent_memory` is the same dispatcher as the `agent-memory` console
script.

## Cron

Digest weekly, scan daily (the scan is read-only; redirecting to /dev/null
keeps it a pure health pass over the DB):

```cron
# Mondays 09:00 — render the audit digest for the default namespace
0 9 * * 1  cd /path/to/agent-memory-mcp && uv run agent-memory digest >> ~/.agent-memory/digest-cron.log 2>&1

# Daily 03:15 — consolidation scan (cron entrypoint per DESIGN §8/§12)
15 3 * * *  cd /path/to/agent-memory-mcp && uv run agent-memory consolidate-scan > /dev/null 2>&1
```

Add `--namespace me@this-project` before the subcommand to target a specific
namespace.

## Environment variables

Defaults below are the compiled-in values; every one can be overridden by env
or a pydantic-settings source.

### Connection and identity

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql://agent_memory:agent_memory@localhost:55432/agent_memory` | Postgres connection string |
| `MEMORY_NAMESPACE` | `default@local` | Default namespace for captures and queries |
| `DIGEST_DIR` | `~/.agent-memory/digest` | Where digest markdown files are written (`~` expanded at use) |

### Embeddings

| Variable | Default | Meaning |
| --- | --- | --- |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `EMBED_DEVICE` | `cpu` | torch device for the embedder |
| `EMBED_IMPL` | `local` | `local` or `fake` (deterministic hash embedder for tests) |
| `PGVECTOR_DIM` | `384` | Vector column dimension (must match the model; tests use 8) |
| `FAKE_EMBED_OVERRIDES` | `""` | JSON `{text: [vector]}` map consumed by the fake embedder |

### Retrieval scoring

| Variable | Default | Meaning |
| --- | --- | --- |
| `W_REL` | `0.45` | Weight: relevance |
| `W_SAL` | `0.20` | Weight: salience (surprise) |
| `W_ENV` | `0.15` | Weight: environmental freshness |
| `W_USE` | `0.10` | Weight: usefulness |
| `W_SPREAD` | `0.10` | Weight: spreading activation |
| `TAU_ENV_H` | `4320` | Freshness decay constant, hours |
| `TAU_USE_H` | `720` | Usefulness decay constant, hours |
| `SIM_FLOOR` | `0.25` | Minimum similarity to return |
| `TS_RANK_SAT` | `0.1` | Keyword rank saturation |
| `STALE_ENV_FRESH` | `0.2` | env_fresh below this flags staleness (with salience) |
| `SALIENCE_STALE` | `0.7` | salience above this flags staleness (with env_fresh) |
| `PROBE_TOPK` | `12` | Candidates fetched per channel before fusion |
| `FINAL_K` | `8` | Final results returned (also the `k=None` default) |

### Consolidation

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLUSTER_COS` | `0.82` | Cosine above which episodes join a scan cluster |
| `DEDUP_COS` | `0.95` | Near-duplicate collapse threshold for seeding |
| `DEDUP_WINDOW_H` | `24` | Collapse window for incident dedup |
| `SIMILAR_LINK_COS` | `0.75` | Cosine above which lessons get `similar` links |
| `DUP_CLAIM_COS` | `0.95` | Duplicate-claim guard threshold |
| `CONSOLIDATE_MIN_AGE_H` | `1` | Episodes younger than this are not scan-eligible |

## Design

Contract details, schema, and scope decisions: [DESIGN.md](DESIGN.md).
