# Agent Memory MCP

A persistent-memory MCP server for coding agents: episode capture with secret
screening, hybrid keyword+vector retrieval with spreading activation,
exposure/outcome feedback, agent-in-the-loop consolidation with
diversity-priced confidence, manual cross-namespace promotion with tombstone
demotion, and a markdown digest audit layer. 14 MCP tools over stdio, backed
by Postgres + pgvector. The full design, rationale, and scope lines live in
[DESIGN.md](DESIGN.md).

The installed tool is pinned at install time — pull/rebuild doesn't update it; upgrade via `uv tool install --force` at the new tag.

## The 14 tools

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
| `memory_set_namespace` | Set this session's default namespace (Issue #4) |

Every tool takes a trailing optional `namespace` parameter (see
[Namespaces](#namespaces) below).

## Quickstart

```bash
# Clone for docker-compose.yml (the installed tool runs independently of the checkout)
git clone https://github.com/RachaelsDen/agent-memory-mcp
cd agent-memory-mcp
git checkout v1.1.0

# Install (pinned at release; upgrades are deliberate)
uv tool install --force .
uv tool update-shell                 # ensure the tool bin dir is on PATH (reload shell after)

# Database
docker compose up -d --wait          # Postgres 16 + pgvector on :55432
agent-memory migrate                 # applies migrations 001-004 (from the installed package)
```

Then point an MCP host at the server (next section). The server checks for
pending migrations again on every startup, so upgrading means checking out the new tag (`git fetch --tags && git checkout <new-tag>` before reinstalling via `uv tool install --force .`) and restarting — the server applies any new migrations automatically on startup.

### Development

For working on the server itself, clone the repository and run via `uv`:

```bash
git clone https://github.com/RachaelsDen/agent-memory-mcp
cd agent-memory-mcp
uv sync                              # install into .venv
uv run agent-memory migrate          # apply pending migrations
uv run agent-memory                  # run server from checkout
```

The installed tool is pinned and unaffected by repository changes.

## Host configuration

The installed binary `agent-memory` is the command (or the absolute path `/Users/alice/.local/bin/agent-memory` — get it via `uv tool dir --bin` or `which agent-memory` — if PATH isn't inherited by desktop apps).
With no namespace configuration, the server derives an agent namespace from the MCP client's
`clientInfo` name, such as `claude-desktop@local`. Namespaces are exact scopes
(`agent@this-project` convention; no prefix hierarchy). Set an override only when you want to pin
a project or deliberately share another scope.

### Claude Desktop

`claude_desktop_config.json` (use `/Users/alice/.local/bin/agent-memory` if PATH isn't inherited):

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "agent-memory",
      "env": {
        "DATABASE_URL": "postgresql://agent_memory:agent_memory@localhost:55432/agent_memory"
      }
    }
  }
}
```

The namespace is auto-derived from the client identity, for example
`claude-desktop@local`. Set `MEMORY_NAMESPACE` only to override it globally, or call
`memory_set_namespace("agent@this-project")` at session start for project scoping.

### opencode

Repository-root `opencode.json`, shown here as a deliberate per-project namespace override of the
user-level config:

```json
{
  "mcp": {
    "agent-memory": {
      "type": "local",
      "command": [
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

In a user-level global config (`~/.config/opencode/opencode.json`), omit `MEMORY_NAMESPACE` and let
the server derive `opencode@local`. Projects without an override share that client-derived scope;
pin a repository with `MEMORY_NAMESPACE` or `memory_set_namespace` when project isolation matters.

### Generic MCP client

Any stdio MCP client: spawn the command below with the environment you want.

```bash
agent-memory
```

With no arguments the process serves MCP over stdio; tool errors come back as
MCP error results, never process exits.

### Agent instructions

For your agent to actually USE the memory tools, add the discipline block from [examples/AGENTS-memory.md](examples/AGENTS-memory.md) to your agent's instruction file (AGENTS.md, CLAUDE.md, .cursorrules, or system prompt). Without it, the tools are available but the agent may not think to use them.

## Namespaces

Memory starts with a zero-config agent namespace derived from the MCP client's `clientInfo` name,
such as `opencode@local`; use an override when you need project pinning (`agent@project` by
convention). `global` holds promoted lessons every namespace can see. Namespaces are exact-match
strings with no prefix hierarchy (`opencode` and `opencode@proj` are disjoint scopes).
Lessons move between namespaces only through promotion; `global` is the one
namespace every probe sees automatically, while promoted copies in other
namespaces are visible when you query those namespaces directly. Evidence edges
are agent-directed and may cite episodes from any namespace—the citing lesson's
provenance then carries those episodes' excerpts wherever it is retrieved.
The effective namespace resolves as:

```
MEMORY_NAMESPACE (env)  <  --namespace (CLI flag)  <  memory_set_namespace (session)  <  namespace (per-tool param)
```

The global `--namespace` flag overrides `MEMORY_NAMESPACE` for the whole
process, so `agent-memory --namespace me@proj stats` reports on that
namespace even if the env var says another, and
`python -m agent_memory --namespace me@proj` serves tools whose default
namespace is `me@proj`. A per-tool `namespace` argument always wins over both.

`memory_set_namespace` (called once at session start) sets a session-scoped
default that outranks env and the flag but is itself outranked by any
per-tool param; the override lives only in the server process (one stdio
server = one client session) and dies with it. It rejects `global` as a
session default — global lessons are created only by `memory_promote` — and
rejects empty or whitespace-only values. When `MEMORY_NAMESPACE` is unset
at startup, the agent half of the default namespace is derived from the
initialize handshake's clientInfo name (sanitized; e.g. a client naming
itself `Claude Desktop` defaults to `claude-desktop@local`), falling back
to `default@local` when no clientInfo is available. clientInfo applies only
when `MEMORY_NAMESPACE` is left at its default — any explicit configuration
(env var, `--namespace` CLI flag, or other settings source) wins.

For agentic hosts, run one global server config with no `MEMORY_NAMESPACE`; the client-derived
agent namespace works without setup. To pin memory per project, have the agent call
`memory_set_namespace("me@this-project")` at the start of each session, for example from one line
in `AGENTS.md` or `CLAUDE.md`. Per-project host configs provide the same explicit override for
non-agentic clients.

## CLI reference

```
agent-memory [--namespace NAMESPACE] {migrate,digest,stats,consolidate-scan}
agent-memory                    # no subcommand -> MCP server over stdio
agent-memory migrate            # apply pending migrations; prints each applied file
agent-memory digest             # render the audit digest; prints the file path
agent-memory digest --all-namespaces   # one digest per existing namespace; "<ns>\t<path>\t<flagged>" lines
agent-memory stats              # namespace health stats as JSON on stdout
agent-memory consolidate-scan   # consolidation scan clusters as JSON on stdout
agent-memory consolidate-scan --all-namespaces   # {"namespaces": [per-namespace scan payloads]}
```

`python -m agent_memory` is the same dispatcher as the `agent-memory` console
script.

In `digest --all-namespaces` output, every TSV field (namespace, path, flagged
count) is backslash-escaped — `\` as `\\`, tab as `\t`, newline as `\n`,
carriage return as `\r` — so one namespace always prints exactly one 3-field
line. On errors, the middle field carries the literal `ERROR` and the third
field carries the error message (`<ns>\tERROR\t<message>`). Unescape each field
in a single left-to-right pass when consuming the output (chained `str.replace`
calls corrupt sequences like `\\t`, the escaped form of a literal backslash
before a `t`).

## Cron

Digest weekly, scan daily (the scan is read-only; redirecting to /dev/null
keeps it a pure health pass over the DB):

```cron
# Set PATH so cron finds binaries installed in /home/you/.local/bin
PATH=/home/you/.local/bin:/usr/bin:/bin

# Mondays 09:00 — render the audit digest for every namespace that exists
0 9 * * 1  agent-memory digest --all-namespaces >> $HOME/.agent-memory/digest-cron.log 2>&1

# Daily 03:15 — consolidation scan over every namespace (cron entrypoint per DESIGN §8/§12)
15 3 * * *  agent-memory consolidate-scan --all-namespaces > /dev/null 2>&1
```

`--namespace me@this-project` before the subcommand targets ONE specific
namespace; `--all-namespaces` on `digest` or `consolidate-scan` covers every
namespace that exists (promoted-copy `global` included, even while it holds
no rows; an empty database yields no digests and `{"namespaces": []}`). Error lines
in `digest --all-namespaces` parse identically as 3 tab-separated fields; consumers
detect errors when field 2 is `ERROR`. The two flags are mutually exclusive.

## Environment variables

Defaults below are the compiled-in values; every one can be overridden by env
or a pydantic-settings source.

### Connection and identity

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql://agent_memory:agent_memory@localhost:55432/agent_memory` | Postgres connection string |
| `MEMORY_NAMESPACE` | `default@local` | Default namespace for captures and queries; when unset, derived from the client's clientInfo name (`<client>@local`) |
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
