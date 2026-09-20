# agent-memory opencode plugin

An [opencode](https://opencode.ai) plugin for the [agent-memory](../..) MCP server. It injects the
memory discipline block into the system prompt and tells the agent which namespace to use, so the
memory tools get exercised without hand-editing instruction files per project.

## What it does

On every chat, via the `experimental.chat.system.transform` hook, the plugin appends one block to
the system prompt:

1. **Memory discipline** — the recall / capture / report / consolidate rules for the agent-memory
   tools (`memory_probe`, `memory_capture_episode`, `memory_report_usage`,
   `memory_consolidate_scan` → `memory_write_lesson`). The block starts with the `[agent-memory]`
   marker and injection is idempotent, so it is never duplicated. The plugin recognizes both the `[agent-memory]` marker and the `<!-- agent-memory -->` HTML comment from the paste-in block.
2. **Namespace directive** — instructs the agent to call
   `memory_set_namespace("opencode@<project>-<hash>")` now, before any other work (omitted if
   `MEMORY_NAMESPACE` is set in the environment).

The namespace is `opencode@<project-directory-name>-<hash>`: the basename of the working directory
opencode runs in, sanitized to the namespace character set `[a-z0-9._-]` (lowercased, disallowed
characters replaced with `-`, leading/trailing `-` and `.` stripped; an empty result falls back to
`local`), appended with a 6-hex-char SHA-256 hash of the raw directory basename for
cross-host convergence (same repo at different paths on different hosts converges to the same
memory scope). Every project directory automatically gets its own memory scope with zero per-project
configuration. If `MEMORY_NAMESPACE` is already set in the environment, the namespace directive line is
omitted to preserve the explicit scope.

The plugin only injects text. It makes no MCP calls, performs no file I/O, and sets no
environment variables. It expects the agent-memory MCP server to be configured separately (see
the repository README, "Host configuration").

## Install

Requires [Bun](https://bun.sh):

```bash
cd plugins/opencode
bun install
bun run build
```

`dist/` is gitignored — build it after cloning.

## Wire into opencode

Add the built module to the `plugin` list in your `opencode.json` (repository root) or
`~/.config/opencode/opencode.json`, pointing at the dist output:

```json
{
  "plugin": [
    "/absolute/path/to/agent-memory/plugins/opencode/dist/index.js"
  ]
}
```

Alternatively, while developing, drop a local loader into `.opencode/plugins/` in the project
where you run opencode:

```ts
import plugin from "/absolute/path/to/agent-memory/plugins/opencode/dist/index.js";

export const AgentMemory = plugin;
```

Restart opencode after changing plugin config; use `opencode debug config` to inspect the
resolved configuration.

## When NOT to use the namespace directive

If you set `MEMORY_NAMESPACE` in the MCP server's `environment` config (in `opencode.json`), the plugin cannot read the MCP server's subprocess environment because the plugin executes in the opencode parent process. As an architectural limitation, `process.env.MEMORY_NAMESPACE` in the plugin will be empty, and its automatic namespace directive would override your explicit server configuration.

To prevent this, disable the namespace directive using `disableNamespace: true` (or `disableNamespaceDirective: true`) in a local plugin loader in `.opencode/plugins/`:

```ts
import plugin from "/absolute/path/to/agent-memory/plugins/opencode/dist/index.js";

export const AgentMemory = plugin({ disableNamespace: true });
```

When disabled, the plugin injects the memory discipline block into the system prompt without appending the `memory_set_namespace` directive.

## Tests

```bash
cd plugins/opencode && bun test
```
