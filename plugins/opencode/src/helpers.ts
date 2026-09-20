import crypto from "node:crypto";
import path from "node:path";

export const MARKER = "[agent-memory]";

const MEMORY_DISCIPLINE = `${MARKER}
## Memory Discipline

1. **Recall before acting**: Call \`memory_probe\` with intent before non-trivial tasks. Respect confidence scores and \`disputed\` flags.
2. **Capture at surprise**: Call \`memory_capture_episode\` on prediction errors. Do not log routine successes.
3. **Report what helped**: Call \`memory_report_usage\` with verdicts to drive the trust flywheel.
4. **Consolidate patterns**: When patterns repeat across sessions, \`memory_consolidate_scan\` → \`memory_write_lesson\` with evidence. Single incidents are not lessons.

Do NOT: probe trivially, capture routine successes, write lessons from single incidents, store secrets.
`;

/**
 * Resolve the agent-memory namespace for a working directory:
 * `opencode@<basename>-<hash>`. The basename is sanitized exactly like the
 * server's `_sanitize_agent_name` (charset [a-z0-9._-], lowercased, each RUN of
 * disallowed chars collapsed to one "-", leading/trailing "-" and "."
 * stripped, truncated to at most 64 chars, falling back to `local` if empty),
 * and appended with a 6-hex-char SHA-256 hash derived from the raw, unsanitized
 * directory basename.
 *
 * Tradeoff: We hash the raw basename (not the full filesystem path) to
 * guarantee cross-host convergence (same repo at different paths on different
 * hosts produces the identical namespace). If two distinct projects share the
 * same directory basename, they will resolve to the same namespace; users can
 * disambiguate them by setting an explicit `MEMORY_NAMESPACE` environment
 * variable or configuring `disableNamespace: true`.
 */
export function resolveNamespace(cwd: string = process.cwd()): string {
  const normalized = path.resolve(cwd);
  const rawBasename = path.basename(normalized);
  const hash = crypto
    .createHash("sha256")
    .update(rawBasename)
    .digest("hex")
    .slice(0, 6);
  let scope = rawBasename
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^[.-]+|[.-]+$/g, "");
  if (scope.length > 64) {
    scope = scope.slice(0, 64).replace(/[.-]+$/g, "");
  }
  return `opencode@${scope || "local"}-${hash}`;
}

export function buildInjection(namespace?: string): string {
  if (!namespace) {
    return MEMORY_DISCIPLINE;
  }
  return `${MEMORY_DISCIPLINE}
Your memory namespace is \`${namespace}\`.
Call \`memory_set_namespace("${namespace}")\` now, before any other work.
`;
}
