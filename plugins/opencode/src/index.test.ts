import { describe, expect, test } from "bun:test";
import type { PluginInput } from "@opencode-ai/plugin";

import plugin from "./index";
import { MARKER, buildInjection, resolveNamespace } from "./helpers";

function makePluginInput(): PluginInput {
  return {
    client: null as unknown as PluginInput["client"],
    project: null as unknown as PluginInput["project"],
    directory: process.cwd(),
    worktree: process.cwd(),
    serverUrl: new URL("http://localhost"),
    $: null as unknown as PluginInput["$"],
    experimental_workspace: null as unknown as PluginInput["experimental_workspace"],
  };
}

type Transform = (
  input: { model: unknown },
  output: { system?: string[] },
) => Promise<void>;

async function getTransform(): Promise<Transform> {
  const hooks = await plugin(makePluginInput());
  const transform = (hooks as Record<string, unknown>)[
    "experimental.chat.system.transform"
  ];
  if (typeof transform !== "function") {
    throw new Error("expected experimental.chat.system.transform hook");
  }
  return transform as Transform;
}

describe("resolveNamespace", () => {
  test("keeps alphanumeric, dash, dot, and underscore basenames with a 6-hex hash suffix", () => {
    expect(resolveNamespace("/home/alice/agent-memory")).toMatch(/^opencode@agent-memory-[0-9a-f]{6}$/);
    expect(resolveNamespace("/srv/data/analytics_2024.v2")).toMatch(/^opencode@analytics_2024.v2-[0-9a-f]{6}$/);
  });

  test("lowercases mixed-case basenames", () => {
    expect(resolveNamespace("/home/alice/MyProject")).toMatch(/^opencode@myproject-[0-9a-f]{6}$/);
    expect(resolveNamespace("/home/alice/CamelCaseApp")).toMatch(/^opencode@camelcaseapp-[0-9a-f]{6}$/);
  });

  test("replaces disallowed characters with a dash and strips edge dashes", () => {
    expect(resolveNamespace("/home/alice/My Project!")).toMatch(/^opencode@my-project-[0-9a-f]{6}$/);
    expect(resolveNamespace("/srv/Café Core")).toMatch(/^opencode@caf-core-[0-9a-f]{6}$/);
    expect(resolveNamespace("/srv/-dashboard-")).toMatch(/^opencode@dashboard-[0-9a-f]{6}$/);
  });

  test("collapses runs of disallowed characters into a single dash", () => {
    expect(resolveNamespace("/srv/My  Fancy  App")).toMatch(/^opencode@my-fancy-app-[0-9a-f]{6}$/);
    expect(resolveNamespace("/srv/a&&b")).toMatch(/^opencode@a-b-[0-9a-f]{6}$/);
  });

  test("produces different hashes for basenames that sanitize to the same scope", () => {
    const nsA = resolveNamespace("/path1/foo bar");
    const nsB = resolveNamespace("/path2/foo@bar");
    expect(nsA).toMatch(/^opencode@foo-bar-[0-9a-f]{6}$/);
    expect(nsB).toMatch(/^opencode@foo-bar-[0-9a-f]{6}$/);
    expect(nsA).not.toBe(nsB);
  });

  test("strips leading and trailing dots but keeps interior dots", () => {
    expect(resolveNamespace("/home/alice/.secrets")).toMatch(/^opencode@secrets-[0-9a-f]{6}$/);
    expect(resolveNamespace("/srv/v1.2.3")).toMatch(/^opencode@v1.2.3-[0-9a-f]{6}$/);
  });

  test("falls back to local when sanitization empties the basename", () => {
    expect(resolveNamespace("/srv/...")).toMatch(/^opencode@local-[0-9a-f]{6}$/);
  });

  test("produces the same namespace for distinct paths with the same basename (cross-host stability tradeoff)", () => {
    const nsA = resolveNamespace("/work/customer-a/api");
    const nsB = resolveNamespace("/work/customer-b/api");
    expect(nsA).toBe(nsB);
    expect(nsA).toMatch(/^opencode@api-[0-9a-f]{6}$/);
  });

  test("guarantees cross-host stability: same directory basename at different paths produces identical namespace", () => {
    const hostA = resolveNamespace("/home/alice/projects/agent-memory");
    const hostB = resolveNamespace("/mnt/ci/workspace/agent-memory");
    expect(hostA).toBe(hostB);
    expect(hostA).toMatch(/^opencode@agent-memory-[0-9a-f]{6}$/);
  });

  test("truncates long basenames to 64 chars before hash suffix, producing a component <= 71 chars", () => {
    const longName = "a".repeat(200);
    const ns = resolveNamespace(`/work/${longName}`);
    const scopeHashPart = ns.replace(/^opencode@/, "");
    expect(scopeHashPart.length).toBeLessThanOrEqual(71);
    expect(scopeHashPart).toMatch(/^a{64}-[0-9a-f]{6}$/);
  });

  test("strips trailing dashes/dots introduced by 64-char truncation", () => {
    const longNameWithDashes = "a-".repeat(40);
    const ns = resolveNamespace(`/work/${longNameWithDashes}`);
    const scopeHashPart = ns.replace(/^opencode@/, "");
    expect(scopeHashPart.length).toBeLessThanOrEqual(71);
    expect(scopeHashPart).toMatch(/^a(?:-a)+-[0-9a-f]{6}$/);
  });

  test("produces different hashes for long basenames sharing the first 64+ chars", () => {
    const prefix = "a".repeat(70);
    const nameA = prefix + "foo-10000000000000000000000000000";
    const nameB = prefix + "bar-20000000000000000000000000000";
    const nsA = resolveNamespace(`/work/${nameA}`);
    const nsB = resolveNamespace(`/work/${nameB}`);
    const scopeA = nsA.replace(/^opencode@/, "").replace(/-[0-9a-f]{6}$/, "");
    const scopeB = nsB.replace(/^opencode@/, "").replace(/-[0-9a-f]{6}$/, "");
    expect(scopeA).toBe(scopeB);
    expect(nsA).not.toBe(nsB);
  });
});

describe("buildInjection", () => {
  test("contains the discipline block and the resolved namespace directive when given an explicit namespace", () => {
    const injection = buildInjection("opencode@demo-project");
    expect(injection.startsWith(MARKER)).toBe(true);
    expect(injection).toContain("## Memory Discipline");
    expect(injection).toContain("memory_probe");
    expect(injection).toContain("memory_capture_episode");
    expect(injection).toContain("memory_report_usage");
    expect(injection).toContain("memory_consolidate_scan");
    expect(injection).toContain("memory_write_lesson");
    expect(injection).toContain("Do NOT");
    expect(injection).toContain("Your memory namespace is `opencode@demo-project`");
    expect(injection).toContain('memory_set_namespace("opencode@demo-project")');
  });

  test("omits namespace directive when MEMORY_NAMESPACE is set in process.env", () => {
    const orig = process.env.MEMORY_NAMESPACE;
    try {
      process.env.MEMORY_NAMESPACE = "explicit-env-ns";
      const injection = buildInjection();
      expect(injection).toContain(MARKER);
      expect(injection).toContain("## Memory Discipline");
      expect(injection).not.toContain("memory_set_namespace");
      expect(injection).not.toContain("Your memory namespace is");
    } finally {
      if (orig !== undefined) {
        process.env.MEMORY_NAMESPACE = orig;
      } else {
        delete process.env.MEMORY_NAMESPACE;
      }
    }
  });
});

describe("experimental.chat.system.transform", () => {
  test("appends the injection once to an existing system prompt", async () => {
    const transform = await getTransform();
    const out: { system?: string[] } = { system: ["base prompt"] };

    await transform({ model: "unknown" }, out);

    expect(out.system?.length).toBe(2);
    expect(out.system?.[0]).toBe("base prompt");
    expect(out.system?.[1]).toBe(buildInjection(resolveNamespace()));
  });

  test("is a no-op when the marker is already present", async () => {
    const transform = await getTransform();
    const existing = buildInjection("opencode@already-injected");
    const out: { system?: string[] } = { system: ["base prompt", existing] };

    await transform({ model: "unknown" }, out);

    expect(out.system?.length).toBe(2);
    expect(out.system?.[1]).toBe(existing);
  });

  test("does not append discipline block when <!-- agent-memory --> is present, but injects namespace directive if absent", async () => {
    const transform = await getTransform();
    const pasteInBlock = "<!-- agent-memory -->\n## Memory Discipline\n...";
    const out: { system?: string[] } = { system: [pasteInBlock] };

    await transform({ model: "unknown" }, out);

    expect(out.system?.length).toBe(2);
    expect(out.system?.[0]).toBe(pasteInBlock);
    expect(out.system?.[1]).not.toContain(MARKER);
    expect(out.system?.[1]).not.toContain("## Memory Discipline");
    expect(out.system?.[1]).toContain("Your memory namespace is");
    expect(out.system?.[1]).toContain("memory_set_namespace");
  });

  test("is a no-op when <!-- agent-memory --> and namespace directive are already present", async () => {
    const transform = await getTransform();
    const pasteInBlock = "<!-- agent-memory -->\n## Memory Discipline\n...";
    const nsDirective = 'Your memory namespace is `opencode@test`.\nCall `memory_set_namespace("opencode@test")` now.';
    const out: { system?: string[] } = { system: [pasteInBlock, nsDirective] };

    await transform({ model: "unknown" }, out);

    expect(out.system?.length).toBe(2);
    expect(out.system?.[0]).toBe(pasteInBlock);
    expect(out.system?.[1]).toBe(nsDirective);
  });

  test("creates the system array when it is missing", async () => {
    const transform = await getTransform();
    const out: { system?: string[] } = {};

    await transform({ model: "unknown" }, out);

    expect(Array.isArray(out.system)).toBe(true);
    expect(out.system?.length).toBe(1);
    expect(out.system?.[0]).toContain(MARKER);
  });

  test("omits set_namespace directive when MEMORY_NAMESPACE is set in process.env", async () => {
    const orig = process.env.MEMORY_NAMESPACE;
    try {
      process.env.MEMORY_NAMESPACE = "opencode@custom-env";
      const transform = await getTransform();
      const out: { system?: string[] } = {};

      await transform({ model: "unknown" }, out);

      expect(out.system?.length).toBe(1);
      expect(out.system?.[0]).toContain(MARKER);
      expect(out.system?.[0]).not.toContain("memory_set_namespace");
    } finally {
      if (orig !== undefined) {
        process.env.MEMORY_NAMESPACE = orig;
      } else {
        delete process.env.MEMORY_NAMESPACE;
      }
    }
  });

  test("omits set_namespace directive when disableNamespaceDirective is true in options", async () => {
    const hooks = await plugin(makePluginInput(), { disableNamespaceDirective: true });
    const transform = (hooks as Record<string, unknown>)[
      "experimental.chat.system.transform"
    ] as Transform;
    const out: { system?: string[] } = {};

    await transform({ model: "unknown" }, out);

    expect(out.system?.length).toBe(1);
    expect(out.system?.[0]).toContain(MARKER);
    expect(out.system?.[0]).not.toContain("memory_set_namespace");
    expect(out.system?.[0]).not.toContain("Your memory namespace is");
  });

  test("omits set_namespace directive when disableNamespace is true in options", async () => {
    const hooks = await plugin(makePluginInput(), { disableNamespace: true });
    const transform = (hooks as Record<string, unknown>)[
      "experimental.chat.system.transform"
    ] as Transform;
    const out: { system?: string[] } = {};

    await transform({ model: "unknown" }, out);

    expect(out.system?.length).toBe(1);
    expect(out.system?.[0]).toContain(MARKER);
    expect(out.system?.[0]).not.toContain("memory_set_namespace");
  });

  test("supports factory invocation plugin({ disableNamespaceDirective: true })(input)", async () => {
    const configuredPlugin = plugin({ disableNamespaceDirective: true });
    const hooks = await configuredPlugin(makePluginInput());
    const transform = (hooks as Record<string, unknown>)[
      "experimental.chat.system.transform"
    ] as Transform;
    const out: { system?: string[] } = {};

    await transform({ model: "unknown" }, out);

    expect(out.system?.length).toBe(1);
    expect(out.system?.[0]).toContain(MARKER);
    expect(out.system?.[0]).not.toContain("memory_set_namespace");
  });
});
