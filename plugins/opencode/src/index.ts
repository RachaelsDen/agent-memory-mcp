import type { Hooks, Plugin, PluginInput } from "@opencode-ai/plugin";
import {
  MARKER,
  buildInjection,
  buildNamespaceDirective,
  resolveNamespace,
} from "./helpers";

export interface PluginOptions {
  disableNamespaceDirective?: boolean;
  disableNamespace?: boolean;
}

function isPluginInput(obj: unknown): obj is PluginInput {
  return (
    typeof obj === "object" &&
    obj !== null &&
    ("directory" in obj || "client" in obj || "$" in obj)
  );
}

export type PluginFactory = (options?: PluginOptions) => Plugin;

const plugin = ((
  input: PluginInput | PluginOptions,
  options?: PluginOptions
) => {
  if (!isPluginInput(input)) {
    const factoryOpts = input as PluginOptions;
    const configuredPlugin: Plugin = (realInput, realOpts) =>
      plugin(realInput, { ...factoryOpts, ...realOpts }) as Promise<Hooks>;
    return configuredPlugin as unknown as Promise<Hooks>;
  }

  const disableNamespace = Boolean(
    options?.disableNamespaceDirective ?? options?.disableNamespace
  );

  const hooks: Hooks = {
    "experimental.chat.system.transform": async (_in, out) => {
      let system = out.system;
      if (!Array.isArray(system)) {
        system = [];
        out.system = system;
      }

      const hasDiscipline = system.some(
        (entry) =>
          (entry ?? "").includes(MARKER) ||
          (entry ?? "").includes("<!-- agent-memory -->")
      );

      const ns =
        disableNamespace || process.env.MEMORY_NAMESPACE
          ? undefined
          : resolveNamespace(input.directory ?? process.cwd());

      if (hasDiscipline) {
        if (ns) {
          const hasNamespace = system.some((entry) =>
            (entry ?? "").includes("memory_set_namespace")
          );
          if (!hasNamespace) {
            system.push(buildNamespaceDirective(ns));
          }
        }
        return;
      }

      system.push(buildInjection(ns));
    },
  };

  return Promise.resolve(hooks);
}) as Plugin & PluginFactory;

export default plugin;
