<!--
  Agent Memory Discipline Block
  Insert the block below into AGENTS.md, CLAUDE.md, .cursorrules, or system prompt.
-->

<!-- agent-memory -->
## Memory Discipline

1. **Recall before acting**: Call `memory_probe` with intent before non-trivial tasks. Respect confidence scores, evidence, and `disputed` flags.
2. **Capture at surprise**: Call `memory_capture_episode` on prediction errors when outcomes deviate from expectations (goal/expectation/action/outcome/surprise). Do not log routine successes.
3. **Report what helped**: Call `memory_report_usage` with verdicts (`used`, `helped`, `harmed`, `ignored`) to drive the trust flywheel.
4. **Consolidate patterns**: Run `memory_consolidate_scan` and write lessons via `memory_write_lesson` with evidence when patterns repeat. Single incidents are not lessons.

**Do NOT**:
- Probe trivially.
- Capture routine successes.
- Write lessons from single isolated incidents.
- Store secrets or credentials.
<!-- agent-memory -->
