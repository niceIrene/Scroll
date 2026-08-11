"""Terminal-Bench-specific assets (prompts) for agents run on that dataset.

Terminal-Bench runs through Harbor (sandboxed), not a native runner, so this
package currently just houses the system prompt that frames an agent for the
container/bash task environment. A config selects it via ``system_prompt:
terminal_bench``, which the harness injects through ``ctx.system_prompt``.
"""
