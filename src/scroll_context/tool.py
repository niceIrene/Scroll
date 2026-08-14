"""The ``scroll_repl`` tool surface (OpenAI function-calling schema).

Scroll's REPL tool is named ``execute_python`` in `scroll_agent_A`; here it is
``scroll_repl`` so harnesses that already expose a task-side Python tool (e.g.
LOCA-bench's ``python_execute`` MCP server) don't end up with two
near-identically named Python surfaces.
"""

SCROLL_REPL_TOOL_NAME = "scroll_repl"

SCROLL_REPL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": SCROLL_REPL_TOOL_NAME,
        "description": (
            "Your memory REPL: runs Python in a persistent runner-side "
            "namespace — your surface for retrieving and reasoning over your "
            "own conversation memory, NOT for task-environment work (use the "
            "task's own tools for that). Variables, imports, and defs persist "
            "across calls; only what you print() is returned. Inside it, `ms` "
            "is a read-only window onto hist.conversation_history "
            "(ms.search(...), ms.expand(...), ms.sql_query(...)) for recalling "
            "turns that have been evicted from this prompt. Keep full data in "
            "variables, print only the slice you need."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Python source to execute in the persistent namespace.",
                }
            },
            "required": ["source"],
        },
    },
}
