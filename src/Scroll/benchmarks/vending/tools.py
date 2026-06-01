"""Vending environment-specific tool functions.

These functions interact with ``VendingEnv`` and
``DataSourceManager``. Under the RLM-style substrate they are
exposed to the agent's REPL via
:func:`Scroll.benchmarks.vending.agents.agent._make_env_namespace` —
each closure is unwrapped from ``ToolResponse`` to a plain string so
``print(read_email())`` works in a code cell.
"""

from __future__ import annotations

import json
import re

from Scroll.core._tool_state import ToolState, _resp


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def send_email(state: ToolState, to: str, subject: str, body: str):
    state._tick()
    result = state.data.send_email(
        to=to, subject=subject, body=body,
        turn_idx=state.env.turn_idx, env=state.env,
    )
    state._action_log.append(f"send_email to={to} subject={subject}")
    return _resp(result)


def read_email(state: ToolState):
    state._tick()
    emails = state.data.read_emails(limit=1)
    if not emails:
        return _resp("No unread emails.")
    state._action_log.append("read_email")
    return _resp(emails[0])


def read_email_inbox(state: ToolState):
    state._tick()
    emails = state.data.read_emails(limit=20)
    if not emails:
        return _resp("Inbox is empty.")
    state._action_log.append(f"read_email_inbox count={len(emails)}")
    return _resp("\n---\n".join(emails))


def ai_web_search(state: ToolState, query: str):
    state._tick()
    results = state.data.search(query, top_k=5)
    state._action_log.append(f"ai_web_search query={query!r}")
    if not results:
        return _resp("No results found.")
    return _resp("\n".join(results))


def get_money_balance(state: ToolState):
    state._tick()
    env = state.env
    info = {
        "cash": round(env.cash, 2),
        "machine_cash": round(env.machine_cash, 2),
        "inventory_value": round(env.inventory_value(), 2),
        "net_worth": round(env.net_worth(), 2),
    }
    state._action_log.append("get_money_balance")
    return _resp(json.dumps(info, indent=2))


def sub_agent_specs(state: ToolState):
    state._tick()
    specs = (
        "Sub-agent manages physical vending machine operations.\n"
        "Available commands (combine with semicolons):\n"
        "  restock [sku=qty ...] - Move products from storage to machine. "
        "Without sku=qty, restocks all SKUs with default quantity.\n"
        "  set_price sku=price [...] - Set selling prices.\n"
        "  collect_cash - Collect cash from the machine.\n"
        "  check_inventory - View machine and storage inventory.\n"
        "  list_storage - List products in storage.\n"
        "\n"
        "Example: 'restock cola=20 water=15; collect_cash; check_inventory'"
    )
    return _resp(specs)


def run_sub_agent(state: ToolState, instruction: str):
    state._tick()
    env = state.env
    results: list[str] = []

    segments = re.split(r"[;\n]+", instruction)
    for seg in segments:
        seg = seg.strip().lower()
        if not seg:
            continue

        if "restock" in seg:
            pairs = re.findall(r"(\w+)\s*=\s*(\d+)", seg)
            if pairs:
                for sku, qty in pairs:
                    if sku in ("restock", "units", "qty"):
                        continue
                    result = env.restock_sku(sku, int(qty))
                    results.append(f"restock {sku}: {result}")
            else:
                result = env.restock(state.cfg.restock_units_per_sku)
                results.append(result)

        elif "set_price" in seg or "price" in seg:
            pairs = re.findall(r"(\w+)\s*=\s*([\d.]+)", seg)
            for sku, price in pairs:
                if sku in ("set_price", "price"):
                    continue
                if sku in env.catalog:
                    result = env.set_price(sku, float(price))
                    results.append(result)
                else:
                    results.append(f"set_price {sku}: unknown product")

        elif "collect" in seg and "cash" in seg:
            result = env.collect_cash()
            results.append(result)

        elif "inventory" in seg or "check" in seg:
            inv = {
                "machine": dict(env.machine),
                "storage": dict(env.storage),
                "prices": dict(env.prices),
            }
            results.append(f"inventory: {json.dumps(inv)}")

        elif "list_storage" in seg or "storage" in seg:
            results.append(f"storage: {json.dumps(dict(env.storage))}")

        else:
            results.append(f"unknown command: {seg}")

    summary = " | ".join(results) if results else "no actions taken"
    state._sub_agent_log.append(summary)
    state._action_log.append(f"run_sub_agent: {summary}")
    return _resp(summary)


def chat_with_sub_agent(state: ToolState, question: str):
    state._tick()
    if not state._sub_agent_log:
        return _resp("Sub-agent has not performed any actions yet.")
    recent = state._sub_agent_log[-5:]
    return _resp(
        f"Sub-agent recent actions:\n" + "\n".join(f"  - {a}" for a in recent)
    )


# ---------------------------------------------------------------------------
# Closure factories + registration
# ---------------------------------------------------------------------------


def _make_env_tool_closures(state: ToolState) -> list:
    """Create closure-based env-specific tool functions."""

    def send_email_tool(to: str, subject: str, body: str):
        """Send an email to a supplier or other address.

        Args:
            to: The recipient email address.
            subject: The email subject line.
            body: The email body text.
        """
        return send_email(state, to, subject, body)

    def read_email_tool():
        """Read the next unread email from your inbox.

        Returns:
            str: The next unread email as a single text blob (sender,
                subject, body all in one string), or "No unread emails."
                if the inbox is empty. Not JSON, not a dict — call
                .splitlines() / regex on the string if you need fields.
        """
        return read_email(state)

    def read_email_inbox_tool():
        """List up to 20 unread emails from your inbox.

        Returns:
            str: A single string containing every unread email (up to
                20), with each email separated by a literal "\\n---\\n"
                divider. NOT a list[dict] — do NOT call .get(...) on
                its elements; iterating gives you individual characters.
                To split into per-email blobs, do
                ``inbox.split("\\n---\\n")``. Returns "Inbox is empty."
                if there are no unread emails.
        """
        return read_email_inbox(state)

    def ai_web_search_tool(query: str):
        """Search the internet for market information, supplier directories, or product research.

        Args:
            query: The search query string.
        """
        return ai_web_search(state, query)

    def get_money_balance_tool():
        """Check your current financial status.

        Returns:
            str: A pretty-printed JSON string with keys
                ``cash``, ``machine_cash``, ``inventory_value``,
                ``net_worth`` (all floats). Parse with ``json.loads(...)``
                to get a dict — the function itself returns text, not
                a dict.
        """
        return get_money_balance(state)

    def sub_agent_specs_tool():
        """Get information about the sub-agent and its available tools for physical vending machine operations."""
        return sub_agent_specs(state)

    def run_sub_agent_tool(instruction: str):
        """Send an instruction to the sub-agent to perform physical vending machine operations.

        Args:
            instruction: Instruction for the sub-agent. Combine commands with semicolons. Example: 'restock cola=20 water=15; collect_cash; check_inventory'
        """
        return run_sub_agent(state, instruction)

    def chat_with_sub_agent_tool(question: str):
        """Ask the sub-agent a question about what it has done.

        Args:
            question: Your question for the sub-agent.
        """
        return chat_with_sub_agent(state, question)

    return [
        send_email_tool,
        read_email_tool,
        read_email_inbox_tool,
        ai_web_search_tool,
        get_money_balance_tool,
        sub_agent_specs_tool,
        run_sub_agent_tool,
        chat_with_sub_agent_tool,
    ]


ENV_TOOL_NAMES = {
    "send_email_tool", "read_email_tool", "read_email_inbox_tool",
    "ai_web_search_tool", "get_money_balance_tool", "sub_agent_specs_tool",
    "run_sub_agent_tool", "chat_with_sub_agent_tool",
}


def register_env_tools(toolkit, state: ToolState) -> None:
    """No-op kept for compatibility with ``core/_registry.py``.

    Under the RLM substrate, env tools are surfaced as REPL globals
    by :func:`Scroll.benchmarks.vending.agents.agent._make_env_namespace`
    — there's no AgentScope ``Toolkit`` to register on. The registry
    still calls this hook by name; left as a no-op so the lookup
    succeeds.
    """
    return None
