"""Shared constants and helpers for vending-environment agents.

Separated from __init__.py to avoid circular imports — agent modules
need these symbols, but __init__.py re-exports the agent classes.
"""

VENDING_CONTEXT = """\
You are an autonomous AI agent managing a vending machine business on a \
college campus.

Your performance will be evaluated based on your net worth (cash + machine \
cash + inventory value) at the end of the simulation. You must figure out \
how to run the business profitably on your own — no guidance will be provided.

ENVIRONMENT:
- Your vending machine has 12 unit slots across max 6 different SKUs.
- Products available in the market: cola, water, chips, choco, energy, \
gum, juice, nuts.
- Orders from suppliers take 3 days to arrive in your storage.
- Email replies arrive the next day. Same-day read_email_inbox will not show them.
- You must restock the machine from your storage.
- Daily operating fee: $2 (deducted automatically).
- If you cannot cover the daily fee for 10 consecutive days, you go bankrupt.
- Supplier prices are not published — you must discover them yourself.
- Cash from machine sales must be collected manually.
- When you are done for the day, call wait_for_next_day.

DATA-DRIVEN DECISIONS:
- Before restocking: review recent sales to prioritize high-demand items.
- Before ordering: check stock levels and estimate when you will run out.
- Before pricing: compare your costs with selling prices to optimize margins.
- Periodically: review your overall performance and adjust strategy.
- Use your data tools to look up facts — do not rely on memory alone.
"""

def vending_day_prompt(day: int, env=None, setup_prompt: str = "") -> str:
    """Shared day prompt for all vending agents.

    Args:
        day: Current simulation day.
        env: Unused, kept for API compatibility.
        setup_prompt: Optional override prompt for the first day.
    """
    if day == 1:
        if setup_prompt:
            return setup_prompt
        return (
            f"Day {day}. You have $500 starting cash and an empty machine. "
            "Good luck."
        )

    return f"Day {day} has started."
