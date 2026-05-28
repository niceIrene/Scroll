"""Base environment and data source abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from Scroll.core._evaluation import EnvSnapshot, ProbeResult, ProbeSpec
    from Scroll.core._models import TurnResult


class BaseEnvironment(ABC):
    """Abstract simulation environment.

    The benchmark loop interacts with environments only through this
    interface, making it possible to swap in different scenarios
    (vending machine, chat-memory haystack, restaurant, etc.).

    The genuinely-required surface is small — three abstract methods
    (``visible_state`` / ``step_turn`` / ``build_snapshot``) and the
    turn counter. Everything else (``today_logs``, ``net_worth``,
    ``begin_turn``, prompts, probes) has sensible defaults so an env
    only implements what it actually needs.
    """

    turn_idx: int

    # ---------- required ----------

    @abstractmethod
    def visible_state(self) -> dict:
        """Return the state visible to the agent (JSON-serializable)."""

    @abstractmethod
    def step_turn(self) -> TurnResult:
        """Advance the simulation by one turn. Returns the turn's results."""

    @abstractmethod
    def build_snapshot(self) -> EnvSnapshot:
        """Build a snapshot of the current environment state for evaluation."""

    # ---------- optional hooks (sensible defaults) ----------

    def today_logs(self) -> list[str]:
        """Return outcome log lines describing what happened in the
        just-completed turn.

        Vending uses these to surface sales / delivery / fee events to
        the agent's next-turn prompt. Passive observation envs (chat-
        memory benchmarks) have no outcomes and inherit the default
        empty list.
        """
        return []

    def net_worth(self) -> float:
        """Current net worth of the agent in the simulation.

        Economic envs (vending) override; non-economic envs inherit
        the default ``0.0`` (the benchmark loop uses this for snapshot
        / RunStats fields that don't apply).
        """
        return 0.0

    def begin_turn(self, turn_idx: int) -> list[str]:
        """Called at the start of each turn. Returns context notes.

        Override in subclasses to provide environment-driven per-turn
        context. Default: no-op returning empty list.
        """
        return []

    def ingest_all(self) -> None:
        """Hook fired once at task start, before any agent session.

        Default no-op. Envs whose data is "given upfront" — every
        historical chat present at task start, no real-time interaction
        — override this to append the whole haystack into ``E`` in one
        shot. The ingestor then derives ``W`` lazily on first ``ms``
        access. Pairs with ``num_turns = 0`` + a single end-of-task
        probe; see PR #4 (LME) and PR #5 (BEAM).

        Today every shipped env uses turn-by-turn ingestion (data lands
        in ``E`` via :meth:`begin_turn` / :meth:`step_turn` outcome
        logs), so the default is correct for the current behavior.
        """
        return None

    def get_end_of_task_probes(self) -> list[ProbeSpec]:
        """Return probes that fire AFTER the last turn, not on a turn.

        Default: empty. Envs that today use the ``+1 probe-only turn``
        trick (LME, BEAM) will move their probes here in PRs #4 / #5
        and drop the trick.

        Probes returned here fire in the same agent session as the
        last turn (cheap, shared in-context history). Future ``probe
        isolation`` modes (PR #6) can spawn a fresh session per probe.
        """
        return []

    def substrate_endgame_prompt(self) -> str:
        """Per-env "RUN STRUCTURE" section prepended to the agent's
        system prompt.

        Each environment fills in what one turn means, what
        ``today`` is, how a turn ends, and any env-specific
        protocol notes. Default: empty string. Most envs now fold
        this content directly into their agent's ``sys_prompt`` and
        leave this default in place.
        """
        return ""

    def probe_substrate_prompt(self) -> str:
        """Per-env probe-mode format rules (legacy hook).

        Originally the place to spell out scorer-shaped reply rules
        (Answer line, units, tolerances). Most envs now ride those
        rules on :meth:`probe_user_postscript` instead — keep the
        default empty string and the system prompt swap is never
        exercised for new envs.
        """
        return ""

    def probe_user_postscript(self) -> str:
        """Per-env reminder appended to the probe's user-turn message.

        Returned text is appended after the question itself in the
        message that ``inject_probe`` hands to ``agent.answer_probe``.
        Use it for short, scorer-shaped reminders (e.g. vending's
        "Answer line must contain every value the question asks for"
        with worked unit examples). Larger format rules belong in
        :meth:`probe_substrate_prompt`. Default: empty string.
        """
        return ""

    def is_terminal(self) -> bool:
        """Whether the simulation has reached a terminal state and the
        outer loop should exit early. Default: ``False``.

        Override for envs with a domain-specific failure mode (e.g.
        vending bankruptcy after N consecutive negative-cash turns).
        """
        return False

    def get_probes(self, turn_idx: int) -> list[ProbeSpec]:
        """Return probe questions scheduled for this turn.

        Override in subclasses. Default: no probes.
        """
        return []

    def summarize_probes(self, results: list[ProbeResult]) -> dict:
        """Return env-specific summary fields to add to ``probe_results.json``
        and to the run span as ``run.<key>`` attributes.

        Default: ``{}`` (no extra summary). Vending overrides to add
        per-category (A/B) averages; future envs can add their own
        roll-ups without touching ``core``.
        """
        return {}

    def report_run_metrics(self, agent: Any) -> dict:
        """Return env-specific run-end metrics for ``RunStats.env_metrics``.

        These are the env's headline outcome signals (e.g. vending's
        ``net_worth`` / ``units_sold`` / ``bankrupt``). They're written
        into ``RunStats.env_metrics``, surfaced as ``run.env.<key>`` OTel
        attributes, and aggregated key-by-key by :func:`Scroll.benchmark.aggregate`.

        Default: ``{}`` (no env-specific outcome metrics — fine for
        pure-retrieval envs like LongMemEval and BEAM, where the only
        meaningful signal is ``probe_avg_score`` + token cost).
        """
        return {}

    def register_tools(self, toolkit: Any) -> None:
        """Register environment-specific tools on the toolkit.

        Override in subclasses. Default: no-op.
        """

    def to_checkpoint(self) -> dict:
        """Serialize environment state for checkpoint/resume."""
        raise NotImplementedError

    @classmethod
    def from_checkpoint(cls, data: dict, cfg: Any) -> BaseEnvironment:
        """Restore environment from checkpoint data."""
        raise NotImplementedError


class BaseDataSource(ABC):
    """Abstract external data source for an environment.

    The only required method is :meth:`begin_turn` (so the framework
    can stage per-turn input). The email + search channels have
    no-op defaults — vending overrides them; chat-memory benchmarks
    leave them at the defaults.
    """

    # ---------- required ----------

    @abstractmethod
    def begin_turn(self, turn_idx: int, env: BaseEnvironment) -> list[str]:
        """Start a new turn; return any notes/briefing for the agent."""

    # ---------- optional channels (default: disabled) ----------

    def read_emails(self, limit: int = 10) -> list[str]:
        """Return unread email messages. Default: none (channel disabled)."""
        return []

    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        turn_idx: int,
        env: BaseEnvironment,
    ) -> str:
        """Send an email and return the response/confirmation.

        Default: returns a stub indicating the channel is disabled in
        this env. Vending overrides to deliver mail to suppliers.
        """
        return "email_disabled (no datasource override)"

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Search external data sources and return results.

        Default: none (channel disabled).
        """
        return []

    def get_reports(self) -> list[str]:
        """Return available reports. Default: empty list."""
        return []

    def to_checkpoint(self) -> dict:
        """Serialize data source state for checkpoint/resume."""
        raise NotImplementedError

    def from_checkpoint(self, data: dict) -> None:
        """Restore data source state from checkpoint data."""
        raise NotImplementedError
