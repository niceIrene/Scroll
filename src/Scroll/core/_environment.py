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

    def get_end_of_task_probes(self) -> list[ProbeSpec]:
        """Return probes that fire AFTER the last turn, not on a turn.

        Default: empty. LME and BEAM override to fire their QA / probe
        questions here under ``agent_during_ingestion=false``.
        """
        return []

    def probe_isolation(self) -> str:
        """How end-of-task probes are isolated from each other.

        Two modes:

        - ``"shared"`` (default): all probes share one agent session —
          probe N sees probes 1..N-1's exchange in history. Cheap
          (system prompt amortized) but minor in-context leak.
        - ``"fresh"``: each probe (except the first) ends the current
          agent session and starts a new one. The probe answers from
          ``W`` only — the cleanest SCROLL-purity test. More LLM
          cost (re-warmed system prompt per probe) and agents'
          ``end_session`` / ``start_session`` subclass hooks fire
          repeatedly, so they must be safe to call multiple times.

        Only meaningful for envs whose ``get_end_of_task_probes``
        returns ``len > 1`` — BEAM today (LME has a single probe,
        so the mode is moot).
        """
        return "shared"

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
