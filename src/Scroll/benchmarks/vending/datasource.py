from __future__ import annotations

import random
from dataclasses import dataclass

from Scroll.core import BaseDataSource, serialize_rng, restore_rng
from Scroll.benchmarks.vending.env import VendingEnv


@dataclass
class Mail:
    day: int
    source: str
    subject: str
    body: str
    read: bool = False


class DataSourceManager(BaseDataSource):
    def __init__(self, seed: int, data_cfg: dict | None = None) -> None:
        self.rng = random.Random(seed + 101)
        self.cfg = data_cfg or {}
        self.inbox: list[Mail] = []
        self._scheduled: list[tuple[int, str, dict]] = []
        self._sub_agent_log: list[str] = []
        self._mail_log: list[str] = []
        self._daily_context: list[str] = []

        self._supplier_emails = {
            "metro_wholesale@example.com": {"cola": 1.02, "water": 0.53, "chips": 0.88, "gum": 0.34},
            "snackhub_supply@example.com": {"choco": 0.69, "nuts": 1.08, "juice": 0.84, "energy": 1.18},
            "allmart_vendor@example.com": {"cola": 1.03, "water": 0.56, "chips": 0.93, "juice": 0.86},
        }
        self._market_docs = [
            "market_report: college campus demand strong for water, cola, chips",
            "market_report: afternoon demand spikes for energy drinks and chocolate bars",
            "market_report: healthier mix can improve conversion in offices: nuts, water, juice",
            "pricing_note: typical margin target ranges between 45% and 80%",
            "supplier_directory: Metro Wholesale — contact metro_wholesale@example.com — carries cola, water, chips, gum",
            "supplier_directory: SnackHub Supply — contact snackhub_supply@example.com — carries choco, nuts, juice, energy drinks",
            "supplier_directory: AllMart Vendor — contact allmart_vendor@example.com — carries cola, water, chips, juice",
            "supplier_guide: email suppliers to request a price quote, then place orders by replying with product names and quantities",
            "supplier_guide: contact every candidate supplier early and compare quotes before committing — suppliers carry different SKUs at different prices, single-sourcing rarely wins",
            "operations_guide: collect machine cash before placing wholesale orders — uncollected cash sits in the machine and cannot be spent on inventory",
            "operations_guide: restock the machine daily — empty slots earn nothing while the daily operating fee still applies",
        ]

    def begin_turn(self, turn_idx: int, env: VendingEnv) -> list[str]:
        self._daily_context = []
        self._run_scheduled(turn_idx, env)
        if self.cfg.get("enable_weather_news", True):
            weather = self.rng.choice(["sunny", "cloudy", "rainy"])
            footfall = self.rng.choice(["low", "normal", "high"])
            weather_note = f"weather_feed day={turn_idx} weather={weather} footfall={footfall}"
            news_note = self.rng.choice(
                [
                    "news_feed: convenience trend stable",
                    "news_feed: price-sensitive week expected",
                    "news_feed: sports event nearby may raise drink demand",
                ]
            )
            self._daily_context.extend([weather_note, news_note])
        return list(self._daily_context)

    def context_lines(self) -> list[str]:
        if not self.cfg.get("enable_email", True):
            return list(self._daily_context)
        unread = [m for m in self.inbox if not m.read]
        inbox_lines = [f"email from={m.source} subject={m.subject} body={m.body}" for m in unread[-20:]]
        return self._daily_context + inbox_lines

    def read_emails(self, limit: int = 10) -> list[str]:
        if not self.cfg.get("enable_email", True):
            return []
        unread = [m for m in self.inbox if not m.read][:limit]
        out = []
        for m in unread:
            m.read = True
            out.append(f"email from={m.source} subject={m.subject} body={m.body}")
        return out

    def search(self, query: str, top_k: int = 3) -> list[str]:
        if not self.cfg.get("enable_search", True):
            return []
        terms = [t.lower() for t in query.split() if t.strip()]
        scored = []
        for doc in self._market_docs:
            score = sum(1 for t in terms if t in doc.lower())
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for s, d in scored[:top_k] if s > 0] or self._market_docs[:top_k]

    def sub_agent_specs(self) -> str:
        if not self.cfg.get("enable_sub_agent_channel", True):
            return "sub_agent disabled"
        return "sub_agent tools: set_price, restock, collect_cash, machine_inventory"

    def run_sub_agent(self, instruction: str, env: VendingEnv, units_per_sku: int) -> str:
        if not self.cfg.get("enable_sub_agent_channel", True):
            return env.restock(units_per_sku)
        instruction_l = instruction.lower()
        actions = []
        if "restock" in instruction_l:
            actions.append(env.restock(units_per_sku))
        if "collect" in instruction_l:
            actions.append(env.collect_cash())
        if "inventory" in instruction_l:
            actions.append(f"machine_inventory={dict(env.machine)}")
        if not actions:
            actions.append("sub_agent_noop")
        joined = " | ".join(actions)
        self._sub_agent_log.append(joined)
        return joined

    def chat_with_sub_agent(self) -> str:
        if not self.cfg.get("enable_sub_agent_channel", True):
            return "sub_agent_report: disabled"
        if not self._sub_agent_log:
            return "sub_agent_report: no actions yet"
        return f"sub_agent_report: last_action={self._sub_agent_log[-1]}"

    def send_email(self, to: str, subject: str, body: str, turn_idx: int, env: VendingEnv) -> str:
        if not self.cfg.get("enable_email", True):
            return f"email_disabled fallback_no_send to={to}"
        record = f"sent_email to={to} subject={subject}"
        self._mail_log.append(record)
        subject_l = subject.lower()
        body_l = body.lower()
        is_price_inquiry = any(
            kw in subject_l or kw in body_l
            for kw in ("quote", "price", "pricing", "inquiry", "catalog", "cost")
        )
        if to in self._supplier_emails and is_price_inquiry:
            catalog = self._supplier_emails[to]
            price_lines = ", ".join(f"{sku}: ${cost:.2f}/unit" for sku, cost in catalog.items())
            reply = f"Thank you for your inquiry. Here are our current prices: {price_lines}. To place an order, reply with subject containing 'order' and list items as sku=quantity in the body (e.g. cola=20 water=15)."
            self._scheduled.append((turn_idx + 1, "mail_reply", {"from": to, "subject": "Re: Price List", "body": reply}))
        is_order = any(
            kw in subject_l or kw in body_l
            for kw in ("order", "purchase", "buy")
        )
        if to in self._supplier_emails and is_order:
            items = self._parse_order_body(body)
            supplier_catalog = self._supplier_emails[to]
            valid_items = {k: v for k, v in items.items() if k in supplier_catalog}
            rejected_items = {k: v for k, v in items.items() if k not in supplier_catalog}
            if rejected_items:
                self.inbox.append(
                    Mail(
                        day=turn_idx,
                        source=to,
                        subject="Order partially rejected",
                        body=f"items_not_carried={rejected_items} available={list(supplier_catalog.keys())}",
                    )
                )
            if valid_items:
                self._scheduled.append((turn_idx + 1, "supplier_order", {"from": to, "items": valid_items}))
        return record

    def _run_scheduled(self, turn_idx: int, env: VendingEnv) -> None:
        keep: list[tuple[int, str, dict]] = []
        for due_day, kind, payload in self._scheduled:
            if due_day > turn_idx:
                keep.append((due_day, kind, payload))
                continue
            if kind == "mail_reply":
                self.inbox.append(
                    Mail(
                        day=turn_idx,
                        source=payload["from"],
                        subject=payload["subject"],
                        body=payload["body"],
                    )
                )
            elif kind == "supplier_order":
                outcome = env.order(payload["items"])
                self.inbox.append(
                    Mail(
                        day=turn_idx,
                        source=payload["from"],
                        subject="Order confirmation",
                        body=f"order_result={outcome} items={payload['items']}",
                    )
                )
        self._scheduled = keep

    def to_checkpoint(self) -> dict:
        return {
            "rng_state": serialize_rng(self.rng),
            "inbox": [
                {"day": m.day, "source": m.source, "subject": m.subject,
                 "body": m.body, "read": m.read}
                for m in self.inbox
            ],
            "_scheduled": [[day, kind, payload] for day, kind, payload in self._scheduled],
            "_sub_agent_log": list(self._sub_agent_log),
            "_mail_log": list(self._mail_log),
            "_daily_context": list(self._daily_context),
        }

    def from_checkpoint(self, data: dict) -> None:
        restore_rng(self.rng, data["rng_state"])
        self.inbox = [
            Mail(day=m["day"], source=m["source"], subject=m["subject"],
                 body=m["body"], read=m["read"])
            for m in data["inbox"]
        ]
        self._scheduled = [(s[0], s[1], s[2]) for s in data["_scheduled"]]
        self._sub_agent_log = data["_sub_agent_log"]
        self._mail_log = data["_mail_log"]
        self._daily_context = data["_daily_context"]

    def _parse_order_body(self, body: str) -> dict[str, int]:
        items: dict[str, int] = {}
        for part in body.replace(",", " ").split():
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            try:
                q = int(v)
            except ValueError:
                continue
            if q > 0:
                items[k.strip().lower()] = q
        return items
