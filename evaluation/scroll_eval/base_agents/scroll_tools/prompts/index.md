# Step headlines (milestone markers)

Emit a single fenced line `⟦ <what this milestone establishes> ⟧` (⟦ U+27E6, ⟧ U+27E7) **only when a step reaches something worth finding again** — a fact, a decision, a finished sub-task, a dead-end to skip — not routine ones. Keep it one line, under ~15 words, and specific — name the value/file/entity, e.g. `⟦ config.db_host = "prod-3" ⟧`. Headlined steps become entries in the **eviction index**; unheadlined steps are still stored and findable with `search_history`, just not on the index. Headlines are also your durable working notes: **old tool outputs age out of this prompt after a few steps — your headlines don't** — so when a search/read establishes something you'll need later (a seq, a value, a conflict), record it in the headline instead of trusting the output to stay visible.

**Absence is a milestone too.** When targeted searching establishes that the specific asked-for fact is NOT in the history — only related discussion is — record the verdict the same way: `⟦ not found: <the specific fact> — only adjacent discussion (checked <spans/sessions>) ⟧`. A recorded absence verdict is evidence you keep; without it, later steps re-find the same adjacent material and mistake it for the fact.

# Your eviction index (the `[memory]` message)

History that is no longer in this prompt — prior conversation turns, and any of your own steps that scrolled out — was folded into **one in-context index** over rounds of compaction: a rough, glanceable map of everything that came before. Oldest entries are at the top, newest at the bottom; recent lines carry more detail, older ones less. The index is the cheap overview; **your tools are the precise interface** to the same history. **You're free to start either way** — scan the index to find the region and drill into it, or go straight to `search_history` by keyword — and combine them (an index line gives a `seq` range or session numbers; a search hit gives a `seq` to widen around).

Line shapes:

- `[L0] seq <lo>–<hi>` heading a group of `· seq <n>  ⟦ <headline> ⟧` rows — one compaction sweep: the heading's span covers EVERYTHING that left the prompt (headlined or not); the `·` rows are just its milestone steps.
- `· S<n> seq <lo>–<hi>  ⟦ A - B ⟧` — one recent prior session (**A** = the topic of its first turn, **B** = a summary of the whole session).
- `[Lk] S<a>–S<b>  seq <lo>–<hi>  — N sessions  ⟦ A - B ⟧  · topics: …` — a **chunk**: N consecutive older sessions squeezed to one line. **A** is the first member's opening summary and **B** the last member's closing summary — they bracket an era, they do not describe every member. The `topics:` strip lists words that recur across the member summaries — scent for what the era's *middle* contains, not a complete list; **if a topic word relates to your question, that chunk's span is worth one drill-down query even though its endpoints look unrelated.** Higher `k` = older and coarser.

Headlines and summaries are overviews, not ground truth: **verify dates, quantities, and quoted values in the raw turns before using them in an answer.**

Drill in coarse-to-fine:
1. Scan the index for a line whose headline, session numbers, or `seq` range looks relevant.
2. Sweep that span with a bounded search: `search_history(query="<topic terms>", seq_range=[<lo>, <hi>], k=20)`. For an `S<n>` session line, bound by session instead: `search_history(query="<terms>", step_range=[<n>, <n>])`.
3. Read the turns that matter in full: `expand_turns(seqs=[...])` — also works after a global search (a hit gives a `seq`; expand a few neighbours around it for context).

When a question is about *what a session covered*, **pick the on-topic session(s) by their summaries first** — then drill into only those (a bounded search or expand over their spans).

**Before you conclude something is absent — and for enumerate-everything questions** (every time X came up, all sessions about Y): a global `search_history` returns the *k best matches*, not all of them, and hits *about the topic* are not the asked-for fact. Walk the map: pick every span whose summaries, sessions, or dates could plausibly hold the answer, and sweep each with a bounded `search_history(query=..., seq_range=[<lo>, <hi>])`. Then **consolidate before answering**: merge the seqs your global searches surfaced with the seqs the map sweeps surfaced — one deduped set — and expand whatever only one side found. An "it isn't in the history" verdict should name the spans you swept, and "the topic appears but the specific fact was never stated" is a legitimate — often the correct — answer.

Lean on the index when you can't name a keyword — *when*, in what *order*, how something *changed*, or what a whole region was about; reach for `search_history` when you have a specific term.
