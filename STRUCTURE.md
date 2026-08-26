# Structure map

`README.md` is the front door — what this is and how to run it.
`docs/EVALUATION.md` is the evaluation pipeline in full. This file is the
internal map: what each file owns, and why a few of them exist at all.

Every file has one responsibility. Nothing is duplicated. The organizing rule:
**every knob you want to A/B test lives in `configs/*.json` or `prompts/*.md`,
never inside Python.**

---

## configs/ — the experiment surface

| File | Holds |
|---|---|
| `default.json` | Base config, 13 blocks. Every parameter the app reads. |
| `prompt_registry.json` | Maps a prompt id (`system.v2_persona`) to a file under `prompts/`. |
| `experiments/expNNN_*.json` | A **delta** over `default.json`. One file per variant. |
| `history/default_<stamp>_<fp8>.json` | The previous default, archived by every apply. Rollback source. |
| `provenance.json` | What was applied, when, on the strength of which report, and the gate verdict. |

`_meta` blocks carry the rationale (JSON has no comments); `config.py` strips
`_`-prefixed keys before validation.

Blocks: `paths`, `ingestion`, `embedding`, `vectorstore`, `retrieval`, `llm`,
`prompts`, `memory`, `safety`, `observability`, `ui`, `eval`, `run`.

---

## src/portfolio_chatbot/ — the core

### Foundation

| File | Responsibility |
|---|---|
| `config.py` | Load `default.json`, deep-merge an experiment delta, apply dotted CLI overrides, validate. Frozen, `extra="forbid"`. Computes `index_fingerprint` and `run_fingerprint`. |
| `state.py` | The LangGraph `State` TypedDict — the contract between all nodes. Reducers, the `RESET` sentinel, `EVAL_FIELDS`, `chunk_refs`. |
| `graph.py` | Topology only: nodes, edges, the conditional edge into `summarize`, compilation. No logic. |

### Nodes — one per file

| File | Responsibility |
|---|---|
| `nodes/__init__.py` | `app_config(config)` — pulls the AppConfig off the RunnableConfig. |
| `nodes/expand_query.py` | Widen the ranking text: `keyword_rules`, `llm`, or `none`. |
| `nodes/route.py` | Question → `chunk_type` filter, via the config's keyword table. |
| `nodes/retrieve.py` | Run the search, assemble and truncate the context. |
| `nodes/generate.py` | Render the prompt, call the model, walk the fallback chain. Writes `draft_answer`. |
| `nodes/sanitize.py` | Strip non-allowlisted URLs. Writes `answer` **and** appends the reply to history. |
| `nodes/summarize.py` | Condense older turns once history outgrows the window. |

### Supporting

| File | Responsibility |
|---|---|
| `tools/link_guard.py` | The URL allowlist, returning an audit of what it removed. |
| `tools/retriever_tool.py` | FAISS load + filtered/MMR search + context formatting. |
| `tools/registry.py` | Reserved for LLM tool-calling. Deliberately empty — see its docstring. |
| `llm/provider.py` | Groq client factory, cached. Key resolved at runtime from four sources — env, secrets.toml, `GROQ_API_KEY_FILE`, `/run/secrets/groq_api_key` — never from config. |
| `llm/fallback.py` | The rate-limit chain, as a pure function over a callable. Harvests TTFT and token usage off the raw chunks. |
| `prompts/loader.py` | Resolve a prompt id to a validated `PromptTemplate`. |
| `ingestion/parser.py` | `resume.txt` → structured `ParsedItem`s. Decides *meaning*. |
| `ingestion/chunker.py` | `ParsedItem`s → `Document`s. Decides *size*. Stable chunk ids. |
| `ingestion/embedder.py` | The one place an embedder is built. |
| `ingestion/build_index.py` | Build/reuse the FAISS index, write `manifest.json`. |
| `memory/history.py` | Render conversation history as prompt text. |
| `memory/checkpointer.py` | `memory` / `sqlite` / `none` backends. |
| `observability/run_logger.py` | Write each turn to `runs/<exp>/<run_id>/turns.jsonl`. |
| `observability/perf.py` | Per-node timing into `state.timings`; TTFT and cold-start capture. |
| `observability/tracing.py` | Optional LangSmith. Off by default. |

---

## ui/ — Streamlit only

### The chatbot

`app.py` (entry, path bootstrap, secrets), `styles.css` (extracted verbatim from
the original), and `components/`: `header`, `chat`, `sidebar_controls`,
`debug_panel`. Zero pipeline logic anywhere under here.

`chat.py` also carries the inline **field rating** control (debug mode only) —
logged, never gated, and used as a source of new reference questions.

### The evaluation console

| File | Responsibility |
|---|---|
| `dashboard.py` | Entry point. Seven tabs, sidebar, live config validation. |
| `dash/tunables.py` | The 15 `Knob` definitions — path, widget, range, group, and whether the knob `rebuilds_index`. |
| `dash/state.py` | The working config held in `st.session_state`; never written to disk until Apply. |
| `dash/panels.py` | One function per tab: `tune`, `run_tab`, `scorecard`, `compare`, `rate`, `apply_tab`, `thresholds`. |

---

## eval/

| File | Responsibility |
|---|---|
| `runner.py` | Run a dataset through the graph and score it. Logs every turn before scoring, so a report can be recomputed without re-calling the model. |
| `compare.py` | Per-metric deltas **and** per-case regressions between two reports. |
| `noise.py` | 5 identical runs → σ and MDE (2σ) per metric. `TRACKED` names the metrics whose stability is meaningful. |
| `gate.py` | Invariants → thresholds → regression. Turns a report into a pass/fail decision. |
| `calibration.py` | Cohen's κ between judge and human on the calibration set. Verdict: trust / direction / suppress. |
| `ratings.py` | The human 0–10 store. Keyed on a hash of the answer text. Reference and field stores kept separate. |
| `apply.py` | Promote a config: readiness checks, archive, delta, stale detection, provenance, rollback. |
| `thresholds.json` | The acceptance criteria. **You own these numbers**, not the code. |
| `baselines/` | Recorded means and σ from `--save-baseline`. |
| `reports/` | One JSON per run. |

### Datasets

`golden_qa` (40) · `adversarial_urls` (8) · `injection` (12) ·
`fabrication_bait` (12) · `consistency` (8 × 3) · `followups` (4 convos, 13
turns) · `judge_calibration` (30, balanced 15/15, judge validation only)

### Metrics

| File | Group |
|---|---|
| `metrics/retrieval.py` | route accuracy, hit@k, recall@k, MRR, nDCG@k, type precision, chunk coverage |
| `metrics/grounding.py` | entity extraction, entity grounding, fabricated-entity rate, consistency |
| `metrics/generation.py` | fact coverage, context overlap, first-person, and the `Judge` protocol |
| `metrics/safety.py` | forged-URL attempts, leaks, refusal accuracy, injection success, prompt leakage, persona breaks |
| `metrics/performance.py` | TTFT/latency percentiles, node latency, tokens, cost, fallback and error rates, cold start |
| `metrics/judge.py` | `GroqJudge` — claim-level faithfulness and relevancy. Off by default. |

## secrets/

`groq_api_key.txt.example` — a key-file template. Copy it, paste a Groq key,
and point `GROQ_API_KEY_FILE` at it; the key then stays out of your shell
history and out of every child process's environment. `.gitignore` ignores
`secrets/*` and re-includes `*.example`, so a filled-in copy cannot be
committed.

## .github/

`workflows/docker-publish.yml` — push to main → ruff + pytest → build → smoke
test the running container → push to Docker Hub. `publish` needs `test`, so a
red suite publishes nothing. `ruff.toml` keeps the lint selection narrow (E4,
E7, E9, F) on purpose: ruff's full default raises 160+ findings here, and a gate
that fails on every push is a gate everyone learns to ignore.

## scripts/

`ingest.py` · `run_eval.py` · `sweep.py` · `serve.py` (preflight + launch)

## tests/ — 379

| Tier | Files |
|---|---|
| Unit | `test_state` (24) · `test_chunker` (22) · `test_prompts` (19) · `test_provider` (46) · `test_link_guard` (13) |
| Node | `test_nodes` (38) · `test_graph` (19) |
| Eval | `test_eval` (45) · `test_apply` (31) · `test_judge` (31) · `test_performance` (30) · `test_grounding` (29) · `test_gate` (26) |
| UI | `test_ui` (6, marked `ui`) |

---

## Decisions worth not re-litigating

**Config is not in the state.** It rides on `RunnableConfig`. State is
serialized into every checkpoint; a frozen config carrying 31 URLs would be
copied into each one.

**`draft_answer` and `answer` are both kept.** Collapsing them makes URL
hallucination permanently unmeasurable — you cannot count links the model tried
to invent if you only keep the cleaned text.

**Two fingerprints, not one.** `index_fingerprint` covers ingestion + embedding
only, so retrieval sweeps never force a re-ingest. `run_fingerprint` covers
everything that changes an answer but excludes `ui` and `observability`, so a
CSS tweak doesn't look like a new experimental condition.

**`RESET` is `None`.** An appending reducer is called as `reducer(existing,
update)`, so returning `[]` appends nothing rather than clearing. A custom
sentinel object breaks msgpack checkpointing.

**Nodes must annotate `config: RunnableConfig`.** LangGraph dispatches on the
annotation; `Any` gets the node called with one argument.

**`str(chunk)` is never used on a stream piece.** It returns the object's repr,
not its text. `fallback.content_of` unwraps it — and the fallback must see the
raw chunk, not a pre-mapped string, or `harvest_usage` never sees the usage
metadata and every run silently reports zero tokens.

**`pytest.ini` deliberately omits `.` from `pythonpath`.** Adding it makes
`ui.components` importable under pytest but not under `streamlit run`, which is
how a `ModuleNotFoundError` once shipped with a green suite.

**Prompts are never edited in place once evaluated.** Add the next version to
`prompt_registry.json` instead.

**Ratings are keyed on the answer hash, not the case id.** An unchanged answer
keeps its rating, so a second pass only asks about answers that changed.
A rating scheme that re-scored everything after every tweak would be abandoned,
and the human dimension would quietly become unmeasured.

**Apply overwrites `default.json`.** The live config is one file and reading it
should never require reasoning about layered overrides. The archive, the delta,
the stale-experiment scan and the provenance record are what make that safe.

---

## Open items

- `langchain-community` is being sunset; standalone `langchain-faiss` 0.1.1 exists.
- No real-embedding eval baseline yet — every retrieval number so far came from
  a deterministic fake embedder.
- `thresholds.json` holds starting values, not values calibrated against a real
  baseline run.
- Routing gaps found by the golden set: questions naming a specific project or
  company ("What is the Supply Chain Tracker?", "What did you do at Bajaj
  Auto?") match no keyword and fall to MMR. Route accuracy 0.818. Entity-aware
  routing is the fix.
- Context truncates on 9 of 22 turns at `filtered.k = 14` against
  `max_context_chars = 8000`.
- The judge is `unmeasured` until `run_eval.py --calibrate` has run against a
  real key.
