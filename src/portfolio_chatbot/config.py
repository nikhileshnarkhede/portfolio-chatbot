"""
config.py - the single source of truth for every tunable parameter.

Nothing in this project reads a hardcoded pipeline constant. Everything comes
from `configs/default.json`, optionally deep-merged with an experiment delta
from `configs/experiments/*.json`, optionally overridden by dotted CLI flags.

    from portfolio_chatbot.config import load_config

    cfg = load_config()                                  # baseline
    cfg = load_config("exp002_chunk_512")                # named experiment
    cfg = load_config("exp002_chunk_512",
                      overrides=["retrieval.mmr.k=12"])  # ad-hoc sweep point

Three guarantees this module provides, all of which the evaluation layer
depends on:

1. **Validation.** Every model sets `extra="forbid"`, so a typo like
   `"chunk_overlapp"` fails at load time instead of being silently ignored
   and quietly invalidating an experiment.
2. **Immutability.** The returned config is frozen. A node cannot mutate a
   parameter mid-run, so the config recorded in the run log is provably the
   config that produced the answer.
3. **Fingerprints.** `index_fingerprint` hashes only the parameters that
   affect the vector index, so two configs that chunk identically share an
   index instead of rebuilding it. `run_fingerprint` hashes everything that
   affects the answer, and is stamped onto every logged run so results can
   always be traced back to their exact configuration.

Secrets never live here. `GROQ_API_KEY` comes from the environment or
`.streamlit/secrets.toml`; `llm/provider.py` reads it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------
# Project layout. Anchored to this file so the CWD never matters - the app,
# the eval runner and pytest all resolve the same paths.
#   .../portfolio_Chatbot/src/portfolio_chatbot/config.py
#   parents[0] = portfolio_chatbot, parents[1] = src, parents[2] = repo root
# --------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.json"
EXPERIMENT_DIR = CONFIG_DIR / "experiments"


@lru_cache(maxsize=16)
def _digest_cached(path: str, mtime_ns: int, size: int) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _digest_file(path: Path) -> str:
    """Short content hash of `path`, or "absent" if it cannot be stat'ed/read.

    The (mtime, size) stat is the cache key, so an edited file re-hashes while
    an untouched one costs a stat call. Both the stat and the read are guarded:
    the resume is legitimately missing before the first ingest, and a caller
    asking "which index would this config use" must get an answer either way.
    """
    try:
        stat = path.stat()
        return _digest_cached(str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return "absent"


class _Base(BaseModel):
    """Every config section: frozen, and rejects unknown keys."""
    model_config = ConfigDict(frozen=True, extra="forbid")


# ==========================================================================
# Sections
# ==========================================================================

class PathsConfig(_Base):
    raw_resume: str
    processed_dir: str
    index_root: str
    prompts_dir: str
    prompt_registry: str
    runs_dir: str
    eval_reports_dir: str


class StructuralChunkingConfig(_Base):
    section_tags: list[str]
    item_tags: list[str]
    type_map: dict[str, str]
    emit_section_intro: bool = True
    strip_html_comments: bool = True
    collapse_blank_lines: bool = True
    carry_tag_attributes_to_metadata: bool = True

    @model_validator(mode="after")
    def _every_item_tag_has_a_type(self) -> "StructuralChunkingConfig":
        missing = [t for t in self.item_tags if t not in self.type_map]
        if missing:
            raise ValueError(
                f"item_tags missing from type_map: {missing}. Every item tag must "
                f"map to a chunk_type, because retrieval routing filters on it."
            )
        return self


class SplitConfig(_Base):
    enabled: bool = True
    max_chunk_chars: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    min_inner_chunk_chars: int = Field(gt=0)
    reattach_header: bool = True
    separators: list[str]

    @model_validator(mode="after")
    def _overlap_fits(self) -> "SplitConfig":
        if self.chunk_overlap >= self.max_chunk_chars:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"max_chunk_chars ({self.max_chunk_chars}) or splitting never terminates."
            )
        return self


class IngestionConfig(_Base):
    strategy: Literal["structural", "recursive", "fixed", "semantic"]
    structural: StructuralChunkingConfig
    split: SplitConfig
    dump_chunks: bool = True


class EmbeddingConfig(_Base):
    provider: Literal["huggingface"]
    model_name: str
    device: Literal["cpu", "cuda", "mps"]
    normalize_embeddings: bool
    batch_size: int = Field(gt=0)


class VectorStoreConfig(_Base):
    backend: Literal["faiss"]
    distance_strategy: Literal["COSINE", "EUCLIDEAN_DISTANCE", "MAX_INNER_PRODUCT"]
    allow_dangerous_deserialization: bool


class RouteRule(_Base):
    name: str
    keywords: list[str]
    types: list[str]


class RoutingConfig(_Base):
    enabled: bool
    route_on: Literal["raw_question", "expanded_question"]
    routes: list[RouteRule]

    @field_validator("routes")
    @classmethod
    def _lowercase_keywords(cls, routes: list[RouteRule]) -> list[RouteRule]:
        # Routing matches against a lowercased question; a capitalised keyword
        # in the JSON would silently never fire.
        for r in routes:
            bad = [k for k in r.keywords if k != k.lower()]
            if bad:
                raise ValueError(f"route '{r.name}' has non-lowercase keywords: {bad}")
        return routes


class ExpansionRule(_Base):
    keywords: list[str]
    append: str


class QueryExpansionConfig(_Base):
    enabled: bool
    mode: Literal["keyword_rules", "llm", "none"]
    rules: list[ExpansionRule]


class FilteredRetrievalConfig(_Base):
    k: int = Field(gt=0)
    fetch_k: int = Field(gt=0)

    @model_validator(mode="after")
    def _fetch_k_at_least_k(self) -> "FilteredRetrievalConfig":
        if self.fetch_k < self.k:
            raise ValueError(f"fetch_k ({self.fetch_k}) must be >= k ({self.k}).")
        return self


class MMRConfig(_Base):
    k: int = Field(gt=0)
    fetch_k: int = Field(gt=0)
    lambda_mult: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _fetch_k_at_least_k(self) -> "MMRConfig":
        if self.fetch_k < self.k:
            raise ValueError(f"fetch_k ({self.fetch_k}) must be >= k ({self.k}).")
        return self


class ContextConfig(_Base):
    max_context_chars: int = Field(gt=0)
    section_template: str
    joiner: str
    empty_message: str

    @field_validator("section_template")
    @classmethod
    def _has_placeholders(cls, v: str) -> str:
        for token in ("{i}", "{content}"):
            if token not in v:
                raise ValueError(f"section_template must contain {token}")
        return v


class RetrievalConfig(_Base):
    routing: RoutingConfig
    query_expansion: QueryExpansionConfig
    filtered: FilteredRetrievalConfig
    mmr: MMRConfig
    context: ContextConfig


class ModelPricing(_Base):
    """USD per 1M tokens for one model."""
    input_per_1m: float = Field(ge=0.0)
    output_per_1m: float = Field(ge=0.0)


class LLMConfig(_Base):
    provider: Literal["groq"]
    model_chain: list[str] = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int | None = None
    top_p: float = Field(ge=0.0, le=1.0)
    streaming: bool
    request_timeout: int = Field(gt=0)
    max_retries: int = Field(ge=0)
    fallback_on_rate_limit: bool
    remember_working_model: bool
    pricing: dict[str, ModelPricing] = Field(default_factory=dict)

    @property
    def primary_model(self) -> str:
        return self.model_chain[0]

    @property
    def largest_model(self) -> str:
        """Best available guess at the strongest model in the chain.

        Ranked by parameter count parsed from the name ("120b" > "27b" > "20b"),
        falling back to the last entry. Heuristic on purpose: the free-tier
        catalogue moves, and hardcoding a specific model would go stale. Set
        `eval.judge.model` to override.
        """
        def size(name: str) -> float:
            m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", name.lower())
            return float(m.group(1)) if m else -1.0
        return max(self.model_chain, key=size)

    def rate_for(self, model: str) -> tuple[float, float]:
        """(input, output) USD per 1M tokens. Zero when unpriced.

        Unpriced is the honest default: the free tier costs nothing, and
        inventing a rate would produce a confident wrong number in every
        report. Fill `llm.pricing` in from your plan to get real figures.
        """
        entry = self.pricing.get(model) or self.pricing.get("default")
        return (entry.input_per_1m, entry.output_per_1m) if entry else (0.0, 0.0)


class PromptsConfig(_Base):
    system: str
    query_expansion: str
    summarizer: str


class MemoryConfig(_Base):
    enabled: bool
    checkpointer: Literal["memory", "sqlite", "none"]
    summarize_after_n_messages: int = Field(gt=0)
    keep_last_n: int = Field(gt=0)
    summary_max_sentences: int = Field(gt=0)
    user_label: str
    assistant_label: str

    @model_validator(mode="after")
    def _keep_fewer_than_trigger(self) -> "MemoryConfig":
        if self.keep_last_n >= self.summarize_after_n_messages:
            raise ValueError(
                f"keep_last_n ({self.keep_last_n}) must be < summarize_after_n_messages "
                f"({self.summarize_after_n_messages}) or summarization never shrinks history."
            )
        return self


class URLAllowlistConfig(_Base):
    enabled: bool
    strip_bare_urls: bool
    keep_markdown_label: bool
    ignore_trailing_slash: bool
    urls: list[str]

    @property
    def normalized(self) -> frozenset[str]:
        """Allowlist in the form the link guard compares against."""
        if self.ignore_trailing_slash:
            return frozenset(u.rstrip("/") for u in self.urls)
        return frozenset(self.urls)


class SafetyConfig(_Base):
    url_allowlist: URLAllowlistConfig
    refusal_message: str


class LangSmithConfig(_Base):
    enabled: bool
    project: str


class ObservabilityConfig(_Base):
    log_runs: bool
    log_context: bool
    log_retrieved_chunk_ids: bool
    log_latency: bool
    log_token_usage: bool
    langsmith: LangSmithConfig


class ContactLink(_Base):
    label: str
    url: str


class ContactConfig(_Base):
    name: str
    meta: list[str]
    links: list[ContactLink]


class QuickAction(_Base):
    label: str
    question: str


class UIConfig(_Base):
    page_title: str
    page_icon: str
    heading: str
    tagline: str
    caption: str
    assistant_avatar: str
    portfolio_url: str
    stream_delay: float = Field(ge=0.0)
    show_debug: bool
    show_experiment_switcher: bool
    contact: ContactConfig
    quick_actions: list[QuickAction]


class JudgeConfig(_Base):
    """The LLM judge. Deliberately decoupled from `llm.model_chain`.

    A weak judge is worse than no judge: it produces numbers that look like
    measurements and are not. Decoupling also means judge calls draw from a
    different Groq quota than generation calls, so judging does not eat the
    budget that produces the answers.
    """
    enabled: bool = False
    model: str | None = None          # None -> largest model in llm.model_chain
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_context_chars: int = Field(default=6000, gt=0)


class CalibrationConfig(_Base):
    """How far to trust the judge, by measured agreement with a human."""
    min_kappa_trust: float = Field(default=0.75, ge=0.0, le=1.0)
    min_kappa_direction: float = Field(default=0.60, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> "CalibrationConfig":
        if self.min_kappa_direction > self.min_kappa_trust:
            raise ValueError("min_kappa_direction must be <= min_kappa_trust")
        return self


class EvalConfig(_Base):
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)


class RunConfig(_Base):
    experiment_name: str
    seed: int
    notes: str = ""


# ==========================================================================
# Root
# ==========================================================================

class AppConfig(_Base):
    paths: PathsConfig
    ingestion: IngestionConfig
    embedding: EmbeddingConfig
    vectorstore: VectorStoreConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    prompts: PromptsConfig
    memory: MemoryConfig
    safety: SafetyConfig
    observability: ObservabilityConfig
    ui: UIConfig
    eval: EvalConfig = Field(default_factory=EvalConfig)
    run: RunConfig

    # ---------------- resolved absolute paths ----------------

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def resolve(self, relative: str) -> Path:
        """Turn a repo-relative path from `paths` into an absolute Path.

        An absolute value in `paths` is honoured as-is (`Path("/a") / "/tmp/b"`
        is `/tmp/b`), so runs and indexes can be pointed at a scratch disk.
        """
        return PROJECT_ROOT / relative

    def display_path(self, path: Path) -> str:
        """Repo-relative when possible, absolute otherwise.

        `Path.relative_to` RAISES when the target is outside the project, which
        it legitimately is whenever `paths.runs_dir` points elsewhere. Reports
        should not crash over how a path is rendered.
        """
        try:
            return str(Path(path).relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    @property
    def resume_path(self) -> Path:
        return self.resolve(self.paths.raw_resume)

    @property
    def index_path(self) -> Path:
        """Index directory for THIS chunking+embedding configuration.

        Keyed by `index_fingerprint`, so switching between experiments never
        silently queries an index that was built with different chunking.
        """
        return self.resolve(self.paths.index_root) / self.index_fingerprint

    @property
    def run_dir(self) -> Path:
        return self.resolve(self.paths.runs_dir) / self.run.experiment_name

    # ---------------- fingerprints ----------------

    def _hash(self, payload: Any) -> str:
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    @property
    def resume_digest(self) -> str:
        """Content hash of the resume, or "absent" when there is no file.

        The CONTENT, not the path. Two different resumes at the same path are
        two different indexes, and hashing the path alone made them collide:
        swap the file, and every retrieval keeps answering from the vectors
        built out of the old one - with no error, because the index the config
        asks for is right there on disk.

        That is the whole failure this project exists to prevent, and it stops
        being hypothetical the moment somebody supplies their own resume at the
        same path - which is exactly what the container is for.

        A missing file hashes as "absent" rather than raising: `index_path` is
        read while deciding whether an ingest is needed, and it must be
        answerable before the file is guaranteed to exist.

        Cached on (path, mtime, size) because this is reached on every
        retrieval - `load_store` keys its cache on `index_fingerprint` - and
        re-hashing the file per turn would be work done to learn nothing. Touch
        the file and the key changes, so an edit is still picked up.
        """
        return _digest_file(self.resolve(self.paths.raw_resume))

    @property
    def index_fingerprint(self) -> str:
        """Hash of everything that changes the vectors on disk.

        Retrieval parameters (k, mmr, routing) are deliberately EXCLUDED - they
        are applied at query time, so sweeping them must not force a re-ingest.
        """
        return self._hash({
            "resume": self.paths.raw_resume,
            "resume_digest": self.resume_digest,
            "ingestion": self.ingestion.model_dump(mode="json"),
            "embedding": self.embedding.model_dump(mode="json"),
            "vectorstore": self.vectorstore.model_dump(mode="json"),
        })

    @property
    def run_fingerprint(self) -> str:
        """Hash of everything that can change the model's answer.

        Excludes `ui`, `observability` and `run.notes`: cosmetic or logging-only
        changes must not look like a new experimental condition.
        """
        return self._hash({
            "index": self.index_fingerprint,
            "retrieval": self.retrieval.model_dump(mode="json"),
            "llm": self.llm.model_dump(mode="json"),
            "prompts": self.prompts.model_dump(mode="json"),
            "memory": self.memory.model_dump(mode="json"),
            "safety": self.safety.model_dump(mode="json"),
            "seed": self.run.seed,
        })

    @property
    def judge_model(self) -> str:
        """The model the judge actually uses."""
        return self.eval.judge.model or self.llm.largest_model

    # ---------------- serialization ----------------

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def snapshot(self, directory: Path | None = None) -> Path:
        """Write the fully resolved config next to a run's outputs.

        The eval layer reads this, not `default.json` - a report must record the
        config that actually ran, including any CLI overrides.
        """
        directory = directory or self.run_dir
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "resolved_config.json"
        payload = {
            "_resolved": {
                "experiment_name": self.run.experiment_name,
                "index_fingerprint": self.index_fingerprint,
                "run_fingerprint": self.run_fingerprint,
            },
            **self.to_dict(),
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target


# ==========================================================================
# Loading
# ==========================================================================

def _strip_meta(node: Any) -> Any:
    """Drop `_`-prefixed documentation keys before validation.

    JSON has no comments, so `_meta` blocks carry the rationale for each
    experiment. They are documentation, not configuration, and must not reach
    the validator (which forbids unknown keys).
    """
    if isinstance(node, dict):
        return {k: _strip_meta(v) for k, v in node.items() if not k.startswith("_")}
    if isinstance(node, list):
        return [_strip_meta(v) for v in node]
    return node


def deep_merge(base: dict, delta: dict) -> dict:
    """Recursively merge `delta` over `base`.

    Dicts merge key by key. Every other type - lists included - REPLACES the
    base value wholesale. That is deliberate: an experiment that narrows
    `llm.model_chain` to a single model means exactly that, not "append".
    """
    out = copy.deepcopy(base)
    for key, value in delta.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(text: str) -> Any:
    """Parse a CLI override value as JSON, falling back to a bare string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_overrides(data: dict, overrides: list[str]) -> dict:
    """Apply dotted `key.path=value` overrides. Used by scripts/sweep.py.

        apply_overrides(cfg, ["retrieval.mmr.k=12", "llm.temperature=0.0"])

    Values are parsed as JSON, so `12` is an int, `0.0` a float, `true` a bool,
    `"gpt"` a string, and `[1,2]` a list. Paths must already exist, so a typo
    raises instead of quietly inventing a key the validator would then reject.
    """
    out = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key.path=value, got: {item!r}")
        dotted, raw = item.split("=", 1)
        parts = dotted.strip().split(".")
        cursor: Any = out
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                raise KeyError(f"Unknown config path in override {item!r}: '{part}' not found")
            cursor = cursor[part]
        leaf = parts[-1]
        if not isinstance(cursor, dict) or leaf not in cursor:
            raise KeyError(f"Unknown config path in override {item!r}: '{leaf}' not found")
        cursor[leaf] = _coerce(raw.strip())
    return out


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


def resolve_experiment_path(experiment: str | Path) -> Path:
    """Accept an experiment name, a bare filename, or a full path.

        "exp002_chunk_512"                          -> configs/experiments/exp002_chunk_512.json
        "exp002_chunk_512.json"                     -> configs/experiments/exp002_chunk_512.json
        "configs/experiments/exp002_chunk_512.json" -> as given
    """
    p = Path(experiment)
    if p.is_absolute() or p.exists():
        return p
    if p.suffix != ".json":
        p = p.with_suffix(".json")
    if p.parent == Path("."):
        p = EXPERIMENT_DIR / p.name
    else:
        p = PROJECT_ROOT / p
    return p


def load_config(
    experiment: str | Path | None = None,
    overrides: list[str] | None = None,
    base_path: Path = DEFAULT_CONFIG_PATH,
) -> AppConfig:
    """Load, merge, override and validate. The only entry point anything should use.

    Layering, lowest precedence first:
        configs/default.json  ->  experiment delta  ->  CLI overrides
    """
    data = _strip_meta(_read_json(base_path))

    if experiment is not None:
        exp_path = resolve_experiment_path(experiment)
        data = deep_merge(data, _strip_meta(_read_json(exp_path)))

    if overrides:
        data = apply_overrides(data, overrides)

    return AppConfig.model_validate(data)


def list_experiments() -> list[str]:
    """Experiment names available under configs/experiments/."""
    if not EXPERIMENT_DIR.exists():
        return []
    return sorted(p.stem for p in EXPERIMENT_DIR.glob("*.json"))


__all__ = [
    "AppConfig",
    "load_config",
    "list_experiments",
    "deep_merge",
    "apply_overrides",
    "resolve_experiment_path",
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "DEFAULT_CONFIG_PATH",
    "EXPERIMENT_DIR",
]
