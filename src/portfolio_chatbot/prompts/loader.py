"""
loader.py - resolve a prompt id to a validated PromptTemplate.

Prompts live as Markdown under `prompts/`, are named by id in
`configs/prompt_registry.json`, and are selected by id in `configs/*.json`.
Swapping the system prompt is therefore a config edit, and a prompt A/B is a
one-line experiment delta - no code changes, no redeploy.

The validation here is the point of the module. A prompt is a string with
holes in it, and the two ways that goes wrong are both silent:

* A placeholder in the file that nothing supplies -> KeyError at generation
  time, mid-conversation, after the retrieval work is already done.
* A variable the caller supplies that the file never uses -> the prompt looks
  fine and quietly ignores your context or chat history. Nothing raises. You
  discover it as an unexplained drop in groundedness three experiments later.

`load_prompt` checks both directions against the registry's declared
`input_variables` and refuses to return a mismatched template.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from langchain_core.prompts import PromptTemplate

from ..config import PROJECT_ROOT, AppConfig

#: Matches {placeholder} but not the escaped {{literal}} form.
_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{(\w+)\}(?!\})")


@dataclass(frozen=True)
class PromptSpec:
    """One registry entry, plus the resolved template text."""
    prompt_id: str
    path: str
    family: str
    version: int
    input_variables: tuple[str, ...]
    status: str
    description: str
    evaluated: bool
    text: str

    @property
    def chars(self) -> int:
        return len(self.text)


def placeholders(text: str) -> set[str]:
    """Placeholder names in a template, ignoring `{{escaped}}` braces."""
    return set(_PLACEHOLDER_RE.findall(text))


@lru_cache(maxsize=8)
def _read_registry(registry_path: str) -> dict:
    path = Path(registry_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt registry not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data.get("prompts")
    if not isinstance(prompts, dict):
        raise ValueError(f"{path} has no 'prompts' object.")
    return prompts


@lru_cache(maxsize=32)
def _load(registry_path: str, prompt_id: str) -> PromptSpec:
    prompts = _read_registry(registry_path)

    entry = prompts.get(prompt_id)
    if entry is None:
        available = ", ".join(sorted(prompts)) or "none"
        raise KeyError(f"Unknown prompt id {prompt_id!r}. Registered: {available}")

    file_path = PROJECT_ROOT / entry["path"]
    if not file_path.exists():
        raise FileNotFoundError(
            f"Prompt {prompt_id!r} points at {entry['path']}, which does not exist."
        )

    text = file_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(
            f"Prompt {prompt_id!r} ({entry['path']}) is empty. Write it before "
            f"selecting it in a config."
        )

    declared = set(entry.get("input_variables", []))
    found = placeholders(text)

    missing = declared - found          # promised but not used by the template
    unexpected = found - declared       # used by the template but never supplied

    if unexpected:
        raise ValueError(
            f"Prompt {prompt_id!r} ({entry['path']}) uses {sorted(unexpected)}, which "
            f"the registry does not declare. Nothing will supply these at runtime and "
            f"formatting will fail mid-conversation. Add them to input_variables, or "
            f"escape a literal brace as {{{{...}}}}."
        )
    if missing:
        raise ValueError(
            f"Prompt {prompt_id!r} ({entry['path']}) never uses {sorted(missing)}, which "
            f"the registry declares. The value would be computed and silently discarded - "
            f"most likely the prompt is ignoring your context or chat history."
        )

    return PromptSpec(
        prompt_id=prompt_id,
        path=entry["path"],
        family=entry.get("family", ""),
        version=int(entry.get("version", 0)),
        input_variables=tuple(sorted(declared)),
        status=entry.get("status", ""),
        description=entry.get("description", ""),
        evaluated=bool(entry.get("evaluated", False)),
        text=text,
    )


def load_prompt(cfg: AppConfig, prompt_id: str) -> PromptSpec:
    """Load and validate one prompt by id."""
    return _load(str(cfg.resolve(cfg.paths.prompt_registry)), prompt_id)


def as_template(spec: PromptSpec) -> PromptTemplate:
    """A LangChain PromptTemplate for a loaded spec."""
    return PromptTemplate(template=spec.text, input_variables=list(spec.input_variables))


def system_prompt(cfg: AppConfig) -> PromptTemplate:
    """The persona prompt named by `prompts.system` in the config."""
    return as_template(load_prompt(cfg, cfg.prompts.system))


def summarizer_prompt(cfg: AppConfig) -> PromptTemplate:
    return as_template(load_prompt(cfg, cfg.prompts.summarizer))


def query_expansion_prompt(cfg: AppConfig) -> PromptTemplate:
    return as_template(load_prompt(cfg, cfg.prompts.query_expansion))


def active_prompts(cfg: AppConfig) -> dict[str, PromptSpec]:
    """Every prompt this config selects, loaded and validated.

    Call once at startup so a broken or missing prompt fails immediately rather
    than partway through an eval run.
    """
    return {
        "system": load_prompt(cfg, cfg.prompts.system),
        "summarizer": load_prompt(cfg, cfg.prompts.summarizer),
        "query_expansion": load_prompt(cfg, cfg.prompts.query_expansion),
    }


def list_prompts(cfg: AppConfig) -> dict[str, dict]:
    """Raw registry contents, for CLI listings and the UI's debug panel."""
    return _read_registry(str(cfg.resolve(cfg.paths.prompt_registry)))


def clear_cache() -> None:
    """Drop cached prompts. Needed when editing a prompt in a live session."""
    _load.cache_clear()
    _read_registry.cache_clear()


__all__ = [
    "PromptSpec", "load_prompt", "as_template", "system_prompt", "summarizer_prompt",
    "query_expansion_prompt", "active_prompts", "list_prompts", "placeholders", "clear_cache",
]
