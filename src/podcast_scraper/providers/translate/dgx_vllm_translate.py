"""TranslateGemma on the DGX, called through ``/v1/completions`` (ADR-156 / RFC-124 S2.3).

WHY NOT THE CHAT ROUTE. ``/v1/chat/completions`` is unusable with this model. It rejects even the
exact structured content the model's own ``chat_template.jinja`` documents
(``content=[{type, source_lang_code, target_lang_code, text}]``): vLLM transforms the content list
before the template sees it, and the template's ``content | length != 1`` guard then fires. So the
client renders the prompt itself and posts it to ``/v1/completions``. Measured on the deployed
service, 2026-09-29; see ADR-156 §3.

WHY THE LANGUAGE NAMES COME FROM OUR REGISTRY. The model's own map (hundreds of regional subtags,
inside ``chat_template.jinja`` in the snapshot) would have to be copied in to be used. Instead the
prompt names the language from ``config/languages.yaml``, which already carries the English name
of every language we have declared. The consequence is the useful part: a language we have not
declared cannot be translated at all, rather than being handed to the model under a guessed name.
Naming the wrong source language to a translator does not produce an error — it produces fluent,
confident, wrong output, which is the failure mode this whole arc exists to avoid.

WHAT THIS CLIENT DOES NOT DO. It does not touch speaker labels. Labels are carried onto the
English line verbatim by the caller (D-24 / S2.6) and never sent through the model, which would
rename the same person inconsistently between units. The client translates the text it is given.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ...languages import language_registry, normalize_language_tag

logger = logging.getLogger(__name__)

#: The prompt TranslateGemma expects, rendered by us because the chat route cannot deliver it.
#: Reproduced from the model's ``chat_template.jinja`` (ADR-156 §3) — keep byte-identical.
_PROMPT_TEMPLATE = (
    "<start_of_turn>user\n"
    "You are a professional {source_name} ({source_code}) to {target_name} ({target_code}) "
    "translator. Your goal is to accurately convey the meaning and nuance of the original text.\n"
    "\n"
    "{text}<end_of_turn>\n"
    "<start_of_turn>model\n"
)

#: vLLM stops at the model turn's end; without this the model continues into a new user turn.
_STOP = ["<end_of_turn>"]

DEFAULT_TIMEOUT_S = 300
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_CONCURRENCY = 4


class TranslateError(RuntimeError):
    """The translator was reached and could not produce a usable translation."""


class TranslateUnavailable(TranslateError):
    """The translator could not be reached, or serves a different model than the profile pins."""


@dataclass
class TranslateResult:
    """One unit's outcome. ``text is None`` means THIS UNIT failed.

    IT SAYS NOTHING ABOUT THE EPISODE, deliberately. An earlier version of this docstring said
    "the episode need not fail" — a policy this class cannot see enough to set, and one RFC-124
    §5.3 decides the other way: a non-English episode without a COMPLETE English set skips
    summary, GI and KG. The client reports per-unit facts; the caller owns the episode.
    """

    text: Optional[str]
    attempts: int = 0
    error: Optional[str] = None
    elapsed_s: float = 0.0
    #: ``length`` means the model was cut off at ``max_tokens``. Recorded because a truncated
    #: translation is the dangerous case: it is not empty, so it looks like a success.
    finish_reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.text is not None


def _language_name(code: Optional[str]) -> str:
    """The English name of a DECLARED language, or raise.

    Raising is the point. A guessed language name reaches the model as an instruction it will
    follow confidently — "translate this Portuguese text" over Spanish input yields fluent output
    nobody can tell is wrong from the artifact alone.
    """
    normalized = normalize_language_tag(code)
    if not normalized:
        raise TranslateError(f"no usable language tag to translate from: {code!r}")
    entry = language_registry().get(normalized)
    if entry is None:
        raise TranslateError(
            f"language {normalized!r} is not declared in config/languages.yaml, so its name "
            "cannot be stated to the translator — add it there rather than guessing one here"
        )
    return entry.name


def render_translate_prompt(text: str, *, source_language: str, target_language: str = "en") -> str:
    """The exact prompt string to POST. Pure, so the format can be pinned by a test."""
    source_code = normalize_language_tag(source_language) or ""
    target_code = normalize_language_tag(target_language) or ""
    if source_code and source_code == target_code:
        raise TranslateError(
            f"refusing to translate {source_code!r} into itself — the caller decided wrongly "
            "that this unit needed translating"
        )
    return _PROMPT_TEMPLATE.format(
        source_name=_language_name(source_language),
        source_code=source_code,
        target_name=_language_name(target_language),
        target_code=target_code,
        text=text,
    )


class DgxVllmTranslateClient:
    """Calls the co-resident TranslateGemma service (``:8005`` by default).

    Co-resident with the summary model rather than swapping against it (ADR-156 §2): translation
    runs immediately before summary for the same episode, so a swap-based arrangement would mean
    two model loads per episode.
    """

    def __init__(
        self,
        cfg: Any,
        *,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self.cfg = cfg
        self.timeout_s = timeout_s
        self.max_attempts = max(1, int(max_attempts))
        self._verified = False

    # -- configuration ---------------------------------------------------------------------
    @property
    def api_base(self) -> Optional[str]:
        return getattr(self.cfg, "translate_api_base", None)

    @property
    def model(self) -> Optional[str]:
        return getattr(self.cfg, "translate_model", None)

    def _headers(self) -> Dict[str, str]:
        key = getattr(self.cfg, "translate_api_key", None) or "EMPTY"
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def is_configured(self) -> bool:
        return bool(self.api_base and self.model)

    # -- the fail-closed served-model check ------------------------------------------------
    def verify_served_model(self) -> None:
        """Assert ``:8005`` serves the model this profile pins. Mismatch raises; unreachable warns.

        Same rule as the summary slot (ADR-143/144): a wrong model loaded on the slot must fail
        the run rather than silently produce a corpus attributed to the model we believe is
        there. Unreachable is different and only warns — the real call surfaces connectivity
        anyway, and hard-failing here would make importing this module offline impossible.
        """
        if self._verified or not self.is_configured():
            return
        url = f"{str(self.api_base).rstrip('/')}/models"
        try:
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 — profile URL
                data = json.loads(resp.read().decode("utf-8")).get("data", [])
        except Exception as exc:  # noqa: BLE001 — unreachable is not a mismatch
            logger.warning(
                "translate: could not verify the served model at %s (%s); the translation call "
                "will surface any real connectivity problem",
                url,
                type(exc).__name__,
            )
            return
        served = {e.get("id") for e in data if isinstance(e, dict)}
        if self.model not in served:
            raise TranslateUnavailable(
                f"{self.api_base} serves {sorted(s for s in served if s)!r}, not {self.model!r}. "
                "Refusing to translate: a corpus attributed to the wrong translation model "
                "cannot be distinguished from a correct one after the fact (ADR-143/144)."
            )
        self._verified = True

    # -- one unit --------------------------------------------------------------------------
    def translate(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str = "en",
        max_tokens: Optional[int] = None,
    ) -> TranslateResult:
        """Translate ONE unit. Returns a result; only misconfiguration raises.

        A transport failure is retried up to ``max_attempts`` with a linear backoff. A unit that
        still fails comes back with ``text is None`` so the caller can mark that unit failed and
        keep the episode (S2.4), rather than one hiccup costing a whole transcript.
        """
        if not self.is_configured():
            raise TranslateUnavailable(
                "translate_api_base / translate_model are unset; there is no translator to call"
            )
        if not text or not text.strip():
            return TranslateResult(text="")

        prompt = render_translate_prompt(
            text, source_language=source_language, target_language=target_language
        )
        self.verify_served_model()

        # Generous by default: the output is a translation of the input, so its length is
        # bounded by the input's, and a cap that truncates produces a SILENTLY short translation.
        budget = max_tokens if max_tokens is not None else max(256, len(text) // 2)
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": int(budget),
                # Deterministic: a translation that changes between runs makes the provenance
                # hash on every claim (S2.11) meaningless.
                "temperature": 0.0,
                "stop": _STOP,
            }
        ).encode("utf-8")

        url = f"{str(self.api_base).rstrip('/')}/completions"
        started = time.monotonic()
        last_error: Optional[str] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                req = urllib.request.Request(url, data=payload, headers=self._headers())
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
                    body = json.loads(resp.read().decode("utf-8"))
                out, finish_reason = _first_completion(body)
                if out is None:
                    last_error = "the response carried no completion text"
                elif finish_reason == "length":
                    # THE DANGEROUS CASE. A unit cut off at `max_tokens` is not empty, so every
                    # check built on "did we get text back" passes it -- and a silently short
                    # translation is worse than a missing one, because the gate that would have
                    # caught a missing unit never fires. Treated as a failure.
                    last_error = (
                        f"the model was cut off at max_tokens ({budget}); the translation is "
                        "truncated, which is not a shorter translation but a wrong one"
                    )
                else:
                    return TranslateResult(
                        text=out,
                        attempts=attempt,
                        elapsed_s=time.monotonic() - started,
                        finish_reason=finish_reason,
                    )
            except Exception as exc:  # noqa: BLE001 — every transport failure is retryable here
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_attempts:
                time.sleep(attempt)
        logger.warning(
            "translate: unit failed after %d attempts: %s", self.max_attempts, last_error
        )
        return TranslateResult(
            text=None,
            attempts=self.max_attempts,
            error=last_error,
            elapsed_s=time.monotonic() - started,
        )

    # -- a whole episode -------------------------------------------------------------------
    def translate_many(
        self,
        texts: Sequence[str],
        *,
        source_language: str,
        target_language: str = "en",
        concurrency: int = DEFAULT_CONCURRENCY,
    ) -> List[TranslateResult]:
        """Translate units with bounded concurrency, preserving input order.

        Order is preserved because a unit's position IS its identity downstream — the English
        render puts one pseudo-segment per unit and carries `unit_id` (S2.4), so a reordered
        result set would silently reattribute text to the wrong turn.

        Concurrency is bounded because the service is co-resident with the summary model on one
        GPU; an unbounded fan-out would contend with the very stage that runs next.
        """
        if not texts:
            return []
        workers = max(1, min(int(concurrency), len(texts)))
        if workers == 1:
            return [
                self.translate(t, source_language=source_language, target_language=target_language)
                for t in texts
            ]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="translate") as pool:
            return list(
                pool.map(
                    lambda t: self.translate(
                        t, source_language=source_language, target_language=target_language
                    ),
                    texts,
                )
            )


def _first_completion(body: Any) -> tuple[Optional[str], Optional[str]]:
    """``(text, finish_reason)`` from a ``/v1/completions`` response.

    ``text`` is ``None`` when there is nothing usable. An empty string is NOT a translation of
    non-empty input, so it is treated as absent — otherwise a silently empty unit would look
    like a successful one. ``finish_reason`` is returned rather than judged here so the caller
    decides what ``length`` means; this function only reports what the server said.
    """
    if not isinstance(body, dict):
        return None, None
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, None
    first = choices[0]
    if not isinstance(first, dict):
        return None, None
    reason = first.get("finish_reason")
    reason = reason if isinstance(reason, str) else None
    text = first.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, reason
    return text.strip(), reason
