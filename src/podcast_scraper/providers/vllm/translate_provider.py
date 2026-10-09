"""GemmaTranslateProvider — TranslateGemma served on the DGX (ADR-157 / RFC-124 S2.3).

A SIBLING of :class:`VLLMProvider`, not a subclass. Both talk to a vLLM endpoint through the
shared :class:`OpenAICompatibleProvider` transport — so retries, the temperature/context
self-healing, the per-request timeout bound (#1852/#1894) and the SDK plumbing are inherited,
not re-implemented — but this one serves a DIFFERENT model on a DIFFERENT port for a DIFFERENT
operation, with its own config namespace (``translate_*``) and its own telemetry identity.

TWO THINGS DIVERGE FROM EVERY OTHER PROVIDER IN THIS TREE, both forced by the model:

1. **It calls ``/v1/completions``, not ``/v1/chat/completions``.** TranslateGemma's chat template
   requires structured content — ``{type, source_lang_code, target_lang_code, text}`` — and vLLM
   strips those custom keys before the template sees them, so the chat route returns HTTP 400
   (verified against the live service; vLLM does not support custom content fields). The prompt
   is therefore rendered client-side from the prompt store and posted raw.

2. **The prompt lives in the prompt store, not in this file.** ``shared/translation/
   translategemma_v1.j2`` is the whole interface to the model, so it gets a SHA256 like every
   other prompt and that hash goes into the result metadata. A prompt paraphrased from memory
   once cost a full measurement run: the omitted sentence was "Produce only the English
   translation, without any additional explanations or commentary", and without it the model
   returned commentary ("Here's a translation that aims for accuracy and nuance: ...") that
   would have landed in ``<base>.txt`` as though somebody had spoken it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple

from ... import config
from ...languages import language_registry, normalize_language_tag
from ...prompts.store import get_prompt_metadata, render_prompt
from ..openai.openai_provider import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

#: The prompt that defines this provider's contract with the model.
PROMPT_NAME = "shared/translation/translategemma_v1"

#: ``render_prompt`` strips trailing whitespace, which every other caller wants because their
#: prompt is a chat MESSAGE. For a raw completions prompt the newline after
#: ``<start_of_turn>model`` is the generation cue the model was trained on, so it is re-added
#: here rather than by changing ``render_prompt`` for its other callers.
_GENERATION_CUE = "\n"

#: vLLM stops at the model turn's end; without this the model runs on into a new user turn.
_STOP = ["<end_of_turn>"]

#: The model card documents a **2K token** total input context. The served container is
#: configured with a larger window, which is not the same as the model supporting it: a unit
#: measured at 4,800 prompt tokens came back as a translation of its FIRST SENTENCE with
#: ``finish_reason: stop`` — a silent 99% content loss, reported as success. Unit packing budgets
#: against THIS, and :meth:`GemmaTranslateProvider.translate` refuses anything over it.
MODEL_INPUT_TOKEN_LIMIT = 2048

#: Fallback when the tokenizer is unreachable: the FEWEST characters per token measured over 141
#: real Spanish units (min 2.20, p05 2.88, median 4.06). Using the minimum makes the estimate
#: pessimistic — it over-counts tokens, so the guard refuses too eagerly rather than letting an
#: oversized unit through. Every context-overflow bug in this repo came from a chars-per-token
#: constant that was fractionally optimistic.
_PESSIMISTIC_CHARS_PER_TOKEN = 2.2

#: Tokens kept free between prompt + completion and the served context, so a prompt count that
#: is off by a few tokens cannot push a request over the edge.
CONTEXT_MARGIN_TOKENS = 32

_VLLM_DUMMY_BEARER = "EMPTY"


class TranslateServedModelMismatch(RuntimeError):
    """The endpoint serves a different model than the profile pins (ADR-143/144).

    Fail-closed: a corpus attributed to the wrong translation model cannot be distinguished from
    a correct one after the fact.
    """


class TranslationUnavailable(RuntimeError):
    """No translator is configured, so there is nothing to call."""


class GemmaTranslateProvider(OpenAICompatibleProvider):
    """Translate one unit of text with TranslateGemma over a vLLM endpoint."""

    _CONFIG_NS: str = "translate"
    _TELEMETRY_PROVIDER: str = "gemma_translate"
    _PROVIDER_LABEL: str = "TranslateGemma"

    def __init__(self, cfg: config.Config):
        super().__init__(cfg)
        # Open models do not reject a non-default temperature the way some OpenAI models do.
        self._temp_fixed_at_default = set()
        self._served_verified = False
        # (model, hash, len) -> exact token count. The same unit is budgeted then sent, and the
        # answer cannot change between them.
        self._token_count_cache: Dict[Tuple[str, int, int], int] = {}
        # The served model's context (prompt + completion), read once from /v1/models; False
        # once a lookup failed, so an unreachable endpoint is not asked again per unit.
        self._served_context: Optional[int] | bool = None

    # -- identity / auth -------------------------------------------------------------------
    def _authenticate(self, cfg: "config.Config") -> None:
        """A local vLLM bearer is optional — no required-key validation (ADR-147)."""
        return None

    def _resolve_api_key(self, cfg: "config.Config") -> Optional[str]:
        explicit: Optional[str] = getattr(cfg, "translate_api_key", None)
        if explicit:
            return explicit
        env_name: str = getattr(cfg, "translate_api_key_env", None) or "TRANSLATE_API_KEY"
        from_env = os.getenv(env_name)
        return from_env or _VLLM_DUMMY_BEARER

    def _token_kwarg(self, n: int, model: Optional[str] = None) -> Dict[str, Any]:
        return {"max_tokens": n}

    @property
    def translate_model(self) -> Optional[str]:
        return getattr(self.cfg, "translate_model", None)

    def is_configured(self) -> bool:
        """Both an endpoint AND a model, because either alone cannot translate anything.

        A DEPLOYMENT question, not a policy one — there is no flag asking whether we want to
        translate (D-41). When this is False the stage records
        `REASON_NO_TRANSLATOR`/"translator_not_configured", which names the missing thing
        instead of implying somebody switched a feature off.
        """
        return bool(getattr(self.cfg, "translate_api_base", None) and self.translate_model)

    # -- lifecycle -------------------------------------------------------------------------
    def initialize(self) -> None:
        """Verify the served model before first use, then the normal init (ADR-147 B3)."""
        if getattr(self.cfg, "translate_verify_served_model", True):
            self._verify_served_model()
        super().initialize()

    def _verify_served_model(self) -> None:
        """Mismatch raises; UNREACHABLE only warns.

        Unreachable is not a mismatch — the real call surfaces connectivity anyway, and hard
        failing here would make importing this module offline impossible.
        """
        if self._served_verified or not self.is_configured():
            return
        expected = str(self.translate_model)
        try:
            served: Set[str] = {m.id for m in self.client.models.list().data}
        except Exception as exc:  # noqa: BLE001 — unreachable != mismatch
            logger.warning(
                "translate: could not verify the served model (%s); the translation call will "
                "surface any real connectivity problem",
                type(exc).__name__,
            )
            return
        if not any(s.casefold() == expected.casefold() for s in served):
            raise TranslateServedModelMismatch(
                f"the translate endpoint serves {sorted(served)!r}, not {expected!r}. Refusing "
                "to translate: a corpus attributed to the wrong translation model cannot be "
                "distinguished from a correct one after the fact (ADR-143/144)."
            )
        self._served_verified = True

    # -- token budgeting -------------------------------------------------------------------
    def served_context_tokens(self) -> Optional[int]:
        """The served model's ``max_model_len`` (prompt + completion), or None when unknown.

        Read from the server, not from a document: ADR-157 records 8192 while the endpoint
        reported 4096 on 2026-10-08, and a request sized for the documented window was refused
        ("maximum context length is 4096 tokens") — failing the episode's translation.
        """
        if self._served_context is None:
            self._served_context = False
            try:
                for m in self.client.models.list().data:
                    if str(getattr(m, "id", "")).casefold() == str(self.translate_model).casefold():
                        n = getattr(m, "max_model_len", None)
                        if n is None:
                            n = (getattr(m, "model_extra", None) or {}).get("max_model_len")
                        if n:
                            self._served_context = int(n)
                        break
            except Exception as exc:  # noqa: BLE001 — unknown context: budget as before
                logger.warning(
                    "translate: could not read the served context length (%s)", type(exc).__name__
                )
        return self._served_context or None

    def count_tokens(self, text: str) -> Optional[int]:
        """Exact token count from vLLM's own ``POST /tokenize`` (the same override VLLMProvider
        has, against this provider's endpoint and model).

        This is the number the SERVER will use, so budgeting against it removes the guess.
        ``None`` on any failure — a tokenizer outage must degrade to the pessimistic estimate,
        never take down the stage it protects.
        """
        base = getattr(self.cfg, "translate_api_base", None)
        model = self.translate_model
        if not base or not model or not text:
            return None
        key = (str(model), hash(text), len(text))
        cached = self._token_count_cache.get(key)
        if cached is not None:
            return cached
        # /tokenize is a vLLM extension at the server ROOT, not under /v1.
        root = str(base).rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
        try:
            req = urllib.request.Request(
                f"{root}/tokenize",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._resolve_api_key(self.cfg)}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — profile URL
                count = json.loads(resp.read().decode("utf-8")).get("count")
        except Exception as exc:  # noqa: BLE001 — never fail a unit over a token count
            logger.debug("translate: /tokenize unavailable (%s)", type(exc).__name__)
            return None
        if not isinstance(count, int) or count <= 0:
            return None
        self._token_count_cache[key] = count
        return count

    def estimate_prompt_tokens(self, prompt: str) -> Tuple[int, str]:
        """``(tokens, how)`` — the server's exact count, or a pessimistic character estimate."""
        exact = self.count_tokens(prompt)
        if exact is not None:
            return exact, "tokenizer"
        return int(len(prompt) / _PESSIMISTIC_CHARS_PER_TOKEN) + 1, "estimate"

    # -- the prompt ------------------------------------------------------------------------
    def _language_name(self, code: Optional[str]) -> str:
        """The English name of a DECLARED language, or raise.

        Raising is the point: a guessed language name reaches the model as an instruction it
        follows confidently, producing fluent output nobody can tell is wrong from the artifact.
        """
        normalized = normalize_language_tag(code)
        if not normalized:
            raise ValueError(f"no usable language tag to translate from: {code!r}")
        entry = language_registry().get(normalized)
        if entry is None:
            raise ValueError(
                f"language {normalized!r} is not declared in config/languages.yaml, so its name "
                "cannot be stated to the translator — declare it there rather than guessing here"
            )
        return entry.name

    def build_prompt(self, text: str, *, source_language: str, target_language: str = "en") -> str:
        """Render the store's template. Pure, so the exact string can be pinned by a test."""
        source_code = normalize_language_tag(source_language) or ""
        target_code = normalize_language_tag(target_language) or ""
        if source_code and source_code == target_code:
            raise ValueError(
                f"refusing to translate {source_code!r} into itself — the caller decided wrongly "
                "that this unit needed translating"
            )
        rendered = render_prompt(
            PROMPT_NAME,
            source_name=self._language_name(source_language),
            source_code=source_code,
            target_name=self._language_name(target_language),
            target_code=target_code,
            # The model's own template applies `| trim` to the text; match it.
            text=text.strip(),
        )
        return rendered + _GENERATION_CUE

    # -- the operation ---------------------------------------------------------------------
    def translate(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str = "en",
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Translate one unit. See :class:`~podcast_scraper.translation.base.TranslationProvider`.

        Returns ``text: None`` for a unit that could not be translated, rather than raising:
        the caller counts failures across an episode and applies RFC-124 §5.3's completeness
        gate. Only misconfiguration and a served-model mismatch raise.
        """
        if not self.is_configured():
            raise TranslationUnavailable(
                "translate_api_base / translate_model are unset; there is no translator to call"
            )
        meta: Dict[str, Any] = {
            "provider": self._TELEMETRY_PROVIDER,
            "model": self.translate_model,
            "source_language": normalize_language_tag(source_language),
            "target_language": normalize_language_tag(target_language),
            "prompt": get_prompt_metadata(PROMPT_NAME),
        }
        if not text or not text.strip():
            return {"text": "", "metadata": {**meta, "skipped": "empty_input"}}

        self._verify_served_model()
        # Narrowed for the typed SDK: `is_configured()` above already established it is set, but
        # the property is Optional[str] and the SDK's `model` parameter is not.
        model = str(self.translate_model)
        prompt = self.build_prompt(
            text, source_language=source_language, target_language=target_language
        )
        # REFUSE AN OVERSIZED UNIT RATHER THAN MISTRANSLATING IT. The model card documents a 2K
        # input context; the served container's window is larger, so the server accepts a unit the
        # model cannot actually read and the model answers with a translation of the first
        # sentence and `finish_reason: stop`. Measured: 4,800 prompt tokens in, 32 out. That is
        # the worst failure shape available — non-empty, "successful", and 99% gone — so it is
        # caught here, before the request, where it can still be reported as a failed unit.
        prompt_tokens, how = self.estimate_prompt_tokens(prompt)
        meta["prompt_tokens_precheck"] = prompt_tokens
        meta["prompt_tokens_precheck_source"] = how
        if prompt_tokens > MODEL_INPUT_TOKEN_LIMIT:
            meta["error"] = (
                f"unit is {prompt_tokens} prompt tokens ({how}), over the model's documented "
                f"{MODEL_INPUT_TOKEN_LIMIT}-token input context; refusing to send it because the "
                "server would accept it and return a translation of the first sentence only"
            )
            logger.warning("translate: %s", meta["error"])
            return {"text": None, "metadata": meta}

        overrides = params or {}
        # Generous by default: a translation's length is bounded by its input's, and a cap that
        # truncates produces a silently short translation rather than an error.
        budget = int(overrides.get("max_tokens") or max(256, len(text) // 2))
        # ...but never past what the served context leaves after the prompt: the server refuses
        # the whole request rather than shortening it (V.6b es, 2026-10-08: 1,144 prompt + 2,953
        # requested = 4,097 > 4,096, two units failed, so did the episode's translation). A
        # translation that truly does not fit is then cut off and reported as truncated below.
        served = self.served_context_tokens()
        if served:
            room = served - prompt_tokens - CONTEXT_MARGIN_TOKENS
            if budget > room:
                meta["max_tokens_capped_from"] = budget
                logger.info(
                    "translate: output budget capped %d -> %d to fit the served context "
                    "(%d tokens, prompt %d)",
                    budget,
                    max(1, room),
                    served,
                    prompt_tokens,
                )
                budget = max(1, room)

        started = time.monotonic()
        try:
            resp = self.client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=budget,
                # Deterministic INTENT. Measured: vLLM at temperature 0 still returns different
                # text for ~9% of identical requests, so this is not a reproducibility guarantee
                # and `en_sha256` must not be read as one.
                temperature=float(overrides.get("temperature", 0.0)),
                stop=_STOP,
                timeout=self._chat_request_timeout,
            )
        except Exception as exc:  # noqa: BLE001 — a unit failure is a result, not an exception
            logger.warning("translate: unit failed (%s): %s", type(exc).__name__, exc)
            return {
                "text": None,
                "metadata": {
                    **meta,
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_s": round(time.monotonic() - started, 3),
                },
            }

        elapsed = time.monotonic() - started
        choice = resp.choices[0] if resp.choices else None
        usage = getattr(resp, "usage", None)
        finish_reason = getattr(choice, "finish_reason", None) if choice else None
        out = (getattr(choice, "text", "") or "").strip() if choice else ""
        meta.update(
            {
                "finish_reason": finish_reason,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "elapsed_s": round(elapsed, 3),
            }
        )
        self._record_translation_call(meta)

        if not out:
            # An empty string is not a translation of non-empty input; accepting it would put a
            # silently blank turn into the English render while looking successful.
            meta["error"] = "the response carried no completion text"
            return {"text": None, "metadata": meta}
        if finish_reason == "length":
            # Not a shorter translation — a wrong one. `finish_reason: stop` on an oversized unit
            # is the OTHER truncation shape and is caught by the caller's length budget, not here.
            meta["error"] = (
                f"the model was cut off at max_tokens ({budget}); the translation is truncated"
            )
            return {"text": None, "metadata": meta}
        return {"text": out, "metadata": meta}

    # -- the sentence-aligned operation ----------------------------------------------------
    def translate_unit(
        self,
        unit: Any,
        *,
        source_language: str,
        target_language: str = "en",
    ) -> Dict[str, Any]:
        """Translate one unit and align the result back to its SENTENCES (RFC-124 §5.1).

        The unit is the translation CONTEXT; the sentence is the alignment atom. The request
        sends the unit's sentences numbered and requires numbered output of the same length.

        VERIFIED, NOT ASSUMED. Measured against the live service 2026-09-30: 2-, 3- and
        5-sentence units all returned numbered output of matching length. The RFC asserted this
        and it would have been the second assumption today to go untested.

        On a length mismatch OR a refused output (``reject_translation_output``): retry once,
        then fall back to translating the unit as ONE block and mark it ``alignment: "unit"``.
        The fallback is not a failure — it costs subtitle granularity for that unit, not
        correctness — but it is recorded so a corpus can be queried for how often it happened.

        Returns ``{"sentences": [{sent_id, en_text}], "alignment": ..., "metadata": {...}}``
        with ``sentences`` empty when the unit could not be translated at all.
        """
        sentences = list(getattr(unit, "sentences", []) or [])
        base_meta: Dict[str, Any] = {
            "unit_id": getattr(unit, "unit_id", None),
            "content_key": getattr(unit, "content_key", None),
        }
        if not sentences:
            return {"sentences": [], "alignment": "empty", "metadata": base_meta}

        if getattr(unit, "oversized", False):
            # Flagged at packing: a single sentence too long to split without breaking the
            # alignment atom. Refused here rather than sent, for the same reason the token
            # guard refuses — the server would answer with its first clause and say `stop`.
            return {
                "sentences": [],
                "alignment": "failed",
                "metadata": {**base_meta, "error": "unit is oversized; refusing to send it"},
            }

        # A single-sentence unit needs no numbering: there is nothing to align.
        # A REFUSED output is retried once, on both paths. The refusals are not stable: two
        # refused El Hilo units (2026-10-09) came back clean on two re-sends each, and one refused
        # unit costs the episode its whole English set (§5.3).
        if len(sentences) == 1:
            refused: Optional[str] = None
            meta = dict(base_meta)
            for attempt in (1, 2):
                got = self.translate(
                    sentences[0].text,
                    source_language=source_language,
                    target_language=target_language,
                )
                meta = {**base_meta, **got["metadata"], "attempts": attempt}
                if got["text"] is None:
                    break
                refused = reject_translation_output(sentences[0].text, got["text"])
                if refused:
                    logger.warning(
                        "translate: REFUSING unit %s (attempt %d) — %s",
                        base_meta["unit_id"],
                        attempt,
                        refused,
                    )
                    continue
                return {
                    "sentences": [{"sent_id": sentences[0].sent_id, "en_text": got["text"]}],
                    "alignment": "sentence",
                    "metadata": meta,
                }
            if refused:
                meta["error"] = f"output refused: {refused}"
            return {"sentences": [], "alignment": "failed", "metadata": meta}

        last_meta: Dict[str, Any] = {}
        for attempt in (1, 2):
            got = self.translate(
                unit.numbered_source,
                source_language=source_language,
                target_language=target_language,
            )
            last_meta = {**base_meta, **got["metadata"], "attempts": attempt}
            if got["text"] is None:
                continue
            parsed = _parse_numbered(got["text"])
            if len(parsed) == len(sentences):
                # A commentary response CAN come back with the right number of numbered lines,
                # in which case the alignment contract passes it. Validated per sentence, so one
                # bad line cannot ride along with good ones.
                refused = next(
                    (
                        r
                        for s, pp in zip(sentences, parsed)
                        if (r := reject_translation_output(s.text, pp))
                    ),
                    None,
                )
                if refused:
                    logger.warning(
                        "translate: REFUSING unit %s (attempt %d) — %s",
                        base_meta["unit_id"],
                        attempt,
                        refused,
                    )
                    last_meta["refused"] = refused
                    continue
                return {
                    "sentences": [
                        {"sent_id": s.sent_id, "en_text": p} for s, p in zip(sentences, parsed)
                    ],
                    "alignment": "sentence",
                    "metadata": last_meta,
                }
            last_meta["alignment_mismatch"] = f"{len(parsed)} of {len(sentences)}"
            logger.info(
                "translate: unit %s returned %d lines for %d sentences (attempt %d)",
                base_meta["unit_id"],
                len(parsed),
                len(sentences),
                attempt,
            )

        # Fall back to the whole unit as one block. Correct text, coarser alignment.
        whole = self.translate(
            unit.source_text, source_language=source_language, target_language=target_language
        )
        meta = {**last_meta, **whole["metadata"], "attempts": 3}
        if whole["text"] is None:
            if last_meta.get("refused"):
                meta["error"] = f"output refused: {last_meta['refused']}"
            return {"sentences": [], "alignment": "failed", "metadata": meta}
        # THIS PATH WAS COMPLETELY UNVALIDATED, and it is where the measured commentary landed.
        # The two numbered attempts mismatch precisely when the model is not following the
        # instruction, and this fallback then accepted whatever the plain request returned as
        # that turn's speech, marked `ok`.
        rejected = reject_translation_output(unit.source_text, whole["text"])
        if rejected:
            logger.warning(
                "translate: REFUSING unit %s on the whole-unit fallback — %s",
                base_meta["unit_id"],
                rejected,
            )
            return {
                "sentences": [],
                "alignment": "failed",
                "metadata": {**meta, "error": f"output refused: {rejected}"},
            }
        return {
            "sentences": [{"sent_id": sentences[0].sent_id, "en_text": whole["text"]}],
            "alignment": "unit",
            "metadata": meta,
        }

    def _record_translation_call(self, meta: Dict[str, Any]) -> None:
        """Feed the run-level counters, the same way every other LLM operation does."""
        pm = getattr(self, "pipeline_metrics", None) or getattr(self.cfg, "pipeline_metrics", None)
        recorder = getattr(pm, "record_llm_translation_call", None)
        if not callable(recorder):
            return
        try:
            recorder(
                input_tokens=int(meta.get("prompt_tokens") or 0),
                output_tokens=int(meta.get("completion_tokens") or 0),
                # Local GPU: a measured zero, not an unmeasured None.
                cost_usd=0.0,
            )
        except Exception:  # noqa: BLE001 — telemetry never breaks the operation
            logger.debug("translate: metrics recorder failed", exc_info=True)

    def cleanup(self) -> None:
        """Nothing to release: the model is served remotely, so this process holds no GPU state.

        Present because the provider protocol requires it. Deliberately not `pass` with no
        explanation — "empty" and "unimplemented" look identical otherwise, and the next reader
        would wonder which one this is.
        """
        return None


_NUMBERED = re.compile(r"^\s*(\d+)[.)]\s*(.*)$")


#: Markers of the model answering the INSTRUCTION instead of translating, or narrating its own
#: work. Every one of these is a string the model actually produced, recorded in the arc notes
#: §9 from 174 real requests — not a guess at what it might say.
#:
#: WHY THIS HAS TO BE A GUARD RATHER THAN A PROMPT FIX. The prompt already says "Produce only the
#: English translation, without any additional explanations or commentary", and adding that line
#: is what stopped most of it. But measured after that: **1 of 47 real units (2.1%)** still came
#: back as `Here are a few options for translating the Spanish text, depending on the specific
#: context and desired emphasis: Option 1...` — 366 tokens, identical across three passes. That
#: text would land in `<base>.txt` as though somebody had spoken it, and every stage downstream
#: would treat it as speech: the summariser, GI's claims, KG's entities, the subtitle cues a
#: listener reads.
#:
#: The numbered-alignment contract catches it ONLY when the line count mismatches. The `unit`
#: fallback then accepts whatever the plain request returns, unvalidated — which is precisely
#: where commentary lands, marked `ok`.
_COMMENTARY_MARKERS: Tuple[str, ...] = (
    "here are a few options",
    "here are several options",
    "here's a translation",
    "here is a translation",
    "here's the translation",
    "i'm ready to translate",
    "i am ready to translate",
    "please provide the text",
    "please provide the spanish",
    "i cannot translate",
    "note that this translation",
    "translation note",
)

#: Recorded commentary phrases that are ALSO ordinary speech, so they count only when the same
#: output narrates translating (`_TRANSLATOR_NARRATION`). "Según el contexto" translates to
#: "depending on the context": El Hilo (2026-10-09) lost its whole English set to a faithful
#: sentence saying exactly that. "Vale, entiendo" is "Okay, I understand"; an AI researcher says
#: "as an AI...". Every recorded commentary string that carried one of these also carried a
#: narration word, so moving them here refuses nothing that was refused for a real reason.
_SPEECH_LIKE_MARKERS: Tuple[str, ...] = (
    "depending on the specific context",
    "depending on the context",
    "okay, i understand",
    "as an ai",
)
_TRANSLATOR_NARRATION = re.compile(r"\btranslat|\boptions?\b", re.IGNORECASE)

#: A numbered "Option 1:" / "Option 2:" list is the shape the commentary case took, and the
#: option text itself can read like a plausible translation — so the enumeration is the tell.
_OPTION_LIST = re.compile(r"^\s*option\s*\d+\s*[:.\)]", re.IGNORECASE | re.MULTILINE)

#: Below this ratio of output to input characters, treat the result as truncated.
#:
#: Measured: a 4,060-word unit (4,800 prompt tokens) returned **32 completion tokens** —
#: `finish_reason: stop`, HTTP 200, non-empty, its first sentence only, 99.3% of the content
#: silently gone, identical across three passes. `finish_reason == "length"` does not catch it
#: because the model said `stop`.
#:
#: 0.25, and the floor is measured rather than guessed. Over the 103 sentences of the first real
#: translation run (Pass A, 2026-09-30) the output/input character ratio was:
#:
#:     min 0.72   p05 0.81   p50 1.00   max 1.55
#:
#: So the tightest real sentence sits **2.9x above** this floor, and zero of the 103 would have
#: been refused. That margin is the point: a false refusal costs the whole episode its English
#: set (§5.3 withholds the set when any unit fails), so the floor has to sit well below anything
#: a legitimate translation produces. English renders of Spanish are roughly the same length —
#: the p50 is 1.00 — so a fifth of the input is not a terse translation, it is a truncated one.
_MIN_OUTPUT_RATIO = 0.25


def reject_translation_output(source: str, translated: Optional[str]) -> Optional[str]:
    """Why this output must not be accepted as speech, or ``None`` to accept it.

    The three shapes measured in this arc that come back HTTP 200, non-empty, `finish_reason:
    stop` — and wrong. What they have in common is that nothing downstream can tell them from
    success, which is what makes them worth a guard rather than a metric.

    Returning a reason rather than raising: a refused unit is an ordinary failed unit, and the
    §5.3 completeness gate then withholds the whole English set for the episode. That is the
    intended outcome — an episode absent from the English surfaces is recoverable, a corpus with
    fabricated speech in it is not (#876's asymmetry, applied to text instead of names).
    """
    if translated is None:
        return None  # the caller already treats a None as a failure
    body = translated.strip()
    if not body:
        return "empty output"

    low = body.lower()
    for marker in _COMMENTARY_MARKERS:
        if marker in low:
            return f"model commentary, not a translation (matched {marker!r})"
    if _TRANSLATOR_NARRATION.search(body):
        for marker in _SPEECH_LIKE_MARKERS:
            if marker in low:
                return f"model commentary, not a translation (matched {marker!r})"
    if _OPTION_LIST.search(body):
        return "model offered numbered options instead of a translation"

    src = (source or "").strip()
    if src and len(body) < len(src) * _MIN_OUTPUT_RATIO:
        return (
            f"output is {len(body)} chars for {len(src)} of input "
            f"({len(body)/len(src):.0%}) — below the {_MIN_OUTPUT_RATIO:.0%} floor, so it is "
            "truncated rather than terse"
        )
    return None


def _parse_numbered(text: str) -> List[str]:
    """Numbered lines back to a list, tolerant of what a model actually emits.

    Continuation lines are appended to the current item rather than dropped: a translation that
    wraps across lines is still one sentence, and dropping the tail would silently shorten it —
    the failure shape this whole arc keeps running into.
    """
    items: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _NUMBERED.match(line)
        if m:
            items.append(m.group(2).strip())
        elif items:
            items[-1] = (items[-1] + " " + line).strip()
    return [i for i in items if i]
