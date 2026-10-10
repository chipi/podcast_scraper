"""Blind A/B: does the moment score pick better reel moments than extraction order?

Moments (operator 2026-10-10) plays an episode's strongest points back to back. Arm A is
``app_moments.pick_moments`` (the moment score). Arm B is what "top N" means without it: the
insights in the order the player already ranks them (``insights_from_gi``: salience, which ties for
most of an episode, then extraction order). Both arms use the SAME clip rule, count and spacing
(``spread_pick``), so only the ranking differs.

For each episode the two arms' moments are merged, shuffled with a fixed seed, and scored 1-5 by a
judge from a different vendor than the insight generators (prod: Qwen3-30B and DeepSeek V4 Flash),
without knowing which arm a moment came from. Claude scores every episode; Gemini re-scores the
first ``--gemini-limit`` as a cross-check that the verdict does not depend on the judge.

Local, one-off, never in CI (CI must never call a real model). Input is a directory prepared from
prod read-only: ``sample.json`` (one row per episode: ``gi``, ``meta``, ``dur_s``, ``band``) and
``corpus/<gi>`` + ``corpus/<meta>``. Real episode data: keep it under ``.test_outputs/``.

``--mode clips`` (second run, 2026-10-10) holds the moments fixed (the player's order) and varies
only the clip: A = the ``segments`` rule (starts at a sentence, covers nearby quotes, >= 12 s),
B = the first quote alone, C = B's start stretched to A's length with no sentence alignment — the
control that separates "built better" from "longer", which model judges are known to favour. Each
arm is its own prompt, so the judge never sees two versions of one moment. Needs ``segments`` (raw
``*.segments.json`` relpaths) in ``sample.json``; an episode without one falls back to quotes.

Usage:
  python scripts/eval/moments_eval.py .test_outputs/moments_eval --dry-run
  python scripts/eval/moments_eval.py .test_outputs/moments_eval [--limit N] [--gemini-limit 20]
  python scripts/eval/moments_eval.py .test_outputs/moments_eval --mode clips --gemini-limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from podcast_scraper.server.app_gi_view import insights_from_gi  # noqa: E402
from podcast_scraper.server.app_moments import (  # noqa: E402
    build_clip,
    Moment,
    moment_count,
    MomentsConfig,
    pick_moments,
    score_candidates,
    spread_pick,
)

# $/Mtok. Claude: the repo's Sonnet-tier row (config/pricing_assumptions.yaml); the newest
# Sonnet is not in that table, so this is an assumption. Gemini 2.5 Pro: same table. Reported,
# not billed.
PRICES = {"claude": (3.00, 15.00), "gemini": (1.25, 10.00)}
SEED = 20261010
QUOTE_MAX_CHARS = 700
DESCRIPTION_MAX_CHARS = 700

RUBRIC = """You are choosing moments for "Moments", a quick-listening mode of a podcast app.
It plays a few short clips from an episode back to back (each at most 30 seconds), so a
listener on the run gets the episode's best points in a few minutes instead of the whole thing.

Below are candidate moments from ONE episode. Each has the point it makes (one line) and the
transcript of the clip that would play. Score each from 1 to 5 on how much it belongs in that reel:

  5 = must be in the reel: a substantive, specific, interesting point, and the clip makes sense
      on its own
  4 = a good moment; specific and worth hearing
  3 = fine but ordinary, or the clip needs a little context
  2 = weak: generic or obvious, or the clip barely makes sense on its own
  1 = filler, an advert or housekeeping, or the clip makes no sense without context

Judge each candidate on its own merits; do not reward length. Reply with ONLY a JSON array, one
object per candidate, in any order: [{"id": <int>, "score": <1-5>}]"""


def _env_key(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def baseline_pick(artifact: dict[str, Any], duration_s: float, cfg: MomentsConfig) -> list[Moment]:
    """Arm B: the player's existing order (salience, then extraction), same clip/count/spacing."""
    by_id = {m.insight_id: m for m in score_candidates(artifact, cfg)}
    ranked = [by_id[i.id] for i in insights_from_gi(artifact) if i.id in by_id]
    count = moment_count(duration_s, len(ranked), cfg)
    return spread_pick(ranked, count, int(cfg.min_gap_seconds * 1000))


def _episode_header(meta: dict[str, Any]) -> tuple[str, str, str]:
    ep = meta.get("episode") or {}
    feed = meta.get("feed") or {}
    desc = str(ep.get("description") or "")[:DESCRIPTION_MAX_CHARS]
    return str(ep.get("title") or ""), str(feed.get("title") or ""), desc


def build_prompt(title: str, show: str, desc: str, items: list[dict[str, Any]]) -> str:
    lines = [RUBRIC, "", f"Episode: {title}", f"Show: {show}"]
    if desc:
        lines.append(f"Publisher's description: {desc}")
    lines.append("")
    for it in items:
        lines.append(f"[{it['id']}] Point: {it['text']}")
        lines.append(f"    Clip: {it['clip'][:QUOTE_MAX_CHARS]}")
    return "\n".join(lines)


def parse_scores(text: str) -> dict[int, int]:
    """The FIRST JSON array in the reply. A greedy ``\\[.*\\]`` took everything up to the last
    bracket, so a reply with any bracketed text after the array failed to parse and crashed the
    second run at episode 58 (2026-10-10)."""
    text = text.split("</think>")[-1]
    start = text.find("[")
    if start < 0:
        raise ValueError("no JSON array in reply")
    rows, _ = json.JSONDecoder().raw_decode(text[start:])
    out: dict[int, int] = {}
    for row in rows:
        sid, sc = int(row["id"]), int(row["score"])
        if 1 <= sc <= 5:
            out[sid] = sc
    return out


class Claude:
    name = "claude"

    def __init__(self, model: str) -> None:
        import anthropic

        self.model = model
        self._c = anthropic.Anthropic(api_key=_env_key("ANTHROPIC_API_KEY"))

    def ask(self, prompt: str) -> tuple[str, int, int]:
        msg = self._c.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
            timeout=180.0,
        )
        text = "".join(getattr(b, "text", "") or "" for b in msg.content)
        return text, int(msg.usage.input_tokens), int(msg.usage.output_tokens)


class Gemini:
    name = "gemini"

    def __init__(self, model: str) -> None:
        from google import genai

        self.model = model
        self._c = genai.Client(api_key=_env_key("GEMINI_API_KEY"))

    def ask(self, prompt: str) -> tuple[str, int, int]:
        r = self._c.models.generate_content(model=self.model, contents=prompt)
        u = r.usage_metadata
        if u is None:
            return r.text or "", 0, 0
        out_tokens = int((u.candidates_token_count or 0) + (u.thoughts_token_count or 0))
        return r.text or "", int(u.prompt_token_count or 0), out_tokens


class JudgmentCache:
    """Every judgment, appended as it lands, keyed by judge model + prompt. A crash resumes where it
    stopped instead of starting over, and a re-run reuses identical judgments (cost 0)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[str, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows[row["key"]] = row

    @staticmethod
    def key(model: str, prompt: str) -> str:
        import hashlib

        return hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()

    def get(self, model: str, prompt: str) -> dict[int, int] | None:
        row = self.rows.get(self.key(model, prompt))
        return {int(k): v for k, v in row["scores"].items()} if row else None

    def put(self, model: str, prompt: str, scores: dict[int, int], cost: float) -> None:
        row: dict[str, Any] = {
            "key": self.key(model, prompt),
            "model": model,
            "scores": scores,
            "cost": cost,
        }
        self.rows[row["key"]] = row
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")


CACHE: JudgmentCache | None = None


def judge(client: Any, prompt: str, n_items: int) -> tuple[dict[int, int], float]:
    if CACHE is not None:
        hit = CACHE.get(client.model, prompt)
        if hit is not None:
            return hit, 0.0
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            text, tin, tout = client.ask(prompt)
            scores = parse_scores(text)
            if len(scores) < n_items:
                raise ValueError(f"{len(scores)} of {n_items} scored")
            pin, pout = PRICES[client.name]
            cost = tin / 1e6 * pin + tout / 1e6 * pout
            if CACHE is not None:
                CACHE.put(client.model, prompt, scores, cost)
            return scores, cost
        except Exception as e:  # noqa: BLE001 — retried, then surfaced
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{client.name} failed: {last_err}")


def prepare(root: Path, limit: int | None, cfg: MomentsConfig) -> list[dict[str, Any]]:
    rows = json.loads((root / "sample.json").read_text())[: limit or None]
    episodes = []
    for row in rows:
        gi = json.loads((root / "corpus" / row["gi"]).read_text())
        meta = json.loads((root / "corpus" / row["meta"]).read_text())
        a = pick_moments(gi, row["dur_s"], cfg)
        b = baseline_pick(gi, row["dur_s"], cfg)
        union: dict[str, Moment] = {m.insight_id: m for m in a + b}
        order = list(union)
        random.Random(f"{SEED}:{row['gi']}").shuffle(order)
        items = [
            {
                "id": n + 1,
                "insight_id": iid,
                "text": union[iid].text,
                "clip": union[iid].quote_text,
                "clip_s": (union[iid].end_ms - union[iid].start_ms) / 1000,
                "in_a": iid in {m.insight_id for m in a},
                "in_b": iid in {m.insight_id for m in b},
            }
            for n, iid in enumerate(order)
        ]
        title, show, desc = _episode_header(meta)
        episodes.append(
            {
                "gi": row["gi"],
                "band": row.get("band"),
                "dur_s": row["dur_s"],
                "title": title,
                "show": show,
                "prompt": build_prompt(title, show, desc, items),
                "items": items,
                "n_a": len(a),
                "n_b": len(b),
            }
        )
    return episodes


def summarise(episodes: list[dict[str, Any]], judge_name: str) -> dict[str, Any]:
    def arm_mean(ep: dict[str, Any], arm: str) -> float | None:
        s = [
            it["scores"][judge_name]
            for it in ep["items"]
            if it[arm] and judge_name in it.get("scores", {})
        ]
        return statistics.mean(s) if s else None

    done = [e for e in episodes if all(judge_name in it.get("scores", {}) for it in e["items"])]
    a = [arm_mean(e, "in_a") for e in done]
    b = [arm_mean(e, "in_b") for e in done]
    wins = sum(1 for x, y in zip(a, b) if x is not None and y is not None and x > y)
    losses = sum(1 for x, y in zip(a, b) if x is not None and y is not None and x < y)
    by_band: dict[str, list[tuple[float, float]]] = {}
    for e, x, y in zip(done, a, b):
        if x is not None and y is not None:
            by_band.setdefault(e["band"] or "?", []).append((x, y))
    return {
        "episodes": len(done),
        "mean_a": round(statistics.mean([x for x in a if x is not None]), 3) if done else None,
        "mean_b": round(statistics.mean([y for y in b if y is not None]), 3) if done else None,
        "a_wins": wins,
        "b_wins": losses,
        "ties": len(done) - wins - losses,
        "by_band": {
            k: {
                "n": len(v),
                "mean_a": round(statistics.mean(x for x, _ in v), 3),
                "mean_b": round(statistics.mean(y for _, y in v), 3),
            }
            for k, v in sorted(by_band.items())
        },
    }


def gemini_subset(n: int, k: int) -> set[int]:
    """``k`` episode indices spread evenly over ``n`` — sample.json is ordered by length band, so
    taking the first ``k`` (the first run) judged only the shortest episodes."""
    if k <= 0:
        return set()
    if k >= n:
        return set(range(n))
    return {round(i * (n - 1) / (k - 1)) for i in range(k)} if k > 1 else {0}


def load_segments(root: Path, row: dict[str, Any]) -> list[tuple[int, int, str]] | None:
    rel = row.get("segments")
    path = root / "corpus" / rel if rel else None
    if path is None or not path.exists():
        return None
    raw = json.loads(path.read_text())
    raw = raw if isinstance(raw, list) else raw.get("segments", [])
    out = []
    for seg in raw:
        try:
            out.append(
                (int(float(seg["start"]) * 1000), int(float(seg["end"]) * 1000), seg["text"])
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out or None


def _seg_text(segs: list[tuple[int, int, str]], start: int, end: int) -> str:
    return " ".join(t.strip() for s, e, t in segs if e > start and s < end and t.strip())


CLIP_ARMS = ("A", "B", "C")


def prepare_clips(root: Path, limit: int | None, cfg: MomentsConfig) -> list[dict[str, Any]]:
    rows = json.loads((root / "sample.json").read_text())[: limit or None]
    first_quote = MomentsConfig(clip_rule="first_quote")
    episodes = []
    for row in rows:
        gi = json.loads((root / "corpus" / row["gi"]).read_text())
        meta = json.loads((root / "corpus" / row["meta"]).read_text())
        segs = load_segments(root, row)
        picks = pick_moments(gi, row["dur_s"], cfg, segs)
        quotes = {i.id: i.quotes for i in insights_from_gi(gi)}
        clips: dict[str, list[tuple[int, int, str]]] = {a: [] for a in CLIP_ARMS}
        for m in picks:
            b = build_clip(quotes[m.insight_id], None, first_quote)
            assert b is not None
            clips["A"].append((m.start_ms, m.end_ms, m.clip_text))
            clips["B"].append(b)
            c_end = b[0] + (m.end_ms - m.start_ms)
            clips["C"].append((b[0], c_end, _seg_text(segs, b[0], c_end) if segs else b[2]))
        title, show, desc = _episode_header(meta)
        arms = {}
        for arm in CLIP_ARMS:
            order = list(range(len(picks)))
            random.Random(f"{SEED}:{row['gi']}:{arm}").shuffle(order)
            items = [
                {
                    "id": n + 1,
                    "insight_id": picks[i].insight_id,
                    "text": picks[i].text,
                    "clip": clips[arm][i][2],
                    "clip_s": (clips[arm][i][1] - clips[arm][i][0]) / 1000,
                }
                for n, i in enumerate(order)
            ]
            arms[arm] = {"prompt": build_prompt(title, show, desc, items), "items": items}
        episodes.append(
            {
                "gi": row["gi"],
                "band": row.get("band"),
                "dur_s": row["dur_s"],
                "show": show,
                "has_segments": segs is not None,
                "arms": arms,
            }
        )
    return episodes


def summarise_clips(episodes: list[dict[str, Any]], judge_name: str) -> dict[str, Any]:
    def mean(ep: dict[str, Any], arm: str) -> float | None:
        v = [it.get("scores", {}).get(judge_name) for it in ep["arms"][arm]["items"]]
        v = [x for x in v if x is not None]
        return statistics.mean(v) if v else None

    done = [e for e in episodes if all(mean(e, a) is not None for a in CLIP_ARMS)]
    out: dict[str, Any] = {"episodes": len(done)}
    for arm in CLIP_ARMS:
        vals = [mean(e, arm) for e in done]
        out[f"mean_{arm}"] = round(statistics.mean(vals), 3) if done else None  # type: ignore
    for x, y in (("A", "B"), ("A", "C"), ("C", "B")):
        w = sum(1 for e in done if (mean(e, x) or 0) > (mean(e, y) or 0))
        l_ = sum(1 for e in done if (mean(e, x) or 0) < (mean(e, y) or 0))
        out[f"{x}_vs_{y}"] = {"wins": w, "losses": l_, "ties": len(done) - w - l_}
    bands: dict[str, list[dict[str, Any]]] = {}
    for e in done:
        bands.setdefault(e["band"] or "?", []).append(e)
    by_band: dict[str, dict[str, float]] = {}
    for b, es in sorted(bands.items()):
        row: dict[str, float] = {"n": len(es)}
        for arm in CLIP_ARMS:
            row[arm] = round(statistics.mean(mean(e, arm) or 0.0 for e in es), 3)
        by_band[b] = row
    out["by_band"] = by_band
    return out


def run_clips(args: argparse.Namespace) -> int:
    episodes = prepare_clips(args.root, args.limit, MomentsConfig())
    secs = {a: [it["clip_s"] for e in episodes for it in e["arms"][a]["items"]] for a in CLIP_ARMS}
    stats = {
        "episodes": len(episodes),
        "with_segments": sum(1 for e in episodes if e["has_segments"]),
        "moments": sum(len(e["arms"]["A"]["items"]) for e in episodes),
        **{f"clip_seconds_mean_{a}": round(statistics.mean(v), 1) for a, v in secs.items()},
    }
    print(json.dumps(stats, indent=1))
    if args.dry_run:
        chars = sum(len(e["arms"][a]["prompt"]) for e in episodes for a in CLIP_ARMS)
        est_in, est_out = chars / 4, stats["moments"] * 3 * 12 + len(episodes) * 3 * 300
        pin, pout = PRICES["claude"]
        print("\n--- sample prompt (arm A) ---\n" + episodes[0]["arms"]["A"]["prompt"][:2500])
        print(
            f"\nestimate (claude): ~{est_in / 1e3:.0f}k in, ~{est_out / 1e3:.0f}k out, "
            f"~${est_in / 1e6 * pin + est_out / 1e6 * pout:.2f}"
        )
        return 0
    judges: list[Any] = [Claude(args.claude_model)] + (
        [Gemini(args.gemini_model)] if args.gemini_limit else []
    )
    subset = gemini_subset(len(episodes), args.gemini_limit)
    cost = {j.name: 0.0 for j in judges}
    for n, ep in enumerate(episodes):
        for j in judges:
            if j.name == "gemini" and n not in subset:
                continue
            for arm in CLIP_ARMS:
                block = ep["arms"][arm]
                scores, c = judge(j, block["prompt"], len(block["items"]))
                cost[j.name] += c
                for it in block["items"]:
                    it.setdefault("scores", {})[j.name] = scores.get(it["id"])
        print(f"[{n + 1}/{len(episodes)}] {ep['show'][:30]} — cost so far {cost}", flush=True)
    result = {
        "stats": stats,
        "cost_usd": {k: round(v, 3) for k, v in cost.items()},
        "claude": summarise_clips(episodes, "claude"),
        "gemini": (
            summarise_clips([e for i, e in enumerate(episodes) if i in subset], "gemini")
            if len(judges) > 1
            else None
        ),
        "episodes": [
            {
                **{k: v for k, v in e.items() if k != "arms"},
                "arms": {a: {"items": b["items"]} for a, b in e["arms"].items()},
            }
            for e in episodes
        ],
    }
    out = args.root / "results_clips.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("stats", "cost_usd", "claude", "gemini")}, indent=1))
    print(f"wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("root", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gemini-limit", type=int, default=20)
    ap.add_argument("--claude-model", default="claude-sonnet-5-5")
    ap.add_argument("--gemini-model", default="gemini-2.5-pro")
    ap.add_argument(
        "--dry-run", action="store_true", help="print one prompt + a cost estimate; no calls"
    )
    ap.add_argument("--mode", choices=("ranking", "clips"), default="ranking")
    args = ap.parse_args()
    global CACHE
    CACHE = JudgmentCache(args.root / f"judgments_{args.mode}.jsonl")
    if args.mode == "clips":
        return run_clips(args)

    # The first run's arms: the score with first-quote clips vs the player's order.
    cfg = MomentsConfig(ranking="score", clip_rule="first_quote")
    episodes = prepare(args.root, args.limit, cfg)
    overlap = [
        sum(1 for it in e["items"] if it["in_a"] and it["in_b"]) / max(e["n_a"], 1)
        for e in episodes
    ]
    clip_a = [it["clip_s"] for e in episodes for it in e["items"] if it["in_a"]]
    clip_b = [it["clip_s"] for e in episodes for it in e["items"] if it["in_b"]]
    stats = {
        "episodes": len(episodes),
        "moments_per_episode_median": statistics.median(e["n_a"] for e in episodes),
        "items_judged": sum(len(e["items"]) for e in episodes),
        "overlap_a_b_mean": round(statistics.mean(overlap), 3),
        "clip_seconds_mean_a": round(statistics.mean(clip_a), 1),
        "clip_seconds_mean_b": round(statistics.mean(clip_b), 1),
    }
    print(json.dumps(stats, indent=1))
    if args.dry_run:
        chars = sum(len(e["prompt"]) for e in episodes)
        est_in = chars / 4
        est_out = stats["items_judged"] * 12 + len(episodes) * 300
        pin, pout = PRICES["claude"]
        print("\n--- sample prompt ---\n" + episodes[0]["prompt"][:3000])
        print(
            f"\nestimate (claude): ~{est_in / 1e3:.0f}k in, ~{est_out / 1e3:.0f}k out, "
            f"~${est_in / 1e6 * pin + est_out / 1e6 * pout:.2f}"
        )
        return 0

    judges: list[Any] = [Claude(args.claude_model)]
    if args.gemini_limit > 0:
        judges.append(Gemini(args.gemini_model))
    cost = {j.name: 0.0 for j in judges}
    subset = gemini_subset(len(episodes), args.gemini_limit)
    for n, ep in enumerate(episodes):
        for j in judges:
            if j.name == "gemini" and n not in subset:
                continue
            scores, c = judge(j, ep["prompt"], len(ep["items"]))
            cost[j.name] += c
            for it in ep["items"]:
                it.setdefault("scores", {})[j.name] = scores.get(it["id"])
        print(f"[{n + 1}/{len(episodes)}] {ep['show'][:30]} — cost so far {cost}", flush=True)

    result = {
        "stats": stats,
        "cost_usd": {k: round(v, 3) for k, v in cost.items()},
        "claude": summarise(episodes, "claude"),
        "gemini": (
            summarise([e for i, e in enumerate(episodes) if i in subset], "gemini")
            if len(judges) > 1
            else None
        ),
        "episodes": [{k: v for k, v in e.items() if k != "prompt"} for e in episodes],
    }
    out = args.root / "results.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("stats", "cost_usd", "claude", "gemini")}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
