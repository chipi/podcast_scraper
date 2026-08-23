"""Rolling per-feed + cumulative assessment of a 100-ep finale run, judged AS feeds complete.
Blind A/B (Opus 4.8), top-12 production view, length-controlled, vs the 2.4 baseline.
Caches per-episode verdicts so re-runs only judge new episodes. Pulls per-feed cost from the log
and compares to the 9-ep bake-off numbers.

Usage: python rolling_assess.py <candidate_dir> <label> <run_log_path>
"""

import glob
import json
import os
import os as _os
import re
import sys
import urllib.request

ROOT = _os.environ.get("PODCAST_EVAL_ROOT", ".test_outputs/manual")
BASE = f"{ROOT}/prod-v2.4-relabel-fixed/feeds"
CAND = sys.argv[1] if len(sys.argv) > 1 else "v2.5-deepseek-100ep"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "deepseek"
LOGPATH = sys.argv[3] if len(sys.argv) > 3 else open("/tmp/.log_ds100").read().strip()
CANDF = f"{ROOT}/{CAND}/feeds"
SP = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = f"{SP}/rolling_cache_{LABEL}.json"

AK = None
for line in open(".env"):
    if line.startswith("AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY="):
        AK = line.split("=", 1)[1].strip().strip('"').strip("'")
JUDGE = "claude-opus-4-8"

# 9-ep bake-off reference (deepseek-v4-flash, vs same 2.4 baseline)
BAKEOFF = {
    "deepseek": {"sum": 8.44, "ins": 8.44, "top": 8.22, "cost": 0.0092},
    "qwen": {"sum": 8.44, "ins": 8.22, "top": 7.78, "cost": 0.0045},
}
FEEDNAME = {
    "feeds.acast.com": "acast/monetarism",
    "feeds.megaphone.fm_3581": "megaphone/investlikebest",
    "feeds.megaphone.fm_370": "megaphone/nvidia-ai",
    "feeds.megaphone.fm_755": "megaphone/nopriors",
    "feeds.npr.org": "npr/planet-money",
    "feeds.simplecast.com_2e10": "simplecast/hardfork",
    "feeds.simplecast.com_9995": "simplecast/intelligence",
    "rss.flightcast.com": "flightcast",
    "video-api.wsj.com": "wsj/the-journal",
}


def short(feed):
    for k, v in FEEDNAME.items():
        if (
            k.replace("_", "_") in feed
            or feed.startswith("rss_" + k)
            or k.split("_")[0] in feed
            and (len(k.split("_")) == 1 or k.split("_")[1] in feed)
        ):
            return v
    return feed[:22]


def load(p):
    try:
        return json.loads(open(p).read())
    except Exception:
        return None


NEWRUN = "run_20260805"  # candidate's fresh relabel run; excludes the copied-in 2.4 source run


def by_nnnn(root, feed, suffix, sub="metadata", run_glob="run_*"):
    """Map NNNN -> newest matching file path under root/feed/<run_glob>/sub/."""
    out = {}
    for p in glob.glob(f"{root}/{feed}/{run_glob}/{sub}/[0-9][0-9][0-9][0-9] - *{suffix}"):
        b = os.path.basename(p)
        m = re.match(r"(\d{4})", b)
        if not m:
            continue
        k = m.group(1)
        if k not in out or os.path.getmtime(p) > os.path.getmtime(out[k]):
            out[k] = p
    return out


def extract(gi_path, meta_path):
    m = load(meta_path) or {}
    g = load(gi_path) or {}
    summ = m.get("summary")
    if isinstance(summ, str):
        try:
            summ = json.loads(summ)
        except Exception:
            pass
    if isinstance(summ, dict):
        summ = (summ.get("title", "") or "") + "\n- " + "\n- ".join(summ.get("bullets") or [])
    inodes = [n.get("properties", {}) for n in g.get("nodes", []) if n.get("type") == "Insight"]
    inodes.sort(key=lambda p: p.get("rank", 999) if isinstance(p.get("rank"), int) else 999)
    ins = [p.get("text") or p.get("label") for p in inodes[:12]]
    ins = [i for i in ins if i]
    tops = [
        n.get("properties", {}).get("label") for n in g.get("nodes", []) if n.get("type") == "Topic"
    ]
    tops = [t for t in tops if t]
    return {"summary": str(summ)[:2500], "insights": ins, "topics": tops}


PROMPT = """You are a strict editorial judge comparing two AI systems that each
summarized and extracted insights from the SAME podcast episode. You are NOT told which is which.

Below is the episode transcript (may be truncated), then System A's and System B's outputs.

CRITICAL LENGTH-CONTROL RULE: Do NOT reward length. A longer summary or a longer insight list is
NOT better. Judge substance-per-word. If two outputs convey the same substance, the SHORTER one is
BETTER. Heavily penalize padding, filler, hedging, repetition, and low-value additions.

Score each system 1-10 on THREE dimensions, judging against the transcript:
1. SUMMARY - faithful, well-organized, captures the most important points, no fluff/hallucination.
2. INSIGHTS - genuinely significant + accurate + non-redundant claims; penalize
noise/repetition/padding.
3. TOPICS - clean canonical subject headings (good: "monetary policy"; bad: sentence fragments).

Return ONLY JSON:
{"summary":{"A":n,"B":n,"winner":"A|B|tie"},
 "insights":{"A":n,"B":n,"winner":"A|B|tie"},
 "topics":{"A":n,"B":n,"winner":"A|B|tie"},
 "overall_winner":"A|B|tie"}

=== TRANSCRIPT ===
@@TRANSCRIPT@@

=== SYSTEM A ===
SUMMARY:
@@A_SUMMARY@@
INSIGHTS (@@A_NINS@@): @@A_INSIGHTS@@
TOPICS: @@A_TOPICS@@

=== SYSTEM B ===
SUMMARY:
@@B_SUMMARY@@
INSIGHTS (@@B_NINS@@): @@B_INSIGHTS@@
TOPICS: @@B_TOPICS@@
"""


def call_judge(prompt):
    body = json.dumps(
        {"model": JUDGE, "max_tokens": 1200, "messages": [{"role": "user", "content": prompt}]}
    ).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": AK,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    txt = d["content"][0]["text"]
    mobj = re.search(r"\{.*\}", txt, re.S)
    return json.loads(mobj.group(0)) if mobj else None


def judge_ep(feed, nnnn, idx):
    b_gi = by_nnnn(BASE, feed, ".gi.json").get(nnnn)
    b_mj = by_nnnn(BASE, feed, ".metadata.json").get(nnnn)
    c_gi = by_nnnn(CANDF, feed, ".gi.json", run_glob=NEWRUN + "*").get(nnnn)
    c_mj = by_nnnn(CANDF, feed, ".metadata.json", run_glob=NEWRUN + "*").get(nnnn)
    if not (b_gi and c_gi):
        return None
    base = extract(b_gi, b_mj)
    cand = extract(c_gi, c_mj)
    txp = by_nnnn(CANDF, feed, ".adfree.txt", "transcripts").get(nnnn) or by_nnnn(
        BASE, feed, ".adfree.txt", "transcripts"
    ).get(nnnn)
    tx = open(txp).read()[:45000] if txp else ""
    cand_is_A = idx % 2 == 0
    A, B = (cand, base) if cand_is_A else (base, cand)
    repl = {
        "@@TRANSCRIPT@@": tx,
        "@@A_SUMMARY@@": A["summary"],
        "@@A_NINS@@": str(len(A["insights"])),
        "@@A_INSIGHTS@@": " | ".join(A["insights"]),
        "@@A_TOPICS@@": ", ".join(A["topics"]),
        "@@B_SUMMARY@@": B["summary"],
        "@@B_NINS@@": str(len(B["insights"])),
        "@@B_INSIGHTS@@": " | ".join(B["insights"]),
        "@@B_TOPICS@@": ", ".join(B["topics"]),
    }
    p = PROMPT
    for k, val in repl.items():
        p = p.replace(k, val)
    v = call_judge(p)
    if not v:
        return None
    out = {}
    for dim, key in (("summary", "sum"), ("insights", "ins"), ("topics", "top")):
        a, bb = v[dim]["A"], v[dim]["B"]
        cs, bs = (a, bb) if cand_is_A else (bb, a)
        w = v[dim]["winner"]
        if w == "tie":
            win = "tie"
        else:
            win = "cand" if ((w == "A") == cand_is_A) else "base"
        out[key] = {"cand": cs, "base": bs, "win": win}
    ow = v["overall_winner"]
    out["overall"] = "tie" if ow == "tie" else ("cand" if ((ow == "A") == cand_is_A) else "base")
    return out


# per-Mtok prices (in, out). LiteLLM gateway reports estimated_cost_usd=0, so estimate from tokens.
PRICE = {"deepseek": (0.09, 0.18), "qwen": (0.03, 0.13)}


def cost_by_feed():
    """Estimate cost per feed_id from token counts x model price (gateway reports $0)."""
    pin, pout = PRICE.get(LABEL, (0.0, 0.0))
    costs, eps = {}, {}
    for line in open(LOGPATH, errors="ignore"):
        if '"event_type": "llm_cost"' not in line:
            continue
        m = re.search(r"\{.*\}", line)
        if not m:
            continue
        try:
            d = json.loads(m.group(0))
        except Exception:
            continue
        fid = d.get("feed_id", "?")
        c = d.get("prompt_tokens", 0) / 1e6 * pin + d.get("completion_tokens", 0) / 1e6 * pout
        costs[fid] = costs.get(fid, 0.0) + c
        eps.setdefault(fid, set()).add(d.get("episode_id"))
    return costs, {k: len(v) for k, v in eps.items()}


# ---- main ----
cache = load(CACHE_PATH) or {}
feeds = sorted([os.path.basename(d) for d in glob.glob(CANDF + "/*") if os.path.isdir(d)])
newly = 0
for feed in feeds:
    cand_eps = sorted(by_nnnn(CANDF, feed, ".gi.json", run_glob=NEWRUN + "*").keys())
    base_eps = set(by_nnnn(BASE, feed, ".gi.json").keys())
    for i, nnnn in enumerate(cand_eps):
        key = f"{feed}#{nnnn}"
        if key in cache or nnnn not in base_eps:
            continue
        try:
            v = judge_ep(feed, nnnn, len(cache) + newly)
        except Exception as e:
            print(f"  judge err {feed[:20]}#{nnnn}: {e}")
            continue
        if v:
            cache[key] = v
            newly += 1
            json.dump(cache, open(CACHE_PATH, "w"))  # persist after each verdict (timeout-safe)

costs, cost_eps = cost_by_feed()


# aggregate per feed + cumulative
def blank():
    return {
        "n": 0,
        "sum": [0, 0, {"cand": 0, "base": 0, "tie": 0}],
        "ins": [0, 0, {"cand": 0, "base": 0, "tie": 0}],
        "top": [0, 0, {"cand": 0, "base": 0, "tie": 0}],
        "ov": {"cand": 0, "base": 0, "tie": 0},
    }


perfeed = {}
cum = blank()
for key, v in cache.items():
    feed = key.split("#")[0]
    pf = perfeed.setdefault(feed, blank())
    for agg in (pf, cum):
        agg["n"] += 1
        for k in ("sum", "ins", "top"):
            agg[k][0] += v[k]["cand"]
            agg[k][1] += v[k]["base"]
            agg[k][2][v[k]["win"]] += 1
        agg["ov"][v["overall"]] += 1


def fmt_feed(feed, a):
    n = a["n"]
    cs = "/".join(f"{a[k][0]/n:.1f}" for k in ("sum", "ins", "top"))
    bs = "/".join(f"{a[k][1]/n:.1f}" for k in ("sum", "ins", "top"))
    ov = a["ov"]
    wr = f"{ov['cand']}-{ov['base']}" + (f"-{ov['tie']}t" if ov["tie"] else "")
    # cost for this feed
    fc = ""
    for fid, c in costs.items():
        # match feed dir to feed_id by loose host match
        host = feed.replace("rss_", "").split("_")[0]
        if host and host in fid:
            e = cost_eps.get(fid, 0)
            fc = f"${c/e:.4f}/ep" if e else f"${c:.3f}"
            break
    return f"  {short(feed):26} n={n:2}  {LABEL} {cs}  vs 2.4 {bs}  win {wr:8} {fc}"


print(
    f"\n{'='*78}\nROLLING ASSESSMENT — {LABEL} vs 2.4 baseline   (judged {cum['n']}/105 eps, +{newly} new)\n{'='*78}"  # noqa: E501
)
print(f"  per-feed  ({LABEL} sum/ins/top   vs 2.4 sum/ins/top   win=overall {LABEL}-2.4)")
for feed in sorted(perfeed):
    print(fmt_feed(feed, perfeed[feed]))

n = max(1, cum["n"])
print(f"\n  CUMULATIVE ({cum['n']} eps):")
for k, lab in (("sum", "summary"), ("ins", "insights"), ("top", "topics")):
    ov = cum[k][2]
    print(
        f"    {lab:9}: {LABEL} {cum[k][0]/n:.2f}  vs 2.4 {cum[k][1]/n:.2f}   (win {ov['cand']}-{ov['base']}-{ov['tie']}t)"  # noqa: E501
    )
o = cum["ov"]
print(f"    overall  : win {o['cand']}-{o['base']}-{o['tie']}t")
tot_cost = sum(costs.values())
tot_eps = sum(cost_eps.values())
print(f"    cost     : ${tot_cost:.3f} over {tot_eps} eps = ${tot_cost/max(1, tot_eps):.4f}/ep")
ref = BAKEOFF.get(LABEL)
if ref:
    print(
        f"\n  vs 9-ep BAKE-OFF ({LABEL}): sum {ref['sum']} ins {ref['ins']} top {ref['top']} @ ${ref['cost']}/ep"  # noqa: E501
    )
    print(
        f"     100-ep so far:            sum {cum['sum'][0]/n:.2f} ins {cum['ins'][0]/n:.2f} top {cum['top'][0]/n:.2f} @ ${tot_cost/max(1, tot_eps):.4f}/ep"  # noqa: E501
    )
