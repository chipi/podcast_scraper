#!/usr/bin/env python3
"""llms.txt: the agent-facing table of contents of this repo's docs (llmstxt.org format).

    python3 scripts/llms_txt.py           print what llms.txt would be
    python3 scripts/llms_txt.py --write   write llms.txt
    python3 scripts/llms_txt.py --check   exit 1 when llms.txt is stale, a link or an anchor
                                          does not resolve, or the file is over the size limit

The hand-kept part (the summary, notes, sections, and the maps by area and by service) is
llms.config.json at the repo root; every listed doc's name and one-line purpose come from the doc
itself (its first heading and the first sentence of its first paragraph, with its **Status:** when
it has one). The same script serves agentic-ai-homelab, flotilla and podcast_scraper; only the
config differs. Stdlib only. Copied from agentic-ai-homelab 74f61f9 and reformatted to this repo's
line length; the behaviour is the same.

Config: {"title", "summary", "notes": [str], "max_bytes", "purposes": {path: note},
"sections": [section]}; "purposes" replaces a globbed doc's generated purpose where its first
sentence says too little. A section is
  {"name", "sections_of": [paths], "section_notes"?}   one line per `## ` heading of each, linked
                                                      by anchor; section_notes false: the heading
  {"name", "glob": [patterns], "exclude": [patterns]}  one line per matching doc; a .yaml routine
                                                      or fleet spec is named fleet/routine, its
                                                      purpose its description
  {"name", "items": [{"name", "path", "note"?}]}      hand-kept lines; no note: the doc's purpose
A note may hold further markdown links; --check resolves those too.
"""

import fnmatch
import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "llms.config.json")
OUT = os.path.join(ROOT, "llms.txt")
PURPOSE_MAX = 120
META = re.compile(r"^(\*\*|>|\||-{3,}|<!--|!\[|!!!|\?\?\?|#|```|\s*[-*] |\d+\. |→|✅|❌|⚠)")
LINK = re.compile(r"\[([^\]]*)\]\((<[^>]+>|[^)\s]+)\)")


def read(path):
    try:
        with open(os.path.join(ROOT, path), encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise SystemExit(f"llms.txt: {path} listed in llms.config.json: {e.strerror}")
    except UnicodeDecodeError:
        raise SystemExit(f"llms.txt: {path} is not UTF-8 text")


def plain(s):
    """Markdown inline to plain text: links keep their text, emphasis markers go."""
    s = LINK.sub(r"\1", s)
    s = re.sub(r"\*\*|__", "", s)
    return re.sub(r"\s+", " ", s).strip()


def sentence(s):
    s = plain(s)
    m = re.search(r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bvs\.)(?<=[.!?])\s+(?=[A-Z`(\[])", s)
    if m:
        s = s[: m.start()]
    if len(s) > PURPOSE_MAX:
        s = s[:PURPOSE_MAX].rsplit(" ", 1)[0].rstrip(",;:") + " …"
    if s.endswith(":"):  # a line that introduces a code block or a list says nothing on its own
        return ""
    return s


def without_code(text):
    out, fence = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        out.append("" if fence else line)
    return out


def yaml_field(lines, key):
    """A top-level scalar or folded (>-) field of a simple YAML file, as one line; no YAML
    parser needed."""
    for i, line in enumerate(lines):
        if line.startswith(key + ":"):
            v = line[len(key) + 1 :].strip()
            if v not in (">-", ">", "|", "|-"):
                return v.strip("'\"")
            out = []
            for nxt in lines[i + 1 :]:
                if nxt and not nxt.startswith(" "):
                    break
                out.append(nxt.strip())
            return " ".join(x for x in out if x)
    return ""


def is_yaml(path):
    return path.endswith((".yaml", ".yml"))


def title_of(path, lines):
    if is_yaml(path):  # a routine or fleet spec: fleet/routine
        fleet, routine = yaml_field(lines, "fleet"), yaml_field(lines, "routine")
        return (
            f"{fleet}/{routine}"
            if routine
            else f"{fleet} (fleet)" if fleet else os.path.basename(path)
        )
    for line in lines:
        if line.startswith("# "):
            return plain(line[2:]).strip()
    return os.path.basename(path)


def status_of(lines):
    """The doc's status, short: the words before the first parenthesis, dash, comma or full stop
    ("Accepted (operator, 2026-11-08). Amended by …" -> "Accepted"); a leading
    "> **Retired …**" is one.
    """
    for line in lines[:40]:
        t = line.strip()
        m = re.match(r"^\*\*Status:?\*\*:?\s*(.+)$", t) or re.match(
            r"^>\s*\*\*((?:Retired|Superseded)\b.*)$", t
        )
        if m:
            st = (
                re.split(r" \(| — | – |[,;:]|\. ", plain(m.group(1)), maxsplit=1)[0]
                .strip()
                .rstrip(".")
            )
            if len(st) > 40:
                st = st[:40].rsplit(" ", 1)[0] + " …"
            return st or "see the doc"
    return ""


def first_paragraph(lines, start=0):
    """The first prose paragraph from line `start`. A block (lines up to a blank one) that
    starts with
    metadata, a list, a table or a quote is skipped whole, so its wrapped lines are never read
    as prose.
    """
    para, seen_h1, skipping = [], start > 0, False
    for line in lines[start:]:
        if not seen_h1:
            seen_h1 = line.startswith("# ")
            continue
        if line.startswith("#") and para:
            break
        if not line.strip():
            if para:
                break
            skipping = False
            continue
        if skipping:
            continue
        if META.match(line):
            if para:
                break
            skipping = True
            continue
        para.append(line.strip())
    return " ".join(para)


def purpose(path):
    lines = without_code(read(path))
    if is_yaml(path):
        return sentence(yaml_field(lines, "description"))
    first = first_paragraph(lines) or ""
    st = status_of(lines)
    text = sentence(first) if first else ""
    if st:
        return f"{st}. {text}".strip() if text else st
    return text


def slug(text, used):
    """GitHub's heading anchor: link text kept, lowercased, every character that is not a word
    character,
    "-" or a space dropped (so "*" and "`" go, "_" stays), spaces to "-", "-1", "-2" for repeats."""
    s = re.sub(r"\s+", " ", LINK.sub(r"\1", text)).strip().lower()
    s = re.sub(r"[^\w\- ]", "", s, flags=re.UNICODE).replace(" ", "-")
    n = used.get(s, 0)
    used[s] = n + 1
    return s if n == 0 else f"{s}-{n}"


def anchors(path):
    used, out = {}, set()
    for line in without_code(read(path)):
        m = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if m:
            out.add(slug(m.group(2), used))
    return out


def section_lines(path, notes=True):
    lines, used, out = without_code(read(path)), {}, []
    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not m:
            continue
        a = slug(m.group(2), used)
        if m.group(1) != "##":
            continue
        body = []
        for nxt in lines[i + 1 :]:
            if nxt.startswith("#"):
                break
            if not nxt.strip():
                if body:
                    break
                continue
            if META.match(nxt) and not body:
                if nxt.lstrip().startswith(("- ", "* ")):
                    body.append(nxt.strip()[2:])
                    break
                continue
            body.append(nxt.strip())
        note = sentence(" ".join(body)) if body and notes else ""
        name = plain(m.group(2)).split(" (", 1)[
            0
        ]  # "9. LiteLLM (mini `:4001` · …)" -> "9. LiteLLM"
        out.append(f"- [{name}]({path}#{a})" + (f": {note}" if note else ""))
    return out


def doc_line(path, name=None, note=None):
    lines = without_code(read(path))
    name = name or title_of(path, lines)
    note = note if note is not None else purpose(path)
    dest = path.replace(" ", "%20")  # a space ends a markdown link destination
    return f"- [{name}]({dest})" + (f": {note}" if note else "")


def tracked():
    """The files git tracks (None outside a git checkout): globs list only these, so an
    untracked local
    README never makes this machine's llms.txt differ from CI's checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", ROOT, "ls-files", "-z"], capture_output=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return set(out.decode().split("\0"))


def expand(patterns, exclude):
    found, git = set(), tracked()
    for p in patterns:
        for f in glob.glob(os.path.join(ROOT, p), recursive=True):
            rel = os.path.relpath(f, ROOT)
            if (
                os.path.isfile(f)
                and (git is None or rel in git)
                and not any(fnmatch.fnmatch(rel, x) for x in exclude)
            ):
                found.add(rel)
    return sorted(found)


def build(cfg):
    out = [f"# {cfg['title']}", "", f"> {cfg['summary']}", ""]
    for note in cfg.get("notes", []):
        out += [note, ""]
    # each doc once: a hand-kept item wins over a glob, and an earlier glob over a later one
    listed = {it["path"] for sec in cfg["sections"] for it in sec.get("items", [])}
    for sec in cfg["sections"]:
        out.append(f"## {sec['name']}")
        out.append("")
        for path in sec.get("sections_of", []):
            out += section_lines(path, sec.get("section_notes", True))
        if "glob" in sec:
            files = expand(sec["glob"], sec.get("exclude", []))
            if not files:
                raise SystemExit(f"llms.txt: section {sec['name']!r} matches no file")
            files = [f for f in files if f not in listed]
            listed.update(files)
            out += [doc_line(f, note=cfg.get("purposes", {}).get(f)) for f in files]
        for it in sec.get("items", []):
            out.append(doc_line(it["path"], it.get("name"), it.get("note")))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def problems(text, cfg):
    bad, cache = [], {}
    for _, target in LINK.findall(text):
        if re.match(r"^[a-z]+:", target):
            continue
        path, _, anchor = target.strip("<>").replace("%20", " ").partition("#")
        full = os.path.normpath(os.path.join(ROOT, path))
        if not full.startswith(ROOT + os.sep) or not os.path.exists(full):
            bad.append(f"no such file: {target}")
            continue
        if anchor and path.endswith(".md"):
            if path not in cache:
                cache[path] = anchors(path)
            if anchor not in cache[path]:
                bad.append(f"no such heading: {target}")
    size = len(text.encode("utf-8"))
    if size > cfg.get("max_bytes", 30000):
        bad.append(f"llms.txt is {size} bytes, over {cfg.get('max_bytes', 30000)}")
    return bad


def main():
    with open(CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    text = build(cfg)
    if "--write" in sys.argv:
        with open(OUT, "w", encoding="utf-8") as f:
            f.write(text)
        bad = problems(text, cfg)
        for b in bad:
            print("  " + b)
        print(
            f"llms.txt written: {len(text.encode('utf-8'))} bytes, "
            f"{text.count(chr(10) + '- [')} links" + (f", {len(bad)} problem(s)" if bad else "")
        )
        return 1 if bad else 0
    if "--check" in sys.argv:
        bad = problems(text, cfg)
        try:
            cur = open(OUT, encoding="utf-8").read()
        except OSError:
            cur = None
        if cur != text:
            bad.insert(0, "llms.txt is stale: run python3 scripts/llms_txt.py --write")
        for b in bad:
            print("  " + b)
        print("llms.txt " + ("FAIL" if bad else "up to date"))
        return 1 if bad else 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
