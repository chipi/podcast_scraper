"""The product name is declared in six places; they must agree.

On 2026-09-18 the app was renamed "Learning Player" -> "Close Listening" (#2118).
That change touched the web surface and nothing else, so for five days:

  * Android's launcher label still read "Learning Player" — a user-facing wrong
    name on the home screen, not an internal string
  * the playback notification hardcoded it a second time
  * `capacitor.config.ts` still declared the old name, ready to reinstate it on
    any native regeneration
  * iOS had been correct all along, so nothing looked broken from one angle

Nothing caught it because nothing asserts a launcher label. The rename was
verified by web e2e; the only native check is a build, and a build does not care
what the label says. The Tier-3 spec that sat red for four nights was the same
shape: a surface with no assertion over it.

This is that assertion. It cannot tell you the name is GOOD; it tells you every
copy says the same thing, which is the property that silently rotted.

Deliberately in the Python tier: it spans web, Android and iOS, it must run on
CI boxes with no Xcode and no Android SDK, and no single platform's test runner
can see all six files.

Capacitor does not centralise this for us — `cap sync` does not rewrite
`strings.xml` or `Info.plist`, and both are committed. So agreement is enforced
here rather than generated.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
APP = REPO_ROOT / "web" / "learning-player"


def _read(rel: str) -> str:
    return (APP / rel).read_text(encoding="utf-8")


def _declarations() -> dict[str, str]:
    """Every declared product name, keyed by where a reader would look for it."""
    out: dict[str, str] = {}

    m = re.search(r"<title>([^<]+)</title>", _read("index.html"))
    assert m, "index.html has no <title>"
    out["index.html <title>"] = m.group(1).strip()

    out["i18n app.title"] = json.loads(_read("src/i18n/locales/en.json"))["app"]["title"]

    m = re.search(r"appName:\s*'([^']+)'", _read("capacitor.config.ts"))
    assert m, "capacitor.config.ts has no appName"
    out["capacitor appName"] = m.group(1)

    strings = _read("android/app/src/main/res/values/strings.xml")
    for key in ("app_name", "title_activity_main"):
        m = re.search(rf'<string name="{key}">([^<]+)</string>', strings)
        assert m, f"strings.xml has no {key}"
        out[f"android {key}"] = m.group(1)

    plist = _read("ios/App/App/Info.plist")
    m = re.search(r"<key>CFBundleDisplayName</key>\s*<string>([^<]+)</string>", plist)
    assert m, "Info.plist has no CFBundleDisplayName"
    out["ios CFBundleDisplayName"] = m.group(1)

    return out


def test_every_surface_declares_the_same_product_name() -> None:
    declared = _declarations()
    distinct = set(declared.values())
    assert len(distinct) == 1, (
        "the product name disagrees across surfaces:\n"
        + "\n".join(f"    {where:28} {name!r}" for where, name in sorted(declared.items()))
        + "\n  A rename must touch all of them. Capacitor does not propagate the name: "
        "`cap sync` leaves strings.xml and Info.plist alone, and both are committed."
    )


def test_the_playback_notification_reads_the_resource() -> None:
    """The Android notification title must not be a second hardcoded copy.

    It was one, and that is how it drifted: `strings.xml` could be corrected
    while the notification kept showing the old name.
    """
    svc = _read("android/app/src/main/java/app/closelistening/player/PlaybackService.java")
    assert (
        "setContentTitle(getString(R.string.app_name))" in svc
    ), "PlaybackService must build its title from R.string.app_name, not a literal"
    assert not re.search(
        r'setContentTitle\("', svc
    ), "PlaybackService hardcodes a notification title — use R.string.app_name"
