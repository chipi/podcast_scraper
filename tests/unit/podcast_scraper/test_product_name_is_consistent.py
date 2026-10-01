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
    declared = m.group(1).strip()
    # Info.plist may name a BUILD SETTING rather than the literal, because Debug and Release ship
    # different display names ("CL Dev" vs the product name) so that a local build sits beside the
    # TestFlight one instead of replacing it. Resolve it against Release — the shipped build is the
    # one whose name has to agree with every other surface.
    out["ios CFBundleDisplayName"] = _resolve_ios_setting(declared, configuration="Release")

    return out


_SETTING_REF = re.compile(r"^\$\((?P<name>[A-Z0-9_]+)\)$")


def _resolve_ios_setting(value: str, *, configuration: str) -> str:
    """``value`` as that Xcode configuration resolves it; unchanged when it is already literal."""
    ref = _SETTING_REF.match(value)
    if not ref:
        return value
    setting = ref.group("name")
    found = _build_setting(setting, configuration=configuration)
    assert found is not None, (
        f"Info.plist asks for $({setting}) but no XCBuildConfiguration named {configuration!r} "
        f"defines it — the shipped display name would be EMPTY."
    )
    return found


def _build_setting(setting: str, *, configuration: str) -> str | None:
    """A build setting's value in one configuration of the App target, or None.

    Parsed rather than read with a single regex because `project.pbxproj` holds several
    `XCBuildConfiguration` blocks per name — the project-level one and the target-level one — and
    only the target defines this. Taking the first match would read whichever Xcode wrote first.
    """
    pbx = _read("ios/App/App.xcodeproj/project.pbxproj")
    for block in pbx.split("isa = XCBuildConfiguration;")[1:]:
        head = block.split("/* End XCBuildConfiguration section */")[0]
        if not re.search(rf"name = {re.escape(configuration)};", head):
            continue
        m = re.search(rf'{re.escape(setting)} = "?([^";\n]+)"?;', head)
        if m:
            return m.group(1).strip()
    return None


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


def test_the_ios_debug_build_does_not_claim_the_shipped_name() -> None:
    """Debug must be distinguishable from the store build on a home screen holding both.

    iOS identifies an app by bundle id alone, so the two coexist only because Debug has its own
    (`app.closelistening.player.dev`). Once they do, two icons both labelled with the product name
    are indistinguishable — which is the state this guards against. Giving Debug the shipped name
    back would not fail any build; it would just make the operator uninstall the wrong one.
    """
    release = _build_setting("APP_DISPLAY_NAME", configuration="Release")
    debug = _build_setting("APP_DISPLAY_NAME", configuration="Debug")
    assert release and debug, "both configurations must define APP_DISPLAY_NAME"
    assert debug != release, (
        f"Debug and Release both display {release!r}. Debug ships a separate bundle id, so both "
        "apps can be installed at once and there would be no way to tell them apart."
    )

    release_id = _build_setting("PRODUCT_BUNDLE_IDENTIFIER", configuration="Release")
    debug_id = _build_setting("PRODUCT_BUNDLE_IDENTIFIER", configuration="Debug")
    assert release_id and debug_id, "both configurations must define PRODUCT_BUNDLE_IDENTIFIER"
    assert debug_id != release_id, (
        f"Debug and Release share the bundle id {release_id!r}, so installing one REPLACES the "
        "other on the device — the operator hit exactly this with a TestFlight build (2026-09-30)."
    )
