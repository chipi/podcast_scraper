# flake8: noqa: E501
"""Generate v2 fixture transcripts from structured podcast specs.

This script encodes the v2 content patterns from #900 (recurring guests,
cross-episode topics, position arcs, edge-case name ambiguity) and the
commercial-segment template from #109 / RFC-059 §3 as data, then renders
plausible podcast transcripts in the same format the parser expects.

Why a generator (not hand-written):
- Spec adherence is verifiable from the data, not by re-reading prose.
- Re-runs are deterministic — same spec, same transcripts (each episode
  seeds a per-episode RNG from its podcast+episode id).
- KG/GIL/CIL signal patterns we want exercised (recurring entities,
  cross-feed topic spans, sponsor patterns) are encoded as structured
  fields, not buried in narrative.

Output: tests/fixtures/transcripts/v2/*.txt
"""

from __future__ import annotations

import random
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "tests" / "fixtures" / "transcripts" / "v2"


@dataclass
class Guest:
    name: str
    role: str  # Short professional description shown in intro
    expertise: str  # Domain phrase used in the welcome line


@dataclass
class Episode:
    ep_id: str
    title: str
    primary_guest: str  # guest name (must be in podcast.guests)
    primary_topic: str  # canonical topic phrase (used as topic: anchor)
    secondary_topics: list[str]
    sponsor_brands: list[str]  # 3 brands per episode (opening / mid-roll / closing)
    talking_points: list[str]  # Speaker-attributed claims (GIL Quote candidates)
    callbacks: list[str] = field(default_factory=list)
    # Reference to a guest or topic from a prior episode in the same podcast.
    # Each entry: "as <Name> said in episode <ep>, ..." style — exercises CIL.
    position_arc: str | None = None
    # Optional: short statement reflecting a position change for the host
    # or a recurring guest, providing a position-arc signal.


@dataclass
class Podcast:
    pod_id: str
    title: str
    domain: str
    host: str
    guests: dict[str, Guest]
    recurring_orgs: list[str]  # KG/CIL org bridges within podcast
    episodes: list[Episode]
    description: str = ""


# ===========================================================================
# Sponsor brand pool — used across episodes; each episode picks 3 from a
# podcast-specific subset so brand recurrence is non-trivial.
# Brands are also used as recurring KG/CIL org entities.
# ===========================================================================

SPONSOR_TEMPLATES_OPENING = [
    "This episode is brought to you by {brand}. {pitch} Get started at {brand_lower}.com/podcast.",
    "Today's episode is sponsored by {brand}. {pitch} Try {brand_lower}.com today.",
    "Before we dive in, a quick word from our sponsor {brand}. {pitch} Use code POD for an extended trial.",
]

SPONSOR_TEMPLATES_MIDROLL_HOST = [
    "We'll be right back after a quick word from our sponsors.",
]

SPONSOR_TEMPLATES_MIDROLL_AD = [
    "Today's episode is sponsored by {brand}. {pitch} Visit {brand_lower}.com/podcast for a free trial.",
    "This portion is brought to you by {brand}. {pitch} Use promo POD20 to get started.",
]

SPONSOR_TEMPLATES_CLOSING = [
    "Before we wrap, thanks again to our friends at {brand}. {pitch}",
    "And finally, a big thank you to our partners at {brand}. {pitch} Check out {brand_lower}.com.",
]

BRAND_PITCHES = {
    "Linear": "Linear is the issue tracker built for speed — keyboard-first, fast search, and a clean roadmap view your team actually uses.",
    "Stripe": "Stripe makes payments simple, whether you're running a marketplace or a subscription business.",
    "Figma": "Figma is the design tool that brings product, engineering, and design into one shared file.",
    "Notion": "Notion replaces the dozen tools your team is half-using with a single workspace for docs, projects, and wikis.",
    "Vanta": "Vanta automates SOC 2 and ISO compliance so you can focus on building, not on collecting screenshots.",
    "Datadog": "Datadog gives your team unified observability across logs, metrics, and traces so on-call has one place to look.",
    "PagerDuty": "PagerDuty turns noisy alerts into structured incident response with clear ownership and post-incident learning.",
    "Sentry": "Sentry catches the errors your tests didn't and puts the stack trace, breadcrumbs, and context in one view.",
    "Strava": "Strava is the home for athletes — track every ride, run, and swim, and compare segments with friends.",
    "Shopify": "Shopify is the commerce platform that powers everything from your first sale to your hundredth store.",
    "Miro": "Miro is the visual workspace where distributed teams whiteboard, plan, and run workshops.",
    "Squarespace": "Squarespace makes building a beautiful, branded website actually pleasant — no plugins required.",
    "HubSpot": "HubSpot brings your CRM, marketing, and sales tools onto one platform so customer context follows the lead.",
    "Zapier": "Zapier connects the apps your team already uses so manual hand-offs disappear from your workflow.",
    "GoPro": "GoPro captures the moments words don't reach — your next dive, ride, or run, in cinematic detail.",
    "Suunto": "Suunto builds dive computers and outdoor watches that you trust because they don't quit when conditions change.",
    "PADI": "PADI is the global standard for dive education — find a certified instructor in your area.",
    "Adobe": "Adobe Creative Cloud puts Lightroom, Photoshop, and Premiere on every device you edit on.",
    "Peak Design": "Peak Design builds camera bags, straps, and tripods designed by photographers for photographers.",
    "Vanguard": "Vanguard pioneered low-cost index funds and still keeps expense ratios where they belong.",
    "Wealthfront": "Wealthfront automates the boring parts of investing — tax-loss harvesting, rebalancing, and high-yield cash.",
    "Morningstar": "Morningstar gives you independent fund ratings and the analytical depth most platforms gloss over.",
}


# ===========================================================================
# Talking-point templates — each is a SPEAKER-ATTRIBUTED CLAIM intended to
# look like a GIL Quote candidate (sharp, opinionated, attributable).
# Variables: {topic}, {org}, {host}, {guest}.
# ===========================================================================


def render_pitch(brand: str) -> tuple[str, str]:
    pitch = BRAND_PITCHES.get(brand, f"{brand} solves real problems for real teams.")
    return pitch, brand.lower().replace(" ", "")


def opening_ad_block(host: str, brand: str, rng: random.Random) -> list[str]:
    pitch, brand_lower = render_pitch(brand)
    template = rng.choice(SPONSOR_TEMPLATES_OPENING)
    return [f"{host}: {template.format(brand=brand, brand_lower=brand_lower, pitch=pitch)}"]


def midroll_ad_block(host: str, brand: str, rng: random.Random) -> list[str]:
    pitch, brand_lower = render_pitch(brand)
    host_intro = rng.choice(SPONSOR_TEMPLATES_MIDROLL_HOST)
    ad_template = rng.choice(SPONSOR_TEMPLATES_MIDROLL_AD)
    return [
        f"{host}: {host_intro}",
        f"Ad: {ad_template.format(brand=brand, brand_lower=brand_lower, pitch=pitch)}",
        f"{host}: Welcome back to the show.",
    ]


def closing_ad_block(host: str, brand: str, rng: random.Random) -> list[str]:
    pitch, brand_lower = render_pitch(brand)
    template = rng.choice(SPONSOR_TEMPLATES_CLOSING)
    return [f"{host}: {template.format(brand=brand, brand_lower=brand_lower, pitch=pitch)}"]


# ===========================================================================
# Dialog rendering
# ===========================================================================

HOST_TRANSITIONS = [
    "Let's start there.",
    "I want to dig into that.",
    "That's a useful framing.",
    "Take me through the decision.",
    "What does that look like day-to-day?",
    "How do you balance that?",
    "Where do most teams get it wrong?",
    "Walk me through a concrete example.",
    "Push back on the conventional wisdom for me.",
    "What does the failure mode look like?",
    "How do you know when you've gotten it right?",
    "What's the one heuristic you keep coming back to?",
]

GUEST_OPENERS = [
    "Sure.",
    "Yeah, so —",
    "It comes down to this:",
    "Honestly,",
    "The short answer is",
    "I'd put it this way:",
    "Here's how I think about it.",
    "Let me give you the version I actually believe.",
    "There are two angles —",
]

GUEST_ELABORATIONS = [
    "And the second-order effect is the part most people miss.",
    "Once you internalize that, the rest of the decisions get easier.",
    "The exception proves the rule — there are edge cases, but they're rarer than people assume.",
    "I've watched this pattern hold across three different teams and the underlying dynamic doesn't change.",
    "The framing matters because it tells you which trade-offs you're actually making.",
    "If you ignore that, you end up rebuilding it from scratch a year later.",
    "What used to feel like an exception is honestly the median case now.",
    "Most of the cost shows up downstream, which is why teams keep paying it.",
]

HOST_FOLLOWUPS = [
    "That tracks.",
    "Say more about that.",
    "Where does that break down?",
    "Is there a counterexample that would change your view?",
    "That maps to what I've seen.",
    "What would you tell the version of you from five years ago?",
]


def host_line(host: str, prompt: str, rng: random.Random) -> str:
    transition = rng.choice(HOST_TRANSITIONS)
    return f"{host}: {transition} {prompt}"


def guest_line(guest: str, claim: str, rng: random.Random, supporting: str = "") -> str:
    opener = rng.choice(GUEST_OPENERS)
    elaboration = rng.choice(GUEST_ELABORATIONS)
    parts = [opener, claim]
    if supporting:
        parts.append(supporting)
    parts.append(elaboration)
    return f"{guest}: " + " ".join(parts)


def host_followup_line(host: str, rng: random.Random) -> str:
    return f"{host}: {rng.choice(HOST_FOLLOWUPS)}"


def render_episode(podcast: Podcast, ep: Episode) -> str:
    seed = abs(hash(f"{podcast.pod_id}:{ep.ep_id}")) % (2**32)
    rng = random.Random(seed)
    host = podcast.host
    guest = podcast.guests[ep.primary_guest]
    guest_name = guest.name

    # Brand assignments: opening / mid-roll / closing
    opening_brand = ep.sponsor_brands[0]
    midroll_brand = ep.sponsor_brands[1]
    closing_brand = ep.sponsor_brands[2]

    lines: list[str] = []
    lines.append(f"# {podcast.title} — Episode")
    lines.append(f"## {ep.title}")
    lines.append(f"Host: {host}")
    lines.append(f"Guest: {guest_name}")
    lines.append("")
    lines.append("[00:00]")
    # Greeting + intro
    lines.append(
        f"{host}: Welcome back to {podcast.title}. "
        f"Today we're talking about {ep.primary_topic.replace('topic:', '').replace('-', ' ')}, "
        f"and I'm joined by {guest_name}, {guest.role}. {guest_name}, thanks for being here."
    )
    lines.append(
        f"{guest_name}: Thanks, {host}. Great to be back."
        if ep.ep_id != "e01"
        else f"{guest_name}: Thanks, {host}. Excited for this one — {guest.expertise} is something I've been thinking about a lot."
    )

    # Opening ad — host-read
    lines.append("")
    lines.extend(opening_ad_block(host, opening_brand, rng))
    lines.append("")

    # Callback to prior episode (CIL recurrence signal)
    if ep.callbacks:
        callback = rng.choice(ep.callbacks)
        lines.append(f"{host}: Before we get into it — {callback}")
        lines.append(f"{guest_name}: Yeah, exactly. And it ties into what we're covering today.")

    # Position arc (one speaker's view evolves)
    if ep.position_arc:
        lines.append(f"{guest_name}: {ep.position_arc}")
        lines.append(f"{host}: That's a real shift. Worth pulling on.")

    lines.append("")
    lines.append("[03:30]")

    primary_human = ep.primary_topic.replace("topic:", "").replace("-", " ")
    secondary_humans = [t.replace("topic:", "").replace("-", " ") for t in ep.secondary_topics]

    def _render_pass(claims_subset: Iterable[str], pass_label: str) -> None:
        """Render a single pass over claims with host/guest exchanges."""
        expansion_templates = [
            "And when teams adopt {org} in that flow, the trade-off becomes visible — {elab}",
            "{org} is one of the platforms where this shows up cleanly. {elab}",
            "Outside of {org}-class tooling, the failure mode is harder to spot. {elab}",
            "I've seen this with teams running {org} and teams running everything in-house. {elab}",
        ]
        for i, claim in enumerate(claims_subset):
            prompts = [
                f"What's the most underrated piece of {primary_human}?",
                f"Where do {secondary_humans[0]} and {primary_human} intersect?",
                "What's something you used to believe about this that you've revised?",
                f"How does {rng.choice(podcast.recurring_orgs)} fit into this picture?",
                f"What's the {pass_label} angle most listeners don't have access to?",
                f"How does {secondary_humans[-1]} change the calculus here?",
                "Where do people most often confuse correlation with causation in this area?",
                "What's the version of the question that actually matters?",
            ]
            lines.append(host_line(host, rng.choice(prompts), rng))
            supporting = ""
            if i % 2 == 0 and podcast.recurring_orgs:
                org = rng.choice(podcast.recurring_orgs)
                supporting = f"We've seen this play out at teams using {org}."
            lines.append(guest_line(guest_name, claim, rng, supporting))
            # Host follow-up + guest expansion on every third turn (less spammy
            # than every-other turn, but still adds density and org mentions).
            if i % 3 == 2:
                lines.append(host_followup_line(host, rng))
                org = rng.choice(podcast.recurring_orgs) if podcast.recurring_orgs else "the team"
                elab = rng.choice(GUEST_ELABORATIONS).rstrip(".") + "."
                template = rng.choice(expansion_templates)
                lines.append(f"{guest_name}: {template.format(org=org, elab=elab)}")

    # Three passes through ALL talking points — each pass uses different
    # prompt framings ("structural" / "operational" / "contrarian") so the
    # same claims get re-surfaced and attributed multiple times. This both
    # sustains episode length (~1800-2400 words) and gives GIL more
    # opportunities to extract Quote nodes for the same speaker-claim pair.
    _render_pass(ep.talking_points, "structural")

    # Mid-roll ad (with Ad: speaker)
    lines.append("")
    lines.append("[12:00]")
    lines.extend(midroll_ad_block(host, midroll_brand, rng))
    lines.append("")
    lines.append("[14:00]")

    _render_pass(ep.talking_points, "operational")

    # Brief topical pivot to second-half secondary topic.
    lines.append("")
    lines.append("[22:00]")
    lines.append(
        f"{host}: Let's pivot — {secondary_humans[-1]} is where I want to spend the last stretch."
    )
    lines.append(
        f"{guest_name}: That's where the conversation gets interesting, because the "
        f"trade-offs around {secondary_humans[-1]} cross-cut everything we've covered so far."
    )

    _render_pass(ep.talking_points, "contrarian")

    lines.append("")
    lines.append("[28:00]")

    # Wrap-up exchange
    lines.append(f"{host}: Before we wrap, what's one thing listeners can try this week?")
    lines.append(
        f"{guest_name}: Pick one of the ideas we covered and run it for two weeks. The rest is iteration."
    )

    # Closing ad
    lines.append("")
    lines.extend(closing_ad_block(host, closing_brand, rng))
    lines.append("")

    # Outro
    lines.append(f"{host}: {guest_name}, thanks for the conversation.")
    lines.append(f"{guest_name}: Thanks, {host}.")
    lines.append(f"{host}: That's it for today's episode of {podcast.title}. See you next time.")
    return "\n".join(lines) + "\n"


# ===========================================================================
# Spec data — encodes v2 content patterns from issue #900.
# ===========================================================================

# Cross-podcast topic spans (#900 pattern 4):
#   topic:reliability appears in p02 + p05
#   topic:risk-management appears in p02 + p05
#   topic:systems-thinking appears in p02 + p05 + p07
#   topic:soil-erosion appears in p01 + p03 (mountain + marine erosion)


def build_spec() -> list[Podcast]:
    """Construct the canonical v2 podcast specs."""

    # ----- p01 — Singletrack Sessions (mountain biking) -----
    p01 = Podcast(
        pod_id="p01",
        title="Singletrack Sessions",
        domain="Mountain biking",
        host="Maya",
        description="Conversations about trail building, riding skills, and the gear that lasts.",
        guests={
            "Liam": Guest("Liam", "trail builder for the Cascadia Alliance", "trail building"),
            "Sophie": Guest("Sophie", "enduro racer and coach", "enduro skills"),
            "Noah": Guest("Noah", "mechanic at Spoke & Wrench", "drivetrain mechanics"),
        },
        recurring_orgs=["Cascadia Alliance", "Shimano", "RockShox", "Spoke & Wrench"],
        episodes=[
            Episode(
                ep_id="e01",
                title="Building Trails That Last",
                primary_guest="Liam",
                primary_topic="topic:trail-building",
                secondary_topics=["topic:soil-erosion", "topic:land-stewardship"],
                sponsor_brands=["Strava", "Stripe", "Linear"],
                talking_points=[
                    "Drainage is the single highest-leverage design choice on any trail. If water leaves on its own, the trail will last a decade.",
                    "The most common mistake new builders make is grading too aggressively — gentle is durable, steep is romantic.",
                    "Land-stewardship conversations have to happen before the first dig. Otherwise you spend the next year fighting people who feel ignored.",
                    "Bench cuts on hillsides aren't optional. The shortcut on day one is the washout on year three.",
                    "Soil structure matters more than people think — silty soil collapses under tire load, clay holds shape but pools water.",
                    "I changed my mind about machine-built versus hand-built trails. Done right, machines can produce gorgeous flow.",
                ],
            ),
            Episode(
                ep_id="e02",
                title="Enduro Skills Without the Hype",
                primary_guest="Sophie",
                primary_topic="topic:enduro-racing",
                secondary_topics=["topic:soil-erosion", "topic:risk-management"],
                sponsor_brands=["Linear", "Notion", "Vanta"],
                talking_points=[
                    "Speed comes from braking earlier and smoother, not from taking bigger risks.",
                    "On a berm, you want your eyes through the exit, hips centered, and pressure building gradually rather than spiking.",
                    "The fastest riders I race against are the ones who modify least under pressure.",
                    "Tire pressure and casing choice often matter more than a minor suspension tweak.",
                    "Risk management in racing isn't avoiding crashes — it's choosing which crashes are survivable.",
                    "Erosion patterns on race-tracks change everything about line choice from practice to race day.",
                ],
                callbacks=[
                    "Liam was on the show talking about how drainage is the single highest-leverage design choice. That ties directly into how a track holds up across race day."
                ],
                position_arc="I used to think you could brake your way out of any line mistake. After 2024 nationals I revised that — the right line in practice is everything.",
            ),
            Episode(
                ep_id="e03",
                title="The Mechanics of a Quiet, Fast Bike",
                primary_guest="Noah",
                primary_topic="topic:drivetrain-mechanics",
                secondary_topics=["topic:maintenance", "topic:risk-management"],
                sponsor_brands=["Notion", "Stripe", "Shopify"],
                talking_points=[
                    "Most mystery creaks are solved by cleaning contact surfaces, re-greasing threads, and re-torquing to spec.",
                    "A bike should feel quiet and predictable so your attention stays on the trail, not on noises or surprises.",
                    "Chain wear is the single best leading indicator of how soon you'll be replacing cassettes and chainrings.",
                    "Shimano and SRAM have converged more than people admit — choose ergonomics, not branding.",
                    "Suspension service intervals are a risk-management decision: when you defer them, you're betting on the seal holding.",
                    "I changed my mind about tubeless setups for everyday riders. The sealant complexity used to feel like a tax — now I think it pays back.",
                ],
                callbacks=[
                    "Sophie was on last episode talking about tire pressure mattering more than suspension tweaks. I want to push back on that a little from a mechanic's perspective."
                ],
            ),
        ],
    )

    # ----- p02 — Practical Systems (software engineering) -----
    # Priya recurs as guest in e01 + e03 (recurring-guest pattern).
    p02 = Podcast(
        pod_id="p02",
        title="Practical Systems",
        domain="Software engineering",
        host="Ethan",
        description="Reliability, architecture, delivery, and the tradeoffs nobody puts in the blog post.",
        guests={
            "Priya": Guest(
                "Priya",
                "principal SRE at a payments platform",
                "incident response and on-call design",
            ),
            "Jonas": Guest(
                "Jonas",
                "staff engineer focused on platform teams",
                "staff-engineer communication",
            ),
        },
        recurring_orgs=["Linear", "Datadog", "PagerDuty", "Sentry", "Notion"],
        episodes=[
            Episode(
                ep_id="e01",
                title="On-Call That Doesn't Break People",
                primary_guest="Priya",
                primary_topic="topic:on-call-rotation",
                secondary_topics=[
                    "topic:reliability",
                    "topic:incident-response",
                    "topic:systems-thinking",
                ],
                sponsor_brands=["Linear", "PagerDuty", "Sentry"],
                talking_points=[
                    "A good on-call rotation is designed so that waking up is rare, and when it happens the response is obvious.",
                    "Alerting should be action-oriented — if no one can take an immediate action, it probably shouldn't page.",
                    "Error budgets work when they change behavior, not when they are charts no one looks at during planning.",
                    "Healthy systems treat failures as expected and contain them with timeouts, retries, and clear ownership.",
                    "PagerDuty's incident response model assumes someone owns the page — that ownership is the whole game.",
                    "The hardest part of on-call isn't fixing things at 3am — it's designing the system so the fix is obvious to whoever's holding the page.",
                ],
            ),
            Episode(
                ep_id="e02",
                title="Staff-Engineer Communication Patterns",
                primary_guest="Jonas",
                primary_topic="topic:engineering-communication",
                secondary_topics=[
                    "topic:reliability",
                    "topic:incident-postmortems",
                    "topic:risk-management",
                ],
                sponsor_brands=["Notion", "Linear", "Datadog"],
                talking_points=[
                    "A great RFC starts with context and constraints, then options, then a recommendation with explicit risks.",
                    "Writing things down turns disagreement into collaboration because you can point at assumptions instead of people.",
                    "Postmortems that focus on individuals miss the system that made the mistake easy to make.",
                    "Architecture decisions are mostly tradeoffs: cost versus reliability, speed versus correctness, autonomy versus consistency.",
                    "Notion as a docs platform only works when teams treat it as the source of truth, not as a notebook.",
                    "Risk management is a communication problem. The technical work is the easier half.",
                ],
                callbacks=[
                    "Priya was on episode 1 making the case that good on-call rotations are designed so waking up is rare. I want to push that further into how we communicate the design itself.",
                    "Linear keeps coming up as our project tracker — last episode Priya talked about it for incident workflows; today we're using it differently.",
                ],
            ),
            Episode(
                ep_id="e03",
                title="Security as Design, Not a Checklist",
                primary_guest="Priya",
                primary_topic="topic:security-design",
                secondary_topics=[
                    "topic:reliability",
                    "topic:systems-thinking",
                    "topic:risk-management",
                ],
                sponsor_brands=["Vanta", "Sentry", "Linear"],
                talking_points=[
                    "Secrets handling improves when tokens are short-lived, scoped narrowly, and rotated with automation.",
                    "Threat modeling can be lightweight: identify assets, entry points, trust boundaries, and likely abuse cases.",
                    "Vanta-style compliance work and real security work overlap maybe 60% — don't confuse the two.",
                    "Authn/authz separation matters more than people think — most breaches I've seen confused who someone is with what they can do.",
                    "Security incidents are reliability incidents — Datadog can show you the spike, but only the on-call structure makes it survivable.",
                    "I used to think security teams should own all the security work. After three incidents I revised that — embedded ownership beats a central gatekeeper every time.",
                ],
                callbacks=[
                    "I was last on the show talking about on-call rotations. The pattern shows up again here — embed ownership where the work happens.",
                ],
                position_arc="I used to argue for a single central security team. I changed my mind after the 2024 webhook incident — the embedded model gets faster fixes and better signal.",
            ),
        ],
    )

    # ----- p03 — Below the Surface (scuba diving) -----
    # Two-Marcos pattern: Marco here is the dive Marco (Mediterranean wreck diver).
    # Another Marco appears in p05 (different domain) — CIL must NOT merge them.
    p03 = Podcast(
        pod_id="p03",
        title="Below the Surface",
        domain="Scuba diving",
        host="Rina",
        description="Diving conversations: technique, marine biology, and the calm that distinguishes good divers from lucky ones.",
        guests={
            "Marco": Guest(
                "Marco",
                "technical diver and underwater archaeologist",
                "wreck-diving fundamentals",
            ),
            "Hanna": Guest("Hanna", "marine biologist focused on reef systems", "marine biology"),
            "Owen": Guest("Owen", "rebreather instructor and cave diver", "calm under pressure"),
        },
        recurring_orgs=["PADI", "Suunto", "GoPro", "DAN"],
        episodes=[
            Episode(
                ep_id="e01",
                title="Wreck Diving Fundamentals",
                primary_guest="Marco",
                primary_topic="topic:wreck-diving",
                secondary_topics=["topic:dive-planning", "topic:soil-erosion"],
                sponsor_brands=["Suunto", "PADI", "GoPro"],
                talking_points=[
                    "Wreck penetration is a planning problem first, a buoyancy problem second. The dive happens at the desk.",
                    "Silt-out from poor finning technique kills more divers in wrecks than equipment failures.",
                    "Erosion on a wreck site changes the dive every season — what's open this year may be silted shut next year.",
                    "Suunto computers give you the algorithm, but the dive plan is still your responsibility.",
                    "DAN's safety stops aren't suggestions — they're the difference between getting bent and not.",
                    "I think most certifications underestimate how much pre-dive planning matters relative to in-water skill.",
                ],
            ),
            Episode(
                ep_id="e02",
                title="Marine Biology for Divers",
                primary_guest="Hanna",
                primary_topic="topic:marine-biology",
                secondary_topics=["topic:reef-conservation", "topic:soil-erosion"],
                sponsor_brands=["PADI", "GoPro", "Suunto"],
                talking_points=[
                    "Coral bleaching events are not random — they follow water temperature anomalies you can track.",
                    "Erosion in coastal areas drives sediment plumes that reduce reef light by 30-40% during runoff season.",
                    "Most reef damage from divers isn't fin contact — it's anchor strikes from boats that didn't moor.",
                    "GoPro footage has changed marine surveys — citizen scientists submitting tagged video has expanded our datasets enormously.",
                    "PADI's Project AWARE has the right shape, but it depends on individual instructor buy-in to actually move species protections.",
                    "I changed my mind on shark dives. I used to think they were extractive — now I think well-run operators are the strongest political case for protection.",
                ],
                callbacks=[
                    "Marco was on last week talking about wreck planning. The same discipline applies to reef diving — plan the dive, dive the plan."
                ],
            ),
            Episode(
                ep_id="e03",
                title="Calm Under Pressure",
                primary_guest="Owen",
                primary_topic="topic:calm-under-pressure",
                secondary_topics=["topic:rebreather-diving", "topic:risk-management"],
                sponsor_brands=["Suunto", "PADI", "GoPro"],
                talking_points=[
                    "Panic in diving is a breathing pattern before it's a thought. Train the breathing, you reduce the panic surface area.",
                    "Rebreathers reward the divers who treat the unit as a partner, not as a piece of equipment.",
                    "Risk management on technical dives is about removing single points of failure, not adding redundancy on top of bad plans.",
                    "Cave divers who survive long careers all share one trait: they call the dive earlier than their buddies want them to.",
                    "Suunto's air-integration is reliable, but I still check my SPG at every gas switch — habits beat technology.",
                    "The hardest skill in technical diving is leaving when you should leave, not staying when you could push further.",
                ],
                callbacks=[
                    "Hanna mentioned anchor strikes hurting reefs — the same operational discipline applies underwater: plan, brief, execute, debrief."
                ],
            ),
        ],
    )

    # ----- p04 — Frame & Light (photography) -----
    p04 = Podcast(
        pod_id="p04",
        title="Frame & Light",
        domain="Photography",
        host="Leo",
        description="Working photographers on lighting, location, and getting a frame that holds up at print size.",
        guests={
            "Ava": Guest("Ava", "underwater photographer", "underwater imaging"),
            "Tariq": Guest("Tariq", "documentary photographer", "documentary workflow"),
            "Elise": Guest("Elise", "commercial lighting director", "lighting decisions"),
        },
        recurring_orgs=["Adobe", "Peak Design", "Profoto", "Capture One"],
        episodes=[
            Episode(
                ep_id="e01",
                title="Underwater Images That Feel Alive",
                primary_guest="Ava",
                primary_topic="topic:underwater-photography",
                secondary_topics=["topic:frame", "topic:strobe-lighting"],
                sponsor_brands=["Adobe", "Peak Design", "Squarespace"],
                talking_points=[
                    "An underwater frame is built around backscatter management before it's about composition.",
                    "Strobe positioning at 45 degrees is the starting point, not the destination. Read the water column, then adjust.",
                    "Adobe Lightroom underwater is half about white balance and half about removing magenta from blue water.",
                    "Peak Design's housings are good, but for serious dives I still trust purpose-built brands.",
                    "Most underwater photographers I admire got there by spending hundreds of dives without a camera first.",
                    "The frame should reward the eye in the first three seconds, then keep rewarding it on the fourth look.",
                ],
            ),
            Episode(
                ep_id="e02",
                title="Documentary Workflow in the Field",
                primary_guest="Tariq",
                primary_topic="topic:documentary-photography",
                secondary_topics=["topic:frame", "topic:editing-workflow"],
                sponsor_brands=["Adobe", "Squarespace", "Peak Design"],
                talking_points=[
                    "Documentary frames carry the most weight when the photographer is closer than feels comfortable.",
                    "Editing a documentary essay is the second project — you make different work than you shot.",
                    "Adobe's Capture One alternative is real now — I shoot Capture One in the field for the tethered workflow.",
                    "Peak Design slings have changed how I move through a story — fast access, no shoulder fatigue.",
                    "The frame matters less than the relationship behind it. People trust you, then they let you see the moment.",
                    "I changed my mind on color grading documentary work. I used to push for neutral; now I lean into a deliberate palette.",
                ],
                callbacks=[
                    "Ava was on last week talking about strobe positioning underwater. The same principles — read the light, then place it — show up in field documentary work."
                ],
            ),
            Episode(
                ep_id="e03",
                title="Lighting Decisions That Save a Shoot",
                primary_guest="Elise",
                primary_topic="topic:lighting-design",
                secondary_topics=["topic:strobe-lighting", "topic:frame"],
                sponsor_brands=["Adobe", "Peak Design", "Stripe"],
                talking_points=[
                    "On location, the first decision is always: am I shaping the existing light, or building from scratch?",
                    "Two-light setups solve 80% of commercial problems. The third light is for art direction insurance.",
                    "Profoto reliability matters when the client paid for an hour of model time. Strobes that fire are non-negotiable.",
                    "Adobe Premiere is encroaching on stills workflow — I now grade in Premiere for series consistency.",
                    "Frame and light are inseparable: where the light falls dictates where the frame should go.",
                    "Color temperature mistakes cost more shoots than focus mistakes — and they're harder to fix in post.",
                ],
            ),
        ],
    )

    # ----- p05 — Long Horizon Notes (investing) -----
    # Daniel here is the investing Daniel; "Daniel" also appears in p04 as Tariq's reference.
    # Marco here is the investing Marco (different person from p03's dive Marco) — two-Marcos test.
    p05 = Podcast(
        pod_id="p05",
        title="Long Horizon Notes",
        domain="Investing",
        host="Nora",
        description="Long-term investing conversations: index investing, real estate numbers, and the risk you actually face.",
        guests={
            "Daniel": Guest(
                "Daniel", "former bond trader turned index advocate", "index investing"
            ),
            "Isabel": Guest("Isabel", "real estate underwriter", "real estate underwriting"),
            "Kasper": Guest("Kasper", "personal-finance writer focused on risk", "risk management"),
            # Distinct Marco from p03's dive Marco — CIL should keep them separate.
            "Marco": Guest("Marco", "tax-loss harvesting researcher", "tax-loss harvesting"),
        },
        recurring_orgs=["Vanguard", "Wealthfront", "Morningstar", "iShares"],
        episodes=[
            Episode(
                ep_id="e01",
                title="Index Investing Without the Myths",
                primary_guest="Daniel",
                primary_topic="topic:index-investing",
                secondary_topics=[
                    "topic:reliability",
                    "topic:risk-management",
                    "topic:systems-thinking",
                ],
                sponsor_brands=["Vanguard", "Wealthfront", "Stripe"],
                talking_points=[
                    "Index funds are not a strategy — they're the absence of one. That's their advantage.",
                    "Vanguard's structure pioneered low costs, but iShares is now competitive on most major index exposures.",
                    "Reliability in investing means having a plan you can hold through a 30% drawdown, not maximizing returns.",
                    "Risk management for individual investors is mostly behavioral, not analytical.",
                    "Systems thinking applies: the index is a system that accepts every company's failure as expected.",
                    "I changed my mind on small-value tilts. I used to think they were free lunch — now I think they're a behavioral commitment most people can't keep.",
                ],
            ),
            Episode(
                ep_id="e02",
                title="Real Estate: Numbers Before Narratives",
                primary_guest="Isabel",
                primary_topic="topic:real-estate-underwriting",
                secondary_topics=["topic:risk-management", "topic:reliability"],
                sponsor_brands=["Wealthfront", "Morningstar", "Vanguard"],
                talking_points=[
                    "Real estate underwriting is a discipline of running the numbers before believing the story.",
                    "Cap rate compression in the 2020s wasn't a market signal — it was a financing artifact.",
                    "Reliability in real estate means cash-flow positivity at conservative occupancy, not maximum leverage.",
                    "Wealthfront's automated tools don't help with real estate — you still need a spreadsheet and a skeptic.",
                    "Risk management in real estate is mostly about avoiding properties that look 'just barely' workable.",
                    "Most failed deals I've seen failed at underwriting, not at operations. The math told the story upfront.",
                ],
                callbacks=[
                    "Daniel was on episode 1 making the case for behavioral discipline in index investing. The same discipline keeps you from buying the wrong building."
                ],
            ),
            Episode(
                ep_id="e03",
                title="Risk Management for People Who Hate Spreadsheets",
                primary_guest="Kasper",
                primary_topic="topic:risk-management",
                secondary_topics=[
                    "topic:reliability",
                    "topic:systems-thinking",
                    "topic:behavioral-finance",
                ],
                sponsor_brands=["Morningstar", "Wealthfront", "Vanguard"],
                talking_points=[
                    "Risk you can articulate is risk you can manage. Risk you handwave is the risk that hurts you.",
                    "Most personal finance advice ignores systems thinking — your portfolio interacts with your career, your housing, your health.",
                    "Morningstar's analyst ratings are a useful baseline, but they're not a system for action — you have to do that part.",
                    "Reliability in personal finance is having a six-month buffer in cash and never touching it for FOMO.",
                    "The risk most investors face isn't losing money — it's panic-selling at the wrong moment.",
                    "I changed my mind on whole-life insurance. I used to think it was always wrong. After watching it work for a high-income physician, I now think it's situational.",
                ],
                position_arc="I used to write absolute rules — never whole life, always term. After enough edge cases I revised that — the right answer is 'it depends, here's how to think about it.'",
            ),
        ],
    )

    return [p01, p02, p03, p04, p05]


# ===========================================================================
# Edge-case + long-context podcasts (p06-p09) — simpler, scoped renderers.
# ===========================================================================


def render_p06_edge(host: str, ep_id: str, title: str, body_lines: list[str]) -> str:
    lines = [
        f"# Edge Cases — Episode {ep_id}",
        f"## {title}",
        f"Host: {host}",
        "",
        "[00:00]",
        f"{host}: This is an edge-case fixture used for parser and processor robustness tests.",
        "",
    ]
    lines.extend(body_lines)
    lines.append("")
    lines.append(f"{host}: End of edge-case episode {ep_id}.")
    return "\n".join(lines) + "\n"


def render_long_episode(
    podcast_id: str,
    title: str,
    host: str,
    guest: str,
    guest_role: str,
    primary_topic: str,
    secondary_topics: list[str],
    sponsor_brands: list[str],
    talking_points: list[str],
    recurring_orgs: list[str],
    target_words: int = 12000,
) -> str:
    """Render a long-context episode (p07/p08/p09) by repeating thematic blocks."""
    rng = random.Random(abs(hash(f"{podcast_id}:long")) % (2**32))
    lines: list[str] = []
    lines.append(f"# The Long View — Episode")
    lines.append(f"## {title}")
    lines.append(f"Host: {host}")
    lines.append(f"Guest: {guest}")
    lines.append("")
    lines.append("[00:00]")
    lines.append(
        f"{host}: Welcome to The Long View. Today we're exploring {primary_topic.replace('topic:', '').replace('-', ' ')} in depth with {guest}, {guest_role}."
    )
    lines.append(f"{guest}: Thanks, {host}. This is a topic I've spent decades on.")

    # Opening ad
    lines.append("")
    pitch, brand_lower = render_pitch(sponsor_brands[0])
    lines.append(
        f"{host}: This episode is brought to you by {sponsor_brands[0]}. {pitch} Get started at {brand_lower}.com/podcast."
    )

    # Body — cycle through talking points multiple times with variation
    words_so_far = 200
    timestamp_min = 5
    point_idx = 0
    midroll_count = 0
    while words_so_far < target_words:
        # Mid-roll ad roughly every 25 minutes of "show time"
        if timestamp_min > 0 and timestamp_min % 25 == 0:
            pitch, brand_lower = render_pitch(sponsor_brands[1])
            lines.append("")
            lines.append(f"[{timestamp_min:02d}:00]")
            lines.append(f"{host}: A quick word from our sponsors.")
            lines.append(
                f"Ad: Today's episode is sponsored by {sponsor_brands[1]}. {pitch} "
                f"Visit {brand_lower}.com/podcast for an extended trial."
            )
            lines.append(f"{host}: Welcome back to the show.")
            words_so_far += 100
            midroll_count += 1

        prompt_topic = rng.choice(secondary_topics).replace("topic:", "").replace("-", " ")
        primary_human = primary_topic.replace("topic:", "").replace("-", " ")
        lines.append("")
        lines.append(f"[{timestamp_min:02d}:00]")
        # Host question with topical depth
        prompts = [
            f"Let's talk about {prompt_topic}. What's your read?",
            f"Walk me through how {prompt_topic} interacts with the broader {primary_human} story.",
            f"What's the deepest piece of context on {prompt_topic} that listeners are missing?",
            f"Why is {prompt_topic} the part that most analyses get wrong?",
        ]
        lines.append(f"{host}: {rng.choice(prompts)}")
        claim = talking_points[point_idx % len(talking_points)]
        org_clause = (
            f" Teams working with {rng.choice(recurring_orgs)} run into this constantly."
            if recurring_orgs
            else ""
        )
        elaboration = rng.choice(GUEST_ELABORATIONS)
        opener = rng.choice(GUEST_OPENERS)
        lines.append(f"{guest}: {opener} {claim}{org_clause} {elaboration}")
        # Host follow-up + guest second beat to add density.
        lines.append(host_followup_line(host, rng))
        second_claim = talking_points[(point_idx + 3) % len(talking_points)]
        lines.append(
            f"{guest}: That ties to another point — {second_claim} "
            + rng.choice(GUEST_ELABORATIONS)
        )

        words_so_far += (
            len(claim.split()) + len(second_claim.split()) + len(prompts[0].split()) + 60
        )
        timestamp_min += 5
        point_idx += 1

    # Closing ad
    pitch, brand_lower = render_pitch(sponsor_brands[2])
    lines.append("")
    lines.append(
        f"{host}: Before we wrap, thanks again to our partners at {sponsor_brands[2]}. {pitch}"
    )

    # Outro
    lines.append("")
    lines.append(f"{host}: {guest}, thanks for the depth here.")
    lines.append(f"{guest}: Thank you, {host}.")
    lines.append(f"{host}: That's The Long View — see you next time.")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written: list[str] = []

    # p01..p05 — full spec
    for podcast in build_spec():
        for ep in podcast.episodes:
            text = render_episode(podcast, ep)
            out = OUT_DIR / f"{podcast.pod_id}_{ep.ep_id}.txt"
            out.write_text(text, encoding="utf-8")
            written.append(out.name)

    # p01 derivatives — keep the same compact shape v1 used (used by fast tests
    # and multi-episode parser tests). Regenerate from the v2 p01_e01 content
    # so v2 and its derivatives stay consistent.
    p01_e01_path = OUT_DIR / "p01_e01.txt"
    p01_e01_text = p01_e01_path.read_text(encoding="utf-8")
    p01_e01_lines = p01_e01_text.splitlines()

    # Fast variant: first ~60 lines of body + outro
    fast_body = p01_e01_lines[:60]
    fast_lines = (
        ["# Singletrack Sessions — Episode", "## Building Trails That Last (Fast Test)"]
        + fast_body[2:]
        + [
            "",
            "[01:00]",
            "Maya: That's it for today's episode of Singletrack Sessions. See you next time.",
            "",
        ]
    )
    (OUT_DIR / "p01_e01_fast.txt").write_text("\n".join(fast_lines) + "\n", encoding="utf-8")
    written.append("p01_e01_fast.txt")

    # Multi-episode shells (very short — just intro + outro): exercise the
    # multi-episode parser path. Five short episodes, same host.
    multi_template = textwrap.dedent("""\
        # Singletrack Sessions — Episode
        ## {title}
        [00:00]
        Maya: Welcome back to Singletrack Sessions. {opener}
        {guest}: {response}
        [00:10]
        Maya: That's it for today's episode of Singletrack Sessions. See you next time.
        """)
    multi_episodes = [
        (
            "p01_multi_e01.txt",
            "Building Trails (Multi)",
            "Today is about trail building.",
            "Liam",
            "Thanks, Maya — great to be here.",
        ),
        (
            "p01_multi_e02.txt",
            "Enduro Skills (Multi)",
            "Today is about enduro skills.",
            "Sophie",
            "Thanks for having me back.",
        ),
        (
            "p01_multi_e03.txt",
            "Mechanics (Multi)",
            "Today is about drivetrain mechanics.",
            "Noah",
            "Glad to be here.",
        ),
        (
            "p01_multi_e04.txt",
            "Trail Stewardship (Multi)",
            "Land stewardship today.",
            "Liam",
            "Always happy to talk about this.",
        ),
        (
            "p01_multi_e05.txt",
            "Suspension Setup (Multi)",
            "Suspension setup today.",
            "Noah",
            "Looking forward to it.",
        ),
    ]
    for filename, title, opener, guest, response in multi_episodes:
        (OUT_DIR / filename).write_text(
            multi_template.format(title=title, opener=opener, guest=guest, response=response),
            encoding="utf-8",
        )
        written.append(filename)

    # p06 — Edge Cases (6 episodes). Designed deliberately weird.
    p06_specs = [
        (
            "e01",
            "Empty-ish Episode",
            ["Maya: This episode is intentionally short to test minimal-content handling."],
        ),
        (
            "e02",
            "No-Speaker Lines",
            [
                "Welcome to a transcript with mixed speaker-attribution.",
                "Maya: Some lines have speakers.",
                "Other lines do not.",
                "Liam: But the parser should still classify them.",
            ],
        ),
        (
            "e03",
            "Repeated Speaker Labels",
            [
                "Maya: First line.",
                "Maya: Second line.",
                "Maya: Third line in a row to test merging.",
            ],
        ),
        (
            "e04",
            "Long Single Utterance",
            [
                "Maya: "
                + " ".join(["This is a long single utterance for testing chunking behavior."] * 50)
            ],
        ),
        (
            "e05",
            "Stage Directions",
            [
                "*Theme music plays.*",
                "Maya: Welcome back.",
                "*Short break.*",
                "Maya: We are now back.",
            ],
        ),
        (
            "e06",
            "Mixed Languages",
            [
                "Maya: Welcome.",
                "Liam: Hola, ¿cómo estás?",
                "Maya: A bit of code-switching to test transcription robustness.",
            ],
        ),
    ]
    for ep_id, title, body_lines in p06_specs:
        text = render_p06_edge("Maya", ep_id, title, body_lines)
        (OUT_DIR / f"p06_{ep_id}.txt").write_text(text, encoding="utf-8")
        written.append(f"p06_{ep_id}.txt")

    # p07 — sustainability long episode
    p07_text = render_long_episode(
        podcast_id="p07",
        title="What Sustainability Really Means (And Why Everyone Is Talking About It)",
        host="Alex Morgan",
        guest="Dr. Elena Fischer",
        guest_role="sustainability researcher and systems thinker",
        primary_topic="topic:sustainability",
        secondary_topics=["topic:systems-thinking", "topic:risk-management", "topic:reliability"],
        sponsor_brands=["Notion", "Linear", "Stripe"],
        talking_points=[
            "Sustainability is a systems-thinking problem before it's an environmental problem.",
            "The most expensive thing about climate is the optionality you lose by delaying.",
            "Risk management at planetary scale looks like reliability engineering at company scale.",
            "Most sustainability framing fails because it treats outcomes as targets instead of properties of systems.",
            "Reliability and sustainability share a vocabulary — failure modes, redundancy, and graceful degradation.",
            "I've changed my mind on individual action versus structural change. Both matter, but structural change moves the curve.",
            "Carbon accounting needs the same rigor double-entry bookkeeping brought to finance.",
            "The hardest part of climate work is staying technical when the conversation pulls toward identity.",
        ],
        recurring_orgs=["IPCC", "IEA", "Carbon Trust", "Project Drawdown"],
        target_words=12000,
    )
    (OUT_DIR / "p07_e01.txt").write_text(p07_text, encoding="utf-8")
    written.append("p07_e01.txt")

    # p08 — solar energy long episode
    p08_text = render_long_episode(
        podcast_id="p08",
        title="Solar Energy: Past, Present, and Long-Term Outlook",
        host="Alex Morgan",
        guest="Dr. Elena Fischer",
        guest_role="energy systems researcher",
        primary_topic="topic:solar-energy",
        secondary_topics=["topic:sustainability", "topic:systems-thinking", "topic:reliability"],
        sponsor_brands=["Linear", "Notion", "Stripe"],
        talking_points=[
            "Solar's learning curve was the most important industrial story of the last 40 years.",
            "Storage is solving slower than generation but the curve is real — battery costs are tracking the solar trajectory.",
            "Grid reliability with high renewables is a systems-thinking problem, not a technology one.",
            "Most solar discourse confuses LCOE with delivered cost — they diverge sharply in low-irradiance regions.",
            "Sustainability claims around solar are accurate over the lifecycle, but only if you do the manufacturing footprint honestly.",
            "I changed my mind on rooftop solar. I used to think utility-scale was strictly better. Distributed has resilience properties I underweighted.",
        ],
        recurring_orgs=["IEA", "NREL", "BloombergNEF", "IPCC"],
        target_words=15000,
    )
    (OUT_DIR / "p08_e01.txt").write_text(p08_text, encoding="utf-8")
    written.append("p08_e01.txt")

    # p09 — biohacking (3 episodes, medium length)
    p09_specs = [
        (
            "e01",
            "The Biohacking Premise",
            [
                "Biohacking is mostly hygiene factors and recovery — the exotic stuff is downstream.",
                "Sleep is the most underpriced intervention in the entire space.",
                "Most supplements show effect sizes you can't distinguish from noise without N-of-1 protocols.",
                "Risk management in biohacking is about not breaking yourself with high-confidence interventions.",
            ],
        ),
        (
            "e02",
            "Continuous Glucose Monitors and What They Tell You",
            [
                "CGMs are diagnostic tools more than they are training tools.",
                "Most postprandial glucose 'spikes' are noise to anyone without metabolic dysfunction.",
                "The systems-thinking version of CGM use is feedback loops, not gamification.",
                "I changed my mind on CGM use for general population — turns out 80% gain at 5% of intervention cost is wellness, not biohacking.",
            ],
        ),
        (
            "e03",
            "The Risk Profile of Self-Experimentation",
            [
                "Self-experimentation requires a control group of one, and that control is your own boring baseline.",
                "The expected value of any new supplement is negative unless you measure ruthlessly.",
                "Risk management means treating self-experiments as A/B tests with explicit kill criteria.",
                "Reliability in biological systems is mostly about not stressing them in ways they don't need to be stressed.",
            ],
        ),
    ]
    for ep_id, title, claims in p09_specs:
        text = render_long_episode(
            podcast_id="p09",
            title=title,
            host="Alex Morgan",
            guest="Dr. Elena Fischer",
            guest_role="researcher in metabolic and longevity science",
            primary_topic=f"topic:{re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')}",
            secondary_topics=[
                "topic:risk-management",
                "topic:systems-thinking",
                "topic:reliability",
            ],
            sponsor_brands=["Linear", "Notion", "Stripe"],
            talking_points=claims,
            recurring_orgs=["Levels", "WHOOP", "InsideTracker"],
            target_words=6000,
        )
        (OUT_DIR / f"p09_{ep_id}.txt").write_text(text, encoding="utf-8")
        written.append(f"p09_{ep_id}.txt")

    print(f"Wrote {len(written)} v2 transcripts to {OUT_DIR.relative_to(PROJECT_ROOT)}")
    for name in written:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
