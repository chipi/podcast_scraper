"""0031 — the names the deployed naming code gives 126 served episodes, without a relabel.

Instead of a 131-episode ``relabel_only`` run (the LLM on the DGX, about 17 min an episode): the
operator's decision of 2026-10-17. The 131 candidates are the episodes two replays flagged: the 44
of the relabel list (respellings #2363, false guests #2364, question-lead host seats #2366, Kara
Swisher's Pivot promo published as a guest #2365), the episodes of the 8.4 replay over the whole
corpus (promo and ad voices losing a wrong name, guests and host seats gaining one), and Hard
Fork's merged-hosts "Kara Swisher" voice.

How the list was made. The deployed naming code (sha-ded80e0, whose naming modules are unchanged on
main) replayed over each candidate on the prod snapshot of 2026-10-16, against what its segments
store: 195 voices differ. Every one was read against the voice's own words. In: the 132 the two
replays flagged, all read, and 34 more whose words support the change (a self-introduction, a
promo, an ad in another language, a voice addressed by the name it was given). Out: 29 whose words
contradict the change or do not decide it, among them a replay that renames Kevin Roose "Eliezer
Yudkowsky", replays that unname or demote a host speaking as the host (StarTalk's Neil deGrasse
Tyson, ChinaTalk's Jordan Schneider, Planet Money's Kenny Malone), and one right change the rewrite
refuses (a label two voices share, in a transcript that is not a render). Not in: the Rest Is
History seats that nothing in the text names (F7 A2; a re-diarize).

``NAMING_VOICES``: ``(episode id, voice): (stored name, stored role, new name, new role,
evidence)``, applied exactly as 0029 applies its list (``m0029._Episode``): an entry applies only
while the voice still carries the stored name and role, is done once it carries the new ones, and
anything else REFUSES the episode, untouched and listed. Every surface is rewritten voice by voice
with m0025's machinery (metadata, segments and their ad-free copies, transcripts with every offset
carried, GI, KG, bridge, diagnostics, turns, context). UNDO restores from
``.podcast_scraper/upgrade-backups/0031/`` while each file is still what this migration wrote.
"""

from __future__ import annotations

import bisect
import difflib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..corpus_selection import select_served_artifacts
from ..file_rewrite import append_receipts, undo_from_receipts, write_with_backup
from ..migration import Migration, MigrationContext, MigrationResult
from .m0023_transcript_speaker_prefixes_resynced import (
    _align,
    _load,
    _rendered,
    _rows,
    _write_text_with_backup,
    Refused,
)
from .m0029_naming_arc_voices_repaired import _Episode as _VoiceEpisode

MIGRATION_ID = "0031_naming_voices_rederived"
RECEIPTS_FILE = "naming_voices_rederived.jsonl"
BACKUP_TAG = "0031"

Entry = Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str]

#: Read on 2026-10-17 against each voice's own transcript lines (the evidence is its first words).
NAMING_VOICES: Dict[Tuple[str, str], Entry] = {
    ("https://iono.fm/e/1716488", "SPEAKER_02"): (
        None,
        None,
        "David Shapiro",
        "guest",
        "BizNews Radio 'BNC#9 | Shapiro Q&A: SA's ideology problem,...': says 'I "
        "never put obstacles in my path. You know, funny enough, Peter, it's a lot "
        "of criticism....' (8.4 replay)",
    ),
    ("https://iono.fm/e/1726296", "SPEAKER_00"): (
        "Alec Hogg",
        "guest",
        None,
        "unknown",
        "BizNews Radio 'Solly Moeng: The illusion of SA’s...': says 'I'm Alec Hogg, "
        "and here's what Biz News Premium gets you. The Economist, The Financial...' "
        "(relabel-44 replay)",
    ),
    ("95282045-774e-48ba-9b08-589b6b341991", "SPEAKER_03"): (
        "Luigi Zingales",
        "host",
        None,
        "unknown",
        "Capitalisn't 'Why Capitalism Stopped Working In Japan,...': says 'What "
        "failed was the idea that Japan has a special, different capitalist system, "
        "which...' (8.4 replay)",
    ),
    ("5823a271-bc96-45bb-bd37-8ba6d0967cc5", "SPEAKER_02"): (
        None,
        None,
        "Matthew Yglesias",
        "guest",
        "Capitalisn't 'Why Matthew Yglesias Is Skeptical Of...': says 'My problem "
        "with the YouTube is the product is too good, right? It's not like it's bad. "
        "In...' (8.4 replay)",
    ),
    ("5823a271-bc96-45bb-bd37-8ba6d0967cc5", "SPEAKER_04"): (
        "Matt Iglesias",
        "guest",
        None,
        "host",
        "Capitalisn't 'Why Matthew Yglesias Is Skeptical Of...': says 'There are two "
        "words that have become famous this year for political change, especially "
        "on...' (the host's opening, not the guest: 'Matt Iglesias' removed)",
    ),
    ("d867d354-76e2-11f1-84e8-ff42fe6fab7b", "SPEAKER_04"): (
        None,
        None,
        "Niko Ciminelli",
        "guest",
        "ChinaTalk 'Unitree and the Humanoid Robot Revolution': says 'Yeah, so I "
        "think we're seeing in a world where everyone's very excited about their...' "
        "(8.4 replay)",
    ),
    ("47e1118c-b3bf-11f1-a6e8-db04f8f6f9c2", "SPEAKER_04"): (
        None,
        None,
        "Gerald C. Brown",
        "guest",
        "ChinaTalk 'WarTalk: PLA, 2027, Nukes, and the Pivot to...': says 'off the "
        "bat. I love it. No, 2027, everyone kind of got down a few weird rabbit "
        "holes,...' (8.4 replay)",
    ),
    ("f84abc77-c290-42b3-929a-e30fdb5b3abc", "SPEAKER_01"): (
        None,
        None,
        "Richard McColl",
        "host",
        "Colombia Calling - The English Voice in Colombia 'La Escombrera: Colombia's "
        "Largest Urban Mass...': says 'Colombia for this week Congress reconvenes "
        "today for the 2026 to 2030 term due to elect...' (8.4 replay)",
    ),
    ("1350c524-d61c-11f0-9faa-57cdf1b79e75", "SPEAKER_03"): (
        "Gemini",
        "guest",
        None,
        "guest",
        "Explain It to Me 'The end of etiquette': says 'Hey, it's me, Claude. And "
        "I'm Gemini.' ('Hey, it's me, Claude. And I'm Gemini.': not a person)",
    ),
    ("1350c524-d61c-11f0-9faa-57cdf1b79e75", "SPEAKER_16"): (
        "Kara Swisher",
        "guest",
        None,
        "unknown",
        "Explain It to Me 'The end of etiquette': says 'Hi, everyone. I'm Kara "
        "Swisher. Twice a week, we get together on Pivot to sort through...' (#2365: "
        "Pivot promo read by Kara Swisher, published as a guest)",
    ),
    ("1350c524-d61c-11f0-9faa-57cdf1b79e75", "SPEAKER_17"): (
        "Sky Galloway",
        "guest",
        None,
        "unknown",
        "Explain It to Me 'The end of etiquette': says 'No, excuse me. No, pardon "
        "me. No, hey, man. Can I get by you? And I'm Sky Galloway. So...' (#2365: "
        "the same Pivot promo, Scott Galloway (ASR 'Sky'))",
    ),
    ("substack:post:174857100", "SPEAKER_00"): (
        None,
        None,
        "Tom Frieden",
        "guest",
        "Ground Truths 'Tom Frieden & Eric Topol - Discuss the...': says "
        "'Absolutely. Thank you, Eric. Thanks so much, Eric. The Cassandra curse, "
        "actually, the...' (8.4 replay)",
    ),
    ("25d702f6-ed6a-4565-8065-e729eb0f2650", "SPEAKER_01"): (
        "Kara Swisher",
        "guest",
        None,
        "guest",
        "Hard Fork 'A.I. Backlash Turns Violent + Kara Swisher...': says 'I'm good. "
        "I'm good. I'm feeling a little bummed because this morning, I was listening "
        "to...' (both Hard Fork hosts merged on one voice, not Kara Swisher)",
    ),
    ("a02c151a-dde6-4804-9827-a3df80b8352c", "SPEAKER_00"): (
        "Casey Newton",
        "host",
        "Kevin Roose",
        "host",
        "Hard Fork 'OpenAI’s Two-Week Pause + Jill Lepore on the...': says 'Of "
        "Shopify, while Shopify platform test of shopifycom. Connect even more with "
        "someone you...' (the other host: welcomes the guest; SPEAKER_01 is Casey)",
    ),
    ("a02c151a-dde6-4804-9827-a3df80b8352c", "SPEAKER_01"): (
        "Jill Laporte",
        "guest",
        "Casey Newton",
        "host",
        "Hard Fork 'OpenAI’s Two-Week Pause + Jill Lepore on the...': says 'I "
        "thought this was interesting. ICE has now barred its employees from wearing "
        "Meta's...' (addresses Kevin: this is Casey Newton, not the guest)",
    ),
    ("a02c151a-dde6-4804-9827-a3df80b8352c", "SPEAKER_02"): (
        "Jill Lapour",
        "guest",
        "Jill Lepore",
        "guest",
        "Hard Fork 'OpenAI’s Two-Week Pause + Jill Lepore on the...': says 'you may "
        "have a problem on your hands. Yeah. The public sentiment is turning against "
        "these...' (the guest's name respelled (Jill Lepore))",
    ),
    ("6410d1c5-6b80-4908-a7b3-ab62b46c7576", "SPEAKER_01"): (
        "Devesh Kapoor",
        "guest",
        "Devesh Kapur",
        "guest",
        "Ideas of India 'Arvind Subramanian and Devesh Kapur on...': says 'I mean, "
        "clearly, one of the big themes of the book is the different types of ways "
        "that...' (relabel-44 replay)",
    ),
    ("636b13d7-5c55-45ba-93e5-ebaf4ebfa080", "SPEAKER_01"): (
        None,
        "host",
        "Shruti Rajagopalan",
        "host",
        "Ideas of India 'Pratap Bhanu Mehta on Liberalism, Nihilism,...': says 'and "
        "today my guest is doctor Pratabhanu Mehta, who is the Lawrence Rockefeller "
        "professor...' (8.4 replay)",
    ),
    ("urn:bbc:podcast:p0p94prv", "SPEAKER_02"): (
        "Tom Gatti",
        "guest",
        None,
        "unknown",
        "In Our Time 'Archive: The Measurement of Time': says 'Hello, I'm Tom Gatti, "
        "literary editor and host of the Observer Book Club podcast. Join me...' "
        "(8.4 replay)",
    ),
    ("c7f11b7e-e516-11f0-8700-239aa8e47bb9", "SPEAKER_02"): (
        "Nick Coconis",
        "guest",
        "Nick Kokonas",
        "guest",
        "Invest Like the Best with Patrick O'Shaughnessy 'Nick Kokonas - Know What "
        "You Are Selling': says 'Thank you. My thing is I grew up with a dad who he "
        "would say he was an entrepreneur by...' (relabel-44 replay)",
    ),
    ("8322f8e4-6c19-11f0-9a7d-03d179280ede", "SPEAKER_02"): (
        "Rampton Nami",
        "guest",
        "Ramtin Naimi",
        "guest",
        "Invest Like the Best with Patrick O'Shaughnessy 'Ramtin Naimi - Building "
        "Abstract': says 'And I never really had much time for hobbies, and it was "
        "one that I thought my wife and I...' (relabel-44 replay)",
    ),
    ("substack:post:216845528", "SPEAKER_00"): (
        "Brandon",
        "guest",
        "Brandon",
        "host",
        "Latent Space: The AI Engineer Podcast '🔬 An Oscar, Two Asteroids, and the "
        "Algorithm...': says 'Are you talking about introducing explicit priors "
        "that, you know, she based upon some...' ('I'm Brandon, joined by my co-host "
        "RJ': a host)",
    ),
    ("substack:post:216845528", "SPEAKER_01"): (
        None,
        None,
        "John Platt",
        "guest",
        "Latent Space: The AI Engineer Podcast '🔬 An Oscar, Two Asteroids, and the "
        "Algorithm...': says 'When you talk about multiple hypothesis testing, "
        "right, there's predictive models and...' (8.4 replay)",
    ),
    ("substack:post:194810752", "SPEAKER_00"): (
        "Brandon Anderson",
        "host",
        None,
        "host",
        "Latent Space: The AI Engineer Podcast '🔬 Training Transformers to solve 95% "
        "failure...': says 'So we basically opened the lab. We hired a team. We got "
        "all the instruments. We started...' (the founder describing his lab, not "
        "the host Brandon Anderson)",
    ),
    ("substack:post:194810752", "SPEAKER_01"): (
        "Dan Baer",
        "guest",
        "Daniel Bear",
        "guest",
        "Latent Space: The AI Engineer Podcast '🔬 Training Transformers to solve 95% "
        "failure...': says 'transformers developed for single cell data. There just, "
        "like, weren't really datasets...' (relabel-44 replay)",
    ),
    ("substack:post:209779219", "SPEAKER_02"): (
        "Matt McPartland",
        "guest",
        "Matthew McPartlon",
        "guest",
        "Latent Space: The AI Engineer Podcast '🔬The BioAI Phase Shift - Matthew "
        "McPartlon &...': says 'They have several very exciting announcements that I "
        "think they'll tell us about today....' (relabel-44 replay)",
    ),
    ("substack:post:203476809", "SPEAKER_00"): (
        "Adam Masary",
        "guest",
        "Adam Mosseri",
        "guest",
        "Lenny's Podcast: Product | Career | Growth 'Adam Mosseri: AI is a tailwind "
        "for...': says 'No. I think taste matters a ton. In a world where it's "
        "easier to build things, it's more...' (relabel-44 replay)",
    ),
    ("substack:post:192024618", "SPEAKER_00"): (
        "Simon Wilson",
        "guest",
        "Simon Willison",
        "guest",
        "Lenny's Podcast: Product | Career | Growth 'An AI state of the union: We’ve "
        "passed the...': says 'A lot of people woke up in January and February and "
        "started realizing, oh, wow. I can...' (relabel-44 replay)",
    ),
    ("substack:post:207214207", "SPEAKER_00"): (
        "Diane Penn",
        "guest",
        "Dianne Penn",
        "guest",
        "Lenny's Podcast: Product | Career | Growth 'Anthropic’s first technical PM "
        "on token...': says 'In 2023, when I started, nobody said Anthropic and "
        "Claude and coding in the same...' (relabel-44 replay)",
    ),
    ("substack:post:192660974", "SPEAKER_01"): (
        "Amol Avasari",
        "guest",
        "Amol Avasare",
        "guest",
        "Lenny's Podcast: Product | Career | Growth 'Head of Growth (Anthropic): "
        "“Claude is...': says 'we were very much the smallest, least well funded "
        "player in this space. We didn't have...' (relabel-44 replay)",
    ),
    ("substack:post:205673536", "SPEAKER_00"): (
        "Noam Siegel",
        "guest",
        "Noam Segal",
        "guest",
        "Lenny's Podcast: Product | Career | Growth 'How tech workers actually feel "
        "about AI in...': says 'Banned news for the tech community. employee "
        "problem. The honeymoon period with AI is...' (relabel-44 replay)",
    ),
    ("substack:post:194566669", "SPEAKER_00"): (
        "Max Schoning",
        "guest",
        "Max Schoening",
        "guest",
        "Lenny's Podcast: Product | Career | Growth 'Why cultivating agency matters "
        "more than...': says 'Four, it was very easy to always say, well, I will "
        "never be able to do this because...' (relabel-44 replay)",
    ),
    ("substack:post:193130101", "SPEAKER_01"): (
        "Nikhil Singhal",
        "guest",
        "Nikhyl Singhal",
        "guest",
        "Lenny's Podcast: Product | Career | Growth 'Why half of product managers "
        "are in trouble...': says 'is very much in stress. Nothing's constant. "
        "Everyone's in a state of alert. If you talk...' (relabel-44 replay)",
    ),
    ("bc79d943-e6c0-492a-8df9-f7051b3734fc", "SPEAKER_00"): (
        None,
        "host",
        "Tim Scarfe",
        "host",
        "Machine Learning Street Talk (MLST) 'AI Is Learning at the Wrong Level "
        "of...': says 'And this is very pertinent because you've got a paper out "
        "basically saying that we should...' (relabel-44 replay)",
    ),
    ("e32aa079-bf95-4a16-819a-17a5a96eb186", "SPEAKER_02"): (
        None,
        None,
        "John Jumper",
        "guest",
        "Machine Learning Street Talk (MLST) 'He won a Nobel here for AlphaFold. "
        "Then he...': says 'Ich glaube, ich liebe die Bitter-Lesson...', then "
        "'In fact, AlphaFold 2 ist die Opposite von das' (8.4 replay)",
    ),
    ("e1ff02ca-df6a-4672-991f-d0443a7c834c", "SPEAKER_01"): (
        "Enzo",
        "guest",
        "Enzo Blindow",
        "guest",
        "Machine Learning Street Talk (MLST) 'The Secret Engine of AI - "
        "Prolific...': says 'Models, when knowing that they were being observed and "
        "evaluated, actually digressed away...' (relabel-44 replay)",
    ),
    ("e1ff02ca-df6a-4672-991f-d0443a7c834c", "SPEAKER_02"): (
        None,
        "host",
        "Tim Scarfe",
        "host",
        "Machine Learning Street Talk (MLST) 'The Secret Engine of AI - "
        "Prolific...': says 'You think they ever could understand? And you were at "
        "Meta on Instagram, right? Oh,...' ('It's amazing to have you both on MLST': "
        "the host)",
    ),
    ("64f69dd8-9167-4b6f-9c04-6221f62e7ca2", "SPEAKER_00"): (
        "John Rolle",
        "host",
        "John Rolle",
        "guest",
        "Macro Musings with David Beckworth 'John Rolle on The Bahamas' Fixed "
        "Exchange...': says 'Well, David, thank you again. And let me say that I "
        "have taken a lot of opportunities...' (John Rolle is the guest the host "
        "introduces)",
    ),
    ("64f69dd8-9167-4b6f-9c04-6221f62e7ca2", "SPEAKER_02"): (
        None,
        "host",
        "David Beckworth",
        "host",
        "Macro Musings with David Beckworth 'John Rolle on The Bahamas' Fixed "
        "Exchange...': says 'Our guest today is John Roll. John is the governor of "
        "the Central Bank of the Bahamas. He...' (8.4 replay)",
    ),
    ("fef17869-6be2-4595-9940-902e5104af62", "SPEAKER_01"): (
        "Dan Aronoff",
        "guest",
        "Daniel Aronoff",
        "guest",
        "Macro Musings with David Beckworth 'Neha Narula, Anders Brownworth, and "
        "Daniel...': says 'And I'm Dan Aronoff, and I'm in the economics department "
        "at MIT, an economist by...' (relabel-44 replay)",
    ),
    ("cf679038-5568-486d-9d41-d083486d42ce", "SPEAKER_00"): (
        "Chris Orenay",
        "guest",
        None,
        "guest",
        "Macro Musings with David Beckworth 'Yesha Yadav, Chris Odinet, and Andrea "
        "Tosato...': says 'David, thank you for having us. David, thanks so much for "
        "having us again. And just to...' (a guest thanking David; the name belongs "
        "to SPEAKER_01)",
    ),
    ("cf679038-5568-486d-9d41-d083486d42ce", "SPEAKER_01"): (
        "Andrea Tosoto",
        "guest",
        "Chris Orenay",
        "guest",
        "Macro Musings with David Beckworth 'Yesha Yadav, Chris Odinet, and Andrea "
        "Tosato...': says 'Sure. Happy to. And I'm delighted to be here. I've been a "
        "longtime fan of the podcast. So...' (self-introduction 'my name is Chris "
        "Orenay')",
    ),
    ("cf679038-5568-486d-9d41-d083486d42ce", "SPEAKER_02"): (
        "André Tozzato",
        "guest",
        "Andrea Tosato",
        "guest",
        "Macro Musings with David Beckworth 'Yesha Yadav, Chris Odinet, and Andrea "
        "Tosato...': says 'And again, I joined my co-authors and being delighted to "
        "be a big fan of the pod. I'm...' (relabel-44 replay)",
    ),
    ("d7c3f6ee-747c-11f0-8b13-238bd4f86533", "SPEAKER_00"): (
        None,
        "host",
        "Noah Kravitz",
        "host",
        "NVIDIA AI Podcast 'AI, Spatial Intelligence, and 3D Content...': says "
        "'Hello, and welcome to the NVIDIA AI podcast. I'm your host, Noah Kravitz. "
        "This past...' ('I'm your host, Noah Kravitz')",
    ),
    ("d7c3f6ee-747c-11f0-8b13-238bd4f86533", "SPEAKER_01"): (
        "Sonya Fiddler",
        "guest",
        "Sanja Fidler",
        "guest",
        "NVIDIA AI Podcast 'AI, Spatial Intelligence, and 3D Content...': says 'Hi, "
        "Noah, and hi, audience. I'm very excited to be on this AI podcast. maybe "
        "I'll start...' (relabel-44 replay)",
    ),
    ("232137ee-f689-11f0-9dac-2ff078d65b6b", "SPEAKER_01"): (
        None,
        "host",
        "Jason Goldberg",
        "guest",
        "NVIDIA AI Podcast 'From Warehouses to Robot Shoppers: Jason...': says "
        "'there's not universal agreement. It's it in our little pool of retail, "
        "there's a lot of...' (8.4 replay)",
    ),
    ("dca8389c-0f27-11f0-8da7-87d9daeaf3b1", "SPEAKER_00"): (
        None,
        "host",
        "Noah Kravitz",
        "host",
        "NVIDIA AI Podcast 'NVIDIA’s Jacob Liberman on the Power of...': says "
        "'Hello, and welcome to the NVIDIA AI podcast. I'm your host, Noah Kravitz. "
        "Developers are...' ('I'm your host, Noah Kravitz')",
    ),
    ("dca8389c-0f27-11f0-8da7-87d9daeaf3b1", "SPEAKER_01"): (
        "Jacob Lieberman",
        "guest",
        "Jacob Liberman",
        "guest",
        "NVIDIA AI Podcast 'NVIDIA’s Jacob Liberman on the Power of...': says 'the "
        "latest evolution in the way people are using GenAI. We're kind of in the "
        "third era of...' (relabel-44 replay)",
    ),
    ("e12140a6-245e-11f1-83d8-d393763beb3b", "SPEAKER_01"): (
        "Andrei Karpathy",
        "guest",
        "Andrej Karpathy",
        "guest",
        "No Priors: Artificial Intelligence | Technology | Startups 'Andrej Karpathy "
        "on Code Agents,...': says 'express my will to my agents for sixteen hours a "
        "day. Manifest. How can I have not just a...' (relabel-44 replay)",
    ),
    ("ee31f9b4-2623-11f0-9aeb-0fb457794f55", "SPEAKER_02"): (
        "Sarah Guo",
        "host",
        None,
        "guest",
        "No Priors: Artificial Intelligence | Technology | Startups 'O3 and the Next "
        "Leap in Reasoning with...': says 'are focused on thinking carefully before "
        "they respond. And these models are, in sort of...' (the guest explaining "
        "o3, not the host Sarah Guo)",
    ),
    ("6bd6fe0e-c4fe-11f0-888a-af21e27bc591", "SPEAKER_00"): (
        "Cheng Qi",
        "guest",
        "Cheng Chi",
        "guest",
        "No Priors: Artificial Intelligence | Technology | Startups 'Sunday "
        "Robotics: Scaling the Home Robot...': says 'Nobody want to do their dishes. "
        "Nobody want to do their laundry. People will love to...' (relabel-44 replay)",
    ),
    ("51c4092c-b309-11f1-8603-fb8d30e1277f", "SPEAKER_01"): (
        "Stefano Erman",
        "guest",
        "Stefano Ermon",
        "guest",
        "No Priors: Artificial Intelligence | Technology | Startups 'Why Diffusion "
        "Will Win AI Inference with...': says 'Great to be here. For sure, yeah. "
        "I've been doing research in generative models for...' (relabel-44 replay)",
    ),
    ("ea283f31-8b41-42a8-9147-b49f01293e32", "SPEAKER_00"): (
        "Joe Weisenthal",
        "host",
        None,
        "unknown",
        "Odd Lots 'Introducing: Our Town': says 'If you drive ninety miles northeast "
        "of Nashville into the foothills of the Appalachian...' (8.4 replay)",
    ),
    ("ea283f31-8b41-42a8-9147-b49f01293e32", "SPEAKER_03"): (
        "Tracy Alloway",
        "host",
        None,
        "unknown",
        "Odd Lots 'Introducing: Our Town': says 'We need cities on a shining hill "
        "that exemplify and embody the healthy, natural human way...' (a clip in the "
        "trailer for another show, not Tracy Alloway)",
    ),
    ("ccc902c3-3887-455a-ac40-cb253f1cc533", "SPEAKER_07"): (
        "Leila Faldel",
        "guest",
        None,
        "unknown",
        "Planet Money 'Can transforming neighborhoods help kids...': says 'The "
        "midterms are coming, and that means the news cycle, which is already "
        "intense, is...' (8.4 replay)",
    ),
    ("20e14ed3-7dd8-43ac-a467-7f9ab01a1858", "SPEAKER_05"): (
        "Leila Faldel",
        "guest",
        None,
        "unknown",
        "Planet Money 'Chevron, Venezuela and the Paradox of Plenty': says 'What to "
        "watch out for and how to decode the secret messages hidden in your poop "
        "on...' (8.4 replay)",
    ),
    ("7dae09ac-f8ea-485d-9cdf-cbb7c913da62", "SPEAKER_07"): (
        "Leila Faldel",
        "guest",
        None,
        "unknown",
        "Planet Money 'How Black hair care grew Black power': says 'What to watch "
        "out for and how to decode the secret messages hidden in your poop on...' "
        "(8.4 replay)",
    ),
    ("82c87af0-621b-4680-87eb-f7e8e5b5c949", "SPEAKER_03"): (
        "New Yorker",
        "host",
        None,
        "unknown",
        "Planet Money 'How much money President Trump and his...': says 'Now if if "
        "well, as we go through this, I turn pages, is that a problem? No. No. I "
        "think...' ('New Yorker' is not a person)",
    ),
    ("82c87af0-621b-4680-87eb-f7e8e5b5c949", "SPEAKER_09"): (
        "Leila Faldel",
        "guest",
        None,
        "unknown",
        "Planet Money 'How much money President Trump and his...': says 'The "
        "midterms are coming, and that means the news cycle, which is already "
        "intense, is...' (8.4 replay)",
    ),
    ("52fe0cf1-8566-457b-aab0-b32791f4940f", "SPEAKER_01"): (
        "Sarah",
        "guest",
        None,
        "unknown",
        "Planet Money 'How to get through the Strait of Hormuz': says 'You know, "
        "live events can generate a lot of income for towns and cities. This is a "
        "known...' (8.4 replay)",
    ),
    ("52fe0cf1-8566-457b-aab0-b32791f4940f", "SPEAKER_05"): (
        "unknown_guest_1",
        "guest",
        None,
        "guest",
        "Planet Money 'How to get through the Strait of Hormuz': says 'Okay. "
        "Christian Sinclair. I'm the production manager at Fanographics Books. And "
        "then...' ('unknown_guest_1' is a placeholder, not a name)",
    ),
    ("4a8e203d-7e07-4a9c-b7ac-b1b8d8a24c05", "SPEAKER_05"): (
        "Leila Faldel",
        "guest",
        None,
        "unknown",
        "Planet Money 'Iran, protests, and sanctions': says 'The midterms are "
        "coming, that means the news cycle, which is already intense, is about "
        "to...' (8.4 replay)",
    ),
    ("43627a93-796f-4805-89ae-76c7bed4997d", "SPEAKER_19"): (
        "Brittany Luce",
        "guest",
        None,
        "unknown",
        "Planet Money 'Love in the time of Palantir': says 'Hey, what's your "
        "problem? No, seriously. Are your grocery bills too high? Are you...' "
        "(relabel-44 replay)",
    ),
    ("8e813358-f9df-4e62-bdb0-36c6fd5f9d0b", "SPEAKER_05"): (
        "Leila Faldel",
        "guest",
        None,
        "unknown",
        "Planet Money 'So are we in an AI bubble? Here are clues to...': says 'The "
        "midterms are coming, and that means the new cycle, which is already "
        "intense, is about...' (8.4 replay)",
    ),
    ("23620594-fbe9-4538-8ba8-86087587067e", "SPEAKER_05"): (
        "Leila Faldil",
        "guest",
        None,
        "unknown",
        "Planet Money 'The Business of Heated Rivalry': says 'The midterms are "
        "coming, and that means the news cycle, which is already intense, is...' "
        "(8.4 replay)",
    ),
    ("23620594-fbe9-4538-8ba8-86087587067e", "SPEAKER_13"): (
        "Jesse Thorne",
        "guest",
        None,
        "unknown",
        "Planet Money 'The Business of Heated Rivalry': says 'Waylon, it seems like "
        "everything is being called a recession indicator these days. Yeah....' (8.4 "
        "replay)",
    ),
    ("46365717-63d4-4477-8d5f-256cfe49bfbd", "SPEAKER_04"): (
        "Leila Faldel",
        "guest",
        None,
        "unknown",
        "Planet Money 'The ICE hiring boom': says 'The midterms are coming, and that "
        "means the news cycle, which is already intense, is...' (8.4 replay)",
    ),
    ("20992888-a881-48ed-b5bb-903ec0a34ba0", "SPEAKER_06"): (
        "Leila Faldel",
        "guest",
        None,
        "unknown",
        "Planet Money 'The Invention Invention': says 'The midterms are coming, and "
        "that means the news cycle, which is already intense, is...' (8.4 replay)",
    ),
    ("f1dfec6a-dd9f-4207-b17a-8d09af66a51a", "SPEAKER_13"): (
        "Jesse Thorne",
        "guest",
        None,
        "unknown",
        "Planet Money 'The Supreme Court struck down a bunch of...': says 'Wailen, "
        "it seems like everything is being called a recession indicator these days. "
        "Yeah....' (8.4 replay)",
    ),
    ("537b0f9c-14cf-4503-829f-683473110cc0", "SPEAKER_05"): (
        "Leila Faldel",
        "guest",
        None,
        "unknown",
        "Planet Money 'Venezuela’s recent economic history (Update)': says 'The "
        "midterms are coming, that means the new cycle, which is already intense, is "
        "about to...' (8.4 replay)",
    ),
    ("2ffbf42c-503f-4b51-9462-af3c5626c900", "SPEAKER_09"): (
        "Leila Faldil",
        "guest",
        None,
        "unknown",
        "Planet Money 'When designing a GAME, what's in a name?': says 'The midterms "
        "are coming, and that means the news cycle, which is already intense, is...' "
        "(8.4 replay)",
    ),
    ("56901d28-494c-4668-9159-070848c75dbd", "SPEAKER_00"): (
        "Joel Cherico",
        "guest",
        None,
        "unknown",
        "StarTalk with Neil deGrasse Tyson 'Lost in Space Game Night': says 'Oh. I'm "
        "Joel Cherico, and I support StarTalk on Patreon. This is StarTalk with "
        "Neil...' (relabel-44 replay)",
    ),
    ("cc425097-55f4-4ae8-90a8-96d293d2d261", "SPEAKER_00"): (
        "Ali Khan Hemraj",
        "guest",
        None,
        "unknown",
        "StarTalk with Neil deGrasse Tyson 'Solving AI’s Energy Problem with Kathryn "
        "Huff': says 'to a completely different order of magnitude in terms of "
        "deaths per terawatt hour....' (relabel-44 replay)",
    ),
    ("1abce225-afd1-4233-bb25-481995d56e63", "SPEAKER_00"): (
        "Ernie Carducci",
        "guest",
        None,
        "unknown",
        "StarTalk with Neil deGrasse Tyson 'Why Craters Are Round, Size of Life, "
        "The...': says 'Hi, I'm Ernie Carducci from Columbus, Ohio. I'm here with my "
        "son, Ernie, because we...' (relabel-44 replay)",
    ),
    ("88fee8b7-98cf-46f0-9a69-e3f39cfe3979", "SPEAKER_00"): (
        "Lani Lum",
        "guest",
        None,
        "unknown",
        "StarTalk with Neil deGrasse Tyson 'Why Do We Exist with Hakeem Oluseyi': "
        "says 'I'm Lani Lum, and I support StarTalk on Patreon. This is StarTalk "
        "with Neil deGrasse...' (relabel-44 replay)",
    ),
    ("b260150a-1f25-11f1-990b-73ee76699165", "SPEAKER_02"): (
        "Phoebe Judge",
        "guest",
        None,
        "unknown",
        "The Curiosity Shop with Brené Brown and Adam Grant 'Resisting Influence in "
        "an Over-Influenced...': says 'I'm Phoebe Judge. And for more than a decade, "
        "I've hosted the podcast Criminal. Telling...' (8.4 replay)",
    ),
    ("c1b9decd-71f5-4fe5-bf64-ed311b0b40b8", "SPEAKER_00"): (
        "Brian Chan",
        "guest",
        None,
        "unknown",
        "The Daily '48 Days Until the Midterms: Is a Blue Wave...': says 'Hi, I'm "
        "Brian Chan. I'm a tech columnist for the New York Times. The tech industry "
        "is...' (8.4 replay)",
    ),
    ("c1b9decd-71f5-4fe5-bf64-ed311b0b40b8", "SPEAKER_03"): (
        "Daniel Popper",
        "guest",
        None,
        "unknown",
        "The Daily '48 Days Until the Midterms: Is a Blue Wave...': says 'We'll be "
        "right back. Hey, I'm Daniel Popper. You know The Athletic is home to the "
        "world's...' (The Athletic promo read by Daniel Popper)",
    ),
    ("94855b15-9742-4344-bf6c-d6a1e0df51ea", "SPEAKER_12"): (
        "Dani Blum",
        "guest",
        None,
        "unknown",
        "The Daily 'A Failed Execution and the Future of the...': says 'I'm Dani "
        "Blum, a journalist at the New York Times. I spend a lot of my time "
        "reporting on...' (8.4 replay)",
    ),
    ("5f2d7bdb-5340-4995-ae16-7fd05b548488", "SPEAKER_08"): (
        "Brian Chen",
        "guest",
        None,
        "unknown",
        "The Daily 'Are Prediction Markets Gambling? A Lot Rides...': says 'Hi, I'm "
        "Brian Chen. I'm a tech columnist for The New York Times. The tech industry "
        "is...' (8.4 replay)",
    ),
    ("6cdee2c5-a4e5-49f4-ac41-ccf835b24794", "SPEAKER_07"): (
        "Catherine Benhold",
        "guest",
        None,
        "unknown",
        "The Daily 'Arizona’s Food Stamp Crisis Is Coming for...': says 'Did you "
        "know that India is the biggest adopter of crypto globally and that Estonia "
        "offers...' (8.4 replay)",
    ),
    ("f2a8f88f-35f7-4417-82d9-d4f57f343ce9", "SPEAKER_05"): (
        "Catherine Benhold",
        "guest",
        None,
        "unknown",
        "The Daily 'Billy Strings Escaped the Darkness. He’s...': says 'Did you know "
        "that India is the biggest adopter of crypto globally and that Estonia "
        "offers...' (8.4 replay)",
    ),
    ("3123d890-70b4-4b6c-a461-27bb204089fe", "SPEAKER_13"): (
        "Brian Chan",
        "guest",
        None,
        "unknown",
        "The Daily 'Classical Music Is in Crisis. Gustavo...': says 'Hi, I'm Brian "
        "Chan. I'm a tech columnist for the New York Times. Instead of just...' "
        "(relabel-44 replay)",
    ),
    ("6ebbe026-0058-4c5c-b697-9449fe2b9d8f", "SPEAKER_00"): (
        "Tracy Mumford",
        "guest",
        None,
        "unknown",
        "The Daily 'From Trump’s Attorney to Attorney General:...': says 'Hey. I'm "
        "Tracy Mumford. There is a lot happening right now. The headlines podcast "
        "from...' (8.4 replay)",
    ),
    ("bf6646a8-cdd6-4d87-92bd-6d088fe36dc3", "SPEAKER_00"): (
        "Brian Chan",
        "guest",
        None,
        "unknown",
        "The Daily 'He Was Deported to a Country He’d Never...': says 'Hi, I'm Brian "
        "Chan. I'm a tech columnist for The New York Times. The tech industry is...' "
        "(8.4 replay)",
    ),
    ("d996026e-5911-457a-b63a-da9fac3ab5c1", "SPEAKER_00"): (
        "Brian Chan",
        "guest",
        None,
        "unknown",
        "The Daily 'Health Trackers Are Everywhere. Do Babies...': says 'Hi, I'm "
        "Brian Chan. I'm a tech columnist for the New York Times. The tech industry "
        "is...' (8.4 replay)",
    ),
    ("3e537123-f463-4ed2-bbea-daf5665ea07f", "SPEAKER_08"): (
        "Brian Chen",
        "guest",
        None,
        "unknown",
        "The Daily 'How Flock Cameras Took Over America': says 'Hi, I'm Brian Chen. "
        "I'm a tech columnist for The New York Times. Instead of just...' (8.4 "
        "replay)",
    ),
    ("2ede0e96-7f66-4650-a6a3-4f59acf99686", "SPEAKER_02"): (
        "Brian Chan",
        "guest",
        None,
        "unknown",
        "The Daily 'How a Teenager Found Undiagnosed Brain...': says 'We'll be right "
        "back. Hi, I'm Brian Chan. I'm a tech columnist for The New York Times....' "
        "(8.4 replay)",
    ),
    ("29b52580-0f56-41a6-8b8a-288bd604bc01", "SPEAKER_00"): (
        "Tracy Mumford",
        "guest",
        None,
        "unknown",
        "The Daily 'ICE Ramps Back Up, With Deadly Results': says 'Hey. I'm Tracy "
        "Mumford. There is a lot happening right now. The headlines podcast from...' "
        "(8.4 replay)",
    ),
    ("b36b086c-6a67-4e31-8360-67511439d497", "SPEAKER_02"): (
        "Lulu Garcia Navarro",
        "guest",
        "Lulu Garcia Navarro",
        "host",
        "The Daily 'Ina Garten Says You Can Either Get on Her...': says 'I'm Lulu "
        "Garcia Navarro, and today I'm interviewing Ina Garten. Garten's best known "
        "for...' ('I'm Lulu Garcia Navarro, and today I'm interviewing': the host)",
    ),
    ("b36b086c-6a67-4e31-8360-67511439d497", "SPEAKER_03"): (
        "Brian Chan",
        "guest",
        None,
        "unknown",
        "The Daily 'Ina Garten Says You Can Either Get on Her...': says 'Hi, I'm "
        "Brian Chan. I'm a tech columnist for the New York Times. Instead of "
        "just...' (8.4 replay)",
    ),
    ("67c2241b-d7d3-471f-b4ff-0b05d2b3ead3", "SPEAKER_03"): (
        "Catherine Benhold",
        "guest",
        None,
        "unknown",
        "The Daily 'Inside Trump’s New Deal With Iran': says 'Did you know that "
        "India is the biggest adopter of crypto globally and that Estonia offers...' "
        "(8.4 replay)",
    ),
    ("c85ca0b8-06a3-4135-96f2-dbe71a97651a", "SPEAKER_04"): (
        "Brian Chen",
        "guest",
        None,
        "unknown",
        "The Daily 'Inside the Devastating Hack of the F.B.I.': says 'Hi, I'm Brian "
        "Chen. I'm a tech columnist for The New York Times. The tech industry is...' "
        "(8.4 replay)",
    ),
    ("6baab6b1-70a4-4f0c-8031-ed0170be43ea", "SPEAKER_03"): (
        "Catherine Benhold",
        "guest",
        None,
        "unknown",
        "The Daily 'Is 'Gentle Parenting' Ruining French...': says 'Did you know "
        "that India is the biggest adopter of crypto globally and that Estonia "
        "offers...' (8.4 replay)",
    ),
    ("b7d968f5-b670-4f32-808b-3c45895accaf", "SPEAKER_03"): (
        "Catherine Benhold",
        "guest",
        None,
        "unknown",
        "The Daily 'Is America’s Drinking Water the Next Front...': says 'of "
        "municipal water systems. Right now, several Minnesota cities are scrambling "
        "to respond...' (8.4 replay)",
    ),
    ("a11fa7a5-5ae0-4e30-9a22-bb588f5d7230", "SPEAKER_02"): (
        "Tracy Mumford",
        "guest",
        None,
        "unknown",
        "The Daily 'Mick Jagger Knows He May Have Played His...': says 'Hey. I'm "
        "Tracy Mumford. There is a lot happening right now. The headlines podcast "
        "from...' (8.4 replay)",
    ),
    ("23a25459-0387-408d-9958-a70082454e95", "SPEAKER_02"): (
        "Tracy Mumford",
        "guest",
        None,
        "unknown",
        "The Daily 'Robby Hoffman Will Always Feel Poor, No...': says 'Hey. I'm "
        "Tracy Mumford. There is a lot happening right now. The headlines podcast "
        "from...' (8.4 replay)",
    ),
    ("a5194659-370d-42e0-bc0b-c320072b24e6", "SPEAKER_00"): (
        "Brian Chan",
        "guest",
        None,
        "unknown",
        "The Daily 'Sylvester Stallone Hid His Struggles for...': says 'It's getting "
        "more and more and more. Hi, I'm Brian Chan. I'm a tech columnist for The "
        "New...' (8.4 replay)",
    ),
    ("0ec506f0-e5ba-471b-99fd-4ba55d1622c7", "SPEAKER_00"): (
        "Tracy Mumford",
        "guest",
        None,
        "unknown",
        "The Daily 'The Battle Over A.I. in the Classroom': says 'Hey. I'm Tracy "
        "Mumford. There is a lot happening right now. The headlines podcast from...' "
        "(8.4 replay)",
    ),
    ("d9759ebd-1c66-4ffd-907d-f40e391e2a01", "SPEAKER_08"): (
        "Brian Chan",
        "guest",
        None,
        "unknown",
        "The Daily 'The Best TV Shows of the 21st Century': says 'Hi, I'm Brian "
        "Chan. I'm a tech columnist for the New York Times. The tech industry is...' "
        "(8.4 replay)",
    ),
    ("df217bd7-0033-426c-8474-ceff49fcf6e9", "SPEAKER_01"): (
        "Brian Chan",
        "guest",
        None,
        "unknown",
        "The Daily 'The Bond Market Is Flipping Out. Here’s Why...': says 'Hi, I'm "
        "Brian Chan. I'm a tech columnist for The New York Times. The tech industry "
        "is...' (8.4 replay)",
    ),
    ("df217bd7-0033-426c-8474-ceff49fcf6e9", "SPEAKER_03"): (
        "Daniel Popper",
        "guest",
        None,
        "unknown",
        "The Daily 'The Bond Market Is Flipping Out. Here’s Why...': says 'Hey, I'm "
        "Daniel Popper. You know, The Athletic is home to the world's best "
        "sports...' (The Athletic promo read by Daniel Popper)",
    ),
    ("7ee00723-453c-4054-b591-ee23ec884540", "SPEAKER_04"): (
        "Catherine Benhold",
        "guest",
        None,
        "unknown",
        "The Daily 'The Cyclospora Mystery: How Human Poop Got...': says 'where I "
        "got it. I would get nauseous here and there. I had such, like, low "
        "appetite, and...' (8.4 replay)",
    ),
    ("b0e586d9-8253-443e-93ea-b4d6dc8e1ae5", "SPEAKER_04"): (
        "Brian Chen",
        "guest",
        None,
        "unknown",
        "The Daily 'The Fight for the West Bank': says 'It has a beautiful view. Hi, "
        "I'm Brian Chen. I'm a tech columnist for the New York Times....' (8.4 "
        "replay)",
    ),
    ("4e16892d-3e2a-423c-bd8d-f3e26194a309", "SPEAKER_14"): (
        "Brian Chan",
        "guest",
        None,
        "unknown",
        "The Daily 'The Intertwined Legacies of Gloria Steinem...': says 'Hi, I'm "
        "Brian Chan. I'm a tech columnist for The New York Times. Instead of "
        "just...' (8.4 replay)",
    ),
    ("8f5e1454-6790-4e6c-891c-f71788c109dc", "SPEAKER_07"): (
        None,
        "host",
        "Elizabeth Williamson",
        "guest",
        "The Daily 'The Life and, for Now, the Death of the...': says 'So a few "
        "months ago, I took a tour of the Kennedy Center with a tour guide there. "
        "Oh, can...' (the reporter's tour narration: the guest, not a host)",
    ),
    ("8f5e1454-6790-4e6c-891c-f71788c109dc", "SPEAKER_14"): (
        "Brian Chen",
        "guest",
        None,
        "unknown",
        "The Daily 'The Life and, for Now, the Death of the...': says 'Hi, I'm Brian "
        "Chen. I'm a tech columnist for the New York Times. The tech industry is...' "
        "(8.4 replay)",
    ),
    ("0d5b552f-016f-47a7-8325-e993f48e9ea2", "SPEAKER_08"): (
        "Tracy Mumford",
        "guest",
        None,
        "unknown",
        "The Daily 'The Onion’s Latest Joke: Taking Over Infowars': says 'Hey. I'm "
        "Tracy Mumford. There is a lot happening right now. The headlines podcast "
        "from...' (8.4 replay)",
    ),
    ("10fe0293-3eff-4fdd-be3b-6020ff8b0e3a", "SPEAKER_07"): (
        "Catherine Benholt",
        "guest",
        None,
        "unknown",
        "The Daily 'The U.S. Missile Stockpile Is Dangerously Low': says 'Did you "
        "know that India is the biggest adopter of crypto globally and that Estonia "
        "offers...' (8.4 replay)",
    ),
    ("955986ad-c79e-4bcc-8536-3a7280540ab0", "SPEAKER_04"): (
        "Tracy Mumford",
        "guest",
        None,
        "unknown",
        "The Daily 'The Unprecedented Personal Profits of...': says 'Hey. I'm Tracy "
        "Mumford. There is a lot happening right now. The headlines podcast from...' "
        "(8.4 replay)",
    ),
    ("b587b66a-193d-4669-af69-2003af848d97", "SPEAKER_06"): (
        "Dani Blum",
        "guest",
        None,
        "unknown",
        "The Daily 'Three Vans, a Startled Farmer and an Iranian...': says 'I'm Dani "
        "Blum, a journalist at the New York Times. I spend a lot of my time "
        "reporting on...' (relabel-44 replay)",
    ),
    ("e8457907-7885-4f02-bcb4-f2a213d743cf", "SPEAKER_18"): (
        "Brian Chen",
        "guest",
        None,
        "unknown",
        "The Daily 'Trump Banned the Media. This Time, the Media...': says 'Hi, I'm "
        "Brian Chen. I'm a tech columnist for the New York Times. The tech industry "
        "is...' (8.4 replay)",
    ),
    ("0dcb75ec-ecb0-43b2-946f-447a52790176", "SPEAKER_01"): (
        "Brian Chan",
        "guest",
        None,
        "unknown",
        "The Daily 'What Safeguarding Nukes Can Teach Us About...': says 'Hi, I'm "
        "Brian Chan. I'm a tech columnist for The New York Times. The tech industry "
        "is...' (8.4 replay)",
    ),
    ("c2a1670f-c52f-4f65-aae8-9fb9605fe30c", "SPEAKER_02"): (
        "Brian Chan",
        "guest",
        None,
        "unknown",
        "The Daily 'Who Will Rule the 2026 U.S. Open?': says 'Here goes Arthur Ashe "
        "now, leading off for the first serve. Hi, I'm Brian Chan. I'll get...' (8.4 "
        "replay)",
    ),
    ("c89d8533-9397-49b1-9849-0bc560b5a443", "SPEAKER_05"): (
        "Catherine Benholt",
        "guest",
        None,
        "unknown",
        "The Daily 'Why Adults Are Getting Cancer at a Younger...': says 'Did you "
        "know that India is the biggest adopter of crypto globally and that Estonia "
        "offers...' (8.4 replay)",
    ),
    ("50b87109-f780-4c0d-aa33-2e91b4404b1d", "SPEAKER_13"): (
        "Brian Chen",
        "guest",
        None,
        "unknown",
        "The Daily 'Why Washington Is Trying to Rein In College...': says 'Hi, I'm "
        "Brian Chen. I'm a tech columnist for The New York Times. The tech industry "
        "is...' (8.4 replay)",
    ),
    ("94561f1e-45a0-42eb-b138-08792d74fd3b", "SPEAKER_00"): (
        "Michael Barbaro",
        "host",
        None,
        "unknown",
        "The Daily '‘Buy Now, Pay Later’: A New Wave of Consumer...': says 'Null. "
        "Das ist der Prozentsatz an Energie, der mehr abends für meine Buchhaltung "
        "bleibt....' (a German ad, not Michael Barbaro)",
    ),
    ("94561f1e-45a0-42eb-b138-08792d74fd3b", "SPEAKER_05"): (
        "Tracy Mumford",
        "guest",
        None,
        "unknown",
        "The Daily '‘Buy Now, Pay Later’: A New Wave of Consumer...': says 'Hey, I'm "
        "Tracy Mumford. There is a lot happening right now. The Headlines podcast "
        "from...' (8.4 replay)",
    ),
    ("94561f1e-45a0-42eb-b138-08792d74fd3b", "SPEAKER_10"): (
        "Rachel Abrams",
        "host",
        None,
        "unknown",
        "The Daily '‘Buy Now, Pay Later’: A New Wave of Consumer...': says 'Spare "
        "Zeit mit KI-Zahlungsabgleich. Vernetzten Tools und nahtloser Zusammenarbeit "
        "mit...' (a German ad, not Rachel Abrams)",
    ),
    ("71efebcc-f742-4834-ad9a-43e5d222e674", "SPEAKER_00"): (
        None,
        None,
        "Dan Kleinbaum",
        "guest",
        "The Flip 'Why Are Cross-Border Payments So Hard?': says 'There's just so "
        "many manual processes in the loop. So a lot of this stuff just ends up...' "
        "(relabel-44 replay)",
    ),
    ("71efebcc-f742-4834-ad9a-43e5d222e674", "SPEAKER_01"): (
        None,
        None,
        "Justin Norman",
        "host",
        "The Flip 'Why Are Cross-Border Payments So Hard?': says 'That's Benjamin "
        "Fernandez, the co-founder and CEO of the Remittance platform Nala, and...' "
        "(relabel-44 replay)",
    ),
    ("cd854692-d51b-11f0-a46a-c7e480a2e1a9", "SPEAKER_02"): (
        "Sky Galloway",
        "guest",
        None,
        "unknown",
        "The Gray Area with Sean Illing 'Is it bad to want to be rich?': says 'And "
        "I'm Sky Galloway. So we'll bring hot takes. I'll bring the insight. And the "
        "dick...' (#2365: Pivot promo, Scott Galloway (ASR 'Sky'))",
    ),
    ("cd854692-d51b-11f0-a46a-c7e480a2e1a9", "SPEAKER_05"): (
        "Phoebe Judge",
        "guest",
        None,
        "unknown",
        "The Gray Area with Sean Illing 'Is it bad to want to be rich?': says 'I'm "
        "Phoebe Judge, host of the podcast Criminal. Usually I'm telling stories "
        "about other...' (8.4 replay)",
    ),
    ("cd854692-d51b-11f0-a46a-c7e480a2e1a9", "SPEAKER_06"): (
        "Kara Swisher",
        "guest",
        None,
        "unknown",
        "The Gray Area with Sean Illing 'Is it bad to want to be rich?': says 'Hi, "
        "everyone. I'm Kara Swisher. Twice a week, we get together on Pivot to sort "
        "through...' (#2365: Pivot promo read by Kara Swisher)",
    ),
    ("8b954230-fc64-11f0-b7b7-a30da83d8f32", "SPEAKER_00"): (
        "Sky Galloway",
        "guest",
        None,
        "unknown",
        "The Gray Area with Sean Illing 'Our gender divide': says 'And I'm Sky "
        "Galloway. So we'll bring hot takes, I'll bring the insight, and the "
        "dick...' (#2365: Pivot promo, Scott Galloway (ASR 'Sky'))",
    ),
    ("8b954230-fc64-11f0-b7b7-a30da83d8f32", "SPEAKER_09"): (
        "Kara Swisher",
        "guest",
        None,
        "unknown",
        "The Gray Area with Sean Illing 'Our gender divide': says 'Hi, everyone. I'm "
        "Kara Swisher. Twice a week, we get together on Pivot to sort through...' "
        "(#2365: Pivot promo read by Kara Swisher)",
    ),
    ("8b954230-fc64-11f0-b7b7-a30da83d8f32", "SPEAKER_11"): (
        "Phoebe Judge",
        "guest",
        None,
        "unknown",
        "The Gray Area with Sean Illing 'Our gender divide': says 'I'm Phoebe Judge, "
        "host of the podcast, Criminal. Usually I'm telling stories about other...' "
        "(8.4 replay)",
    ),
    ("8b7fcdec-fc64-11f0-b7b7-37646c9e4077", "SPEAKER_02"): (
        "Phoebe Judge",
        "guest",
        None,
        "unknown",
        "The Gray Area with Sean Illing 'The death row assembly line': says 'I'm "
        "Phoebe Judge, host of the podcast Criminal. Usually I'm telling stories "
        "about other...' (8.4 replay)",
    ),
    ("8b7fcdec-fc64-11f0-b7b7-37646c9e4077", "SPEAKER_06"): (
        "Kara Swisher",
        "guest",
        None,
        "unknown",
        "The Gray Area with Sean Illing 'The death row assembly line': says 'Hi, "
        "everyone. I'm Kara Swisher. Twice a week, we get together on Pivot to sort "
        "through...' (#2365: Pivot promo read by Kara Swisher)",
    ),
    ("8b7fcdec-fc64-11f0-b7b7-37646c9e4077", "SPEAKER_09"): (
        "Sky Galloway",
        "guest",
        None,
        "unknown",
        "The Gray Area with Sean Illing 'The death row assembly line': says 'And I'm "
        "Sky Galloway. So we'll bring hot takes. I'll bring the insight and the "
        "dick...' (#2365: Pivot promo, Scott Galloway (ASR 'Sky'))",
    ),
    ("8b7fcdec-fc64-11f0-b7b7-37646c9e4077", "SPEAKER_11"): (
        None,
        "host",
        "Miles Bryan",
        "host",
        "The Gray Area with Sean Illing 'The death row assembly line': says 'Welcome "
        "to the Gray Area. I'm Vox reporter and producer Miles Bryan, sitting in for "
        "Sean...' (8.4 replay)",
    ),
    ("tag:soundcloud,2010:tracks/2135080263", "SPEAKER_02"): (
        "Young Lee",
        "guest",
        "Yung Lie",
        "guest",
        "The Long Run with Luke Timmerman 'Ep182: Andy Rachleff & Yung Lie on "
        "Creative...': says 'Thank you very much for having us. Or the CRISPR. You "
        "just did it. Can I repeat that? I...' (relabel-44 replay)",
    ),
    ("substack:post:191213865", "SPEAKER_01"): (
        None,
        "guest",
        "Jean Lee",
        "guest",
        "The Pragmatic Engineer 'Building WhatsApp with Jean Lee': says 'how did you "
        "get into tech? I've always been a small town girl. My dad was an OG "
        "hipster....' (8.4 replay)",
    ),
    ("substack:post:178618026", "SPEAKER_01"): (
        None,
        None,
        "Gergely Orosz",
        "host",
        "The Pragmatic Engineer 'Netflix’s Engineering Culture': says 'thing. So the "
        "way we approach it at Netflix is Netflix needs no introduction, but its...' "
        "(relabel-44 replay)",
    ),
    ("52822e96-8520-11f1-b415-a3e8f4de5e58", "SPEAKER_00"): (
        None,
        "host",
        "Tom Holland",
        "host",
        "The Rest Is History 'A Murderous Affair: Death at Mayerling (Part...': says "
        "'This episode is brought to you by the Lloyd's five k house deposit, and "
        "this is something...' (8.4 replay)",
    ),
    ("f575756a-4e1e-11f1-8d69-1b8a204015dc", "SPEAKER_00"): (
        "Tom Holland",
        "guest",
        "Tom Holland",
        "host",
        "The Rest Is History 'Greece vs. Persia: The Battle of Marathon...': says "
        "'This episode is brought to you by Lloyd's, which has been backing British "
        "ambition for...' (Tom Holland is the show's host)",
    ),
    ("f575756a-4e1e-11f1-8d69-1b8a204015dc", "SPEAKER_03"): (
        "William",
        "guest",
        None,
        "unknown",
        "The Rest Is History 'Greece vs. Persia: The Battle of Marathon...': says "
        "'On a rainy bridge over an Amsterdam canal, bearded Dutchmen in black capes "
        "and white...' (8.4 replay)",
    ),
    ("5c0bf304-adf5-11f1-a227-3f46219e3faa", "SPEAKER_04"): (
        None,
        "host",
        "Tom Holland",
        "host",
        "The Rest Is History 'Queen Victoria’s Revenge: The Mad Emperor of...': says "
        "'Not Bob the Bug Hunter? What the blazes is he doing in Abyssinia? We "
        "haven't got a...' (8.4 replay)",
    ),
    ("5c0bf304-adf5-11f1-a227-3f46219e3faa", "SPEAKER_05"): (
        None,
        None,
        "Dominic Sandbrook",
        "host",
        "The Rest Is History 'Queen Victoria’s Revenge: The Mad Emperor of...': says "
        "'So that's a lovely little nod to our Canadian listener, Mark Carney there, "
        "because I know...' (8.4 replay)",
    ),
    ("d0138942-a77e-11f1-9f5e-537d1277241d", "SPEAKER_00"): (
        None,
        "host",
        "Dominic Sandbrook",
        "host",
        "The Rest Is History 'Roman Civil War: Marching on the Senate...': says 'So "
        "it happened that Marius was selected to command the war against Mithridates "
        "in place...' (8.4 replay)",
    ),
    ("25c4bcb8-68c7-11f1-9fe0-c7cd9e192007", "SPEAKER_02"): (
        None,
        "host",
        "Dominic Sandbrook",
        "host",
        "The Rest Is History 'South Africa: Mandela and the Death of...': says 'This "
        "episode is brought to you by Lloyd's Business and Commercial Banking. Yeah, "
        "well,...' (8.4 replay)",
    ),
    ("25c4bcb8-68c7-11f1-9fe0-c7cd9e192007", "SPEAKER_03"): (
        None,
        None,
        "Tom Holland",
        "host",
        "The Rest Is History 'South Africa: Mandela and the Death of...': says 'One "
        "of the great things about finance is that it may result in you having to "
        "pay tax. And...' (8.4 replay)",
    ),
    ("01e704f8-69bd-11f1-b262-8fa334988761", "SPEAKER_01"): (
        None,
        None,
        "Paul Rouse",
        "guest",
        "The Rest Is History 'The Fascist World Cup: Mussolini's Football...': says "
        "'I'm ready. You're ready. Brilliant. So Paul, you were masquerading as a "
        "historian of...' (8.4 replay)",
    ),
    ("1a33b978-4df4-11f1-8551-03ebb7cd5a09", "SPEAKER_00"): (
        None,
        "host",
        "Tom Holland",
        "host",
        "The Rest Is History 'The First World War: Blood in the Trenches...': says "
        "'This episode is brought to you by Lloyds, which has been backing British "
        "ambition for...' (8.4 replay)",
    ),
    ("1a33b978-4df4-11f1-8551-03ebb7cd5a09", "SPEAKER_01"): (
        None,
        None,
        "Dominic Sandbrook",
        "host",
        "The Rest Is History 'The First World War: Blood in the Trenches...': says "
        "'Yes, great story, isn't it, Tom? A great lesson in leadership, I think, "
        "for anybody. So...' (8.4 replay)",
    ),
    ("94709434-542c-11f1-b8a0-136b7219bfa7", "SPEAKER_00"): (
        None,
        None,
        "Dominic Sandbrook",
        "host",
        "The Rest Is History 'The First World War: Italy’s Doomed Campaign...': says "
        "'Kingdom of England. Yes, great story, isn't it, Tom? A great lesson in "
        "leadership, I...' (8.4 replay)",
    ),
    ("94709434-542c-11f1-b8a0-136b7219bfa7", "SPEAKER_01"): (
        None,
        "host",
        "Tom Holland",
        "host",
        "The Rest Is History 'The First World War: Italy’s Doomed Campaign...': says "
        "'This episode is brought to you by Lloyds, which has been backing British "
        "ambition for...' (8.4 replay)",
    ),
    ("9b9c3b24-59b7-11f1-ae58-17e305fb826f", "SPEAKER_01"): (
        None,
        "host",
        "Tom Holland",
        "host",
        "The Rest Is History 'The First World War: The Spy Who Took on the...': says "
        "'This episode is brought to you by Lloyd's, which has been backing British "
        "ambition for...' (8.4 replay)",
    ),
    ("70824358-55e2-11f1-a9c2-abe6b164b5e5", "SPEAKER_01"): (
        None,
        None,
        "Dominic Sandbrook",
        "host",
        "The Rest Is History 'The First World War: The Submarine Strikes...': says "
        "'I think, for anybody. So Alfred and his heirs, they marry idealism and "
        "pragmatism....' (8.4 replay)",
    ),
    ("70824358-55e2-11f1-a9c2-abe6b164b5e5", "SPEAKER_02"): (
        None,
        "host",
        "Tom Holland",
        "host",
        "The Rest Is History 'The First World War: The Submarine Strikes...': says "
        "'This episode is brought to you by Lloyd's, which has been backing British "
        "ambition for...' (8.4 replay)",
    ),
    ("19263b9a-7ad2-11f1-aaed-3fa07ea54e9e", "SPEAKER_00"): (
        None,
        "host",
        "Dominic Sandbrook",
        "host",
        "The Rest Is History 'The Odyssey: Hero of the Trojan Horse (Part...': says "
        "'Tell me the tale of a man, muse, who had so many roundabout ways, to "
        "wander, driven off...' (8.4 replay)",
    ),
    ("19263b9a-7ad2-11f1-aaed-3fa07ea54e9e", "SPEAKER_01"): (
        None,
        None,
        "Tom Holland",
        "host",
        "The Rest Is History 'The Odyssey: Hero of the Trojan Horse (Part...': says "
        "'You've said that it's kind of foundational work of Western culture, but "
        "it's in all kinds...' (8.4 replay)",
    ),
    ("0322655e-b118-11f1-b8eb-f79d9c6ca5d1", "SPEAKER_04"): (
        None,
        "host",
        "Tom Holland",
        "host",
        "The Rest Is History 'The Terror: An Assassination in Paris (Part...': says "
        "'How long, Frenchman, will you put up with disorder and divisions? Oh, "
        "Frenchman, a little...' (8.4 replay)",
    ),
    ("5b3a78c0-b826-11f1-8f96-8ba30ba9f7aa", "SPEAKER_01"): (
        None,
        None,
        "Tom Holland",
        "host",
        "The Rest Is History 'The Terror: The Execution of Marie...': says 'wasn't "
        "it? Yeah, because the French monarchy was the most splendid, the most "
        "magnificent...' (8.4 replay)",
    ),
    ("5b3a78c0-b826-11f1-8f96-8ba30ba9f7aa", "SPEAKER_02"): (
        None,
        "host",
        "Dominic Sandbrook",
        "host",
        "The Rest Is History 'The Terror: The Execution of Marie...': says 'It is "
        "now 16 or 17 years since I saw the Queen of France, then the Dauphinesse, "
        "at...' (8.4 replay)",
    ),
    ("15c21496-5e7c-11f1-99e8-2be2f4812e1a", "SPEAKER_03"): (
        None,
        "host",
        "Tom Holland",
        "host",
        "The Rest Is History 'USA: The Star-Spangled Banner (Part 1)': says 'This "
        "episode is brought to you by Lloyd's Business and Commercial Banking. One "
        "of the...' (8.4 replay)",
    ),
    ("9b82fa12-1ef5-11f1-b40d-63850382fe73", "SPEAKER_11"): (
        "William Durampo",
        "guest",
        None,
        "unknown",
        "The Rest Is Politics: Leading 'The Only European Leader Defying Trump?...': "
        "says 'To others, he's a brutal despot, accused of presiding over more "
        "civilian deaths than...' (8.4 replay)",
    ),
    ("c6f32fd4-8028-11f1-85af-8769a81aeba0", "SPEAKER_00"): (
        "Katty Kaye",
        "guest",
        None,
        "unknown",
        "The Rest Is Science 'Forensic Science And The Champawat Tiger': says 'Hello "
        "and welcome to Doing It Anyway, a brand new series from Goalhanger. I'm "
        "Katty Kaye....' (8.4 replay)",
    ),
    ("b1fbfe40-271e-4c8c-be03-2070d007b5f8", "SPEAKER_00"): (
        None,
        "host",
        "Amjad Masad",
        "guest",
        "The a16z Show 'Amjad Masad on Going Direct, Building...': says 'Being "
        "canceled is a choice. You can choose to get canceled and retreat from the "
        "public...' (8.4 replay)",
    ),
    ("bcd869cb-a80e-475f-a044-7191309af9da", "SPEAKER_00"): (
        None,
        "host",
        "Chris Dixon",
        "guest",
        "The a16z Show 'Marc Andreessen and Chris Dixon: What’s at...': says "
        "'Stablecoins or tokenized stocks or dollars on blockchains, rivals the size "
        "of the Visa...' (the crypto guest at 51% of the talk, not a host)",
    ),
    ("bcd869cb-a80e-475f-a044-7191309af9da", "SPEAKER_03"): (
        "Mark Andreessen",
        "guest",
        "Marc Andreessen",
        "guest",
        "The a16z Show 'Marc Andreessen and Chris Dixon: What’s at...': says 'Yeah. "
        "Sure. So 2014. So 2014 was so long ago that the New York Times actually ran "
        "a...' (relabel-44 replay)",
    ),
    ("7cc1133c-61a7-4acb-8a73-6e4b7ba6228a", "SPEAKER_00"): (
        None,
        None,
        "Marc Rowan",
        "guest",
        "The a16z Show 'Marc Rowan on Private Markets, Software...': says '10 stocks "
        "right now in The US are nearly 50% of the S and P, and they're all levered "
        "to...' (relabel-44 replay)",
    ),
    ("c2b497da-b59b-41be-86a6-a966bd3ef724", "SPEAKER_02"): (
        None,
        None,
        "Max Levchin",
        "guest",
        "The a16z Show 'The $100B Niches Hiding Inside Payments': says 'The card "
        "payment interface is the singular best user interface ever created. It is "
        "the...' (8.4 replay)",
    ),
    ("6a2b0111622531b86448b15c", "SPEAKER_02"): (
        None,
        None,
        "Andrew England",
        "guest",
        "Unhedged 'Can Trump make a deal with Iran?': says 'It's been a bit crazy, "
        "but I never said it was a straight road. There are mediation...' (8.4 "
        "replay)",
    ),
    ("69e7c8e807ecece42a43e11e", "SPEAKER_01"): (
        None,
        None,
        "Robert Armstrong",
        "host",
        "Unhedged 'Trading disruption': says 'First of all, it is not gilet weather "
        "here. It is, like, 69 or 71 degrees and sunny and...' (8.4 replay)",
    ),
    ("69c2d3de1861d127d5274302", "SPEAKER_01"): (
        None,
        None,
        "Robert Armstrong",
        "host",
        "Unhedged 'Why are markets listening to Trump?': says 'and it was instant. "
        "And the question you have to ask about this is why are we still doing...' "
        "(8.4 replay)",
    ),
}


class _Episode(_VoiceEpisode):
    """0029's voice-by-voice rewrite, plus the ad-map of every transcript it re-renders.

    ``.adfree.admap.json`` holds offsets into the raw transcript (``source_length``, the pre- and
    post-roll cut points, the excised ranges and their texts). A renamed voice changes the speaker
    prefixes in that text, so without this the ad-map points beside the ads it describes, and 0030
    (which rebuilds ad-maps from the transcript) found 99 to rewrite after this ran: measured on
    the 2026-10-16 snapshot, 0030 then 0031. Each offset moves with the text around it: through
    the segment spans of a rendered transcript, through a character diff of one rewritten in place.
    An ad-map that does not describe the text on disk (an older rewrite left it), or an offset
    inside a rewritten label, leaves the ad-map alone, counted: 0030 rebuilds those.
    """

    def plan(self) -> bool:
        planned = super().plan()
        if planned:
            for path, text in list(self.text_files.items()):
                if path.name.endswith(".txt") and not path.name.endswith(".adfree.txt"):
                    self._carry_admap(path, text)
            self._recount()
        return planned

    def _recount(self) -> None:
        """The metadata's GI and KG counts, from the artifacts as rewritten (0030 does the same).

        The voice rewrite removes and adds KG people; left alone, ``knowledge_graph.node_count``
        and ``edge_count`` kept the old figures (28/30 against a graph of 27/28 on The Curiosity
        Shop's "Resisting Influence", 2026-10-16 snapshot)."""
        meta = self.json_files.get(self.meta) or _load(self.meta)
        if not isinstance(meta, dict):
            return
        before = json.dumps(meta, sort_keys=True)
        stem = str(self.meta)[: -len(".metadata.json")]
        for suffix, block in ((".gi.json", "grounded_insights"), (".kg.json", "knowledge_graph")):
            path = Path(stem + suffix)
            doc = self.json_files.get(path) or _load(path)
            counts = meta.get(block)
            if not isinstance(doc, dict) or not isinstance(counts, dict):
                continue
            nodes = doc.get("nodes") or []
            if block == "grounded_insights" and "insight_count" in counts:
                counts["insight_count"] = sum(1 for n in nodes if n.get("type") == "Insight")
            if block == "knowledge_graph" and doc.get("nodes") is not None:
                counts["node_count"] = len(nodes)
                counts["edge_count"] = len(doc.get("edges") or [])
        if json.dumps(meta, sort_keys=True) != before:
            self.json_files[self.meta] = meta
            self._bump("metadata_counts_synced")

    def _carry_admap(self, path: Path, new_text: str) -> None:
        admap_path = path.with_name(path.name[: -len(".txt")] + ".adfree.admap.json")
        admap = _load(admap_path)
        if not isinstance(admap, dict):
            return
        old_text = path.read_text(encoding="utf-8")
        if admap.get("source_length") != len(old_text) or old_text == new_text:
            self._bump("admaps_left_for_0030" if old_text != new_text else "admaps_unchanged")
            return
        seg_path = path.with_name(path.name[: -len(".txt")] + ".segments.json")
        new_rows = _rows(self.json_files.get(seg_path, _load(seg_path)))
        try:
            old_spans, _labels = _align(old_text, _rendered(_rows(_load(seg_path))))
            new_spans, _labels = _align(new_text, _rendered(new_rows))
            if len(old_spans) != len(new_spans):
                raise Refused("the renders have different segments")
        except Refused:
            opcodes = difflib.SequenceMatcher(
                None, old_text, new_text, autojunk=False
            ).get_opcodes()

            def moved(pos: Any) -> Optional[int]:
                return _move_by_diff(int(pos), opcodes, len(old_text), len(new_text))

        else:

            def moved(pos: Any) -> Optional[int]:
                return _move_offset(int(pos), old_spans, new_spans, len(old_text), len(new_text))

        out = dict(admap)
        out["source_length"] = len(new_text)
        points = [
            (key, moved(admap[key]))
            for key in ("preroll_cut_end", "postroll_cut_start")
            if isinstance(admap.get(key), int)
        ]
        moved_ranges = [(moved(a), moved(b)) for a, b in admap.get("excised_ranges") or []]
        ranges = [[a, b] for a, b in moved_ranges if a is not None and b is not None]
        if any(p is None for _k, p in points) or len(ranges) != len(moved_ranges):
            self._bump("admaps_left_for_0030")
            return
        out.update(points)
        out["excised_ranges"] = ranges
        out["excised_texts"] = [new_text[a:b] for a, b in ranges]
        if isinstance(admap.get("chars_removed"), int):
            old_cut = sum(int(b) - int(a) for a, b in admap.get("excised_ranges") or [])
            new_cut = sum(b - a for a, b in ranges)
            out["chars_removed"] = admap["chars_removed"] + new_cut - old_cut
        if out != admap:
            self.text_files[admap_path] = json.dumps(out, indent=2)
            self._bump("admaps_carried")


def _move_by_diff(pos: int, opcodes: List[Any], old_len: int, new_len: int) -> Optional[int]:
    """*pos* through a character diff: shifted in an unchanged stretch, kept at the start of a
    changed one, ``None`` inside it."""
    if pos == old_len:
        return new_len
    for tag, i1, i2, j1, _j2 in opcodes:
        if i1 <= pos < i2:
            return j1 + (pos - i1) if tag == "equal" else (j1 if pos == i1 else None)
    return None


def _move_offset(
    pos: int, old: List[Tuple[int, int]], new: List[Tuple[int, int]], old_len: int, new_len: int
) -> Optional[int]:
    """*pos* in the old render -> the same place in the new one, or ``None`` inside a label.

    In a segment's text it keeps its distance from the text's start. Between two texts (``" "`` or
    ``"\\nLabel: "``) the gap's first two characters (the line break, the label's first letter)
    and its end keep their place; anywhere else is inside a speaker label that was rewritten.
    """
    i = bisect.bisect_right([s for s, _e in old], pos) - 1
    if i >= 0 and pos <= old[i][1]:
        return new[i][0] + (pos - old[i][0])
    gap_start, new_gap_start = (old[i][1], new[i][1]) if i >= 0 else (0, 0)
    gap_end = old[i + 1][0] if i + 1 < len(old) else old_len
    new_gap_end = new[i + 1][0] if i + 1 < len(new) else new_len
    if pos == gap_end:
        return new_gap_end
    if pos - gap_start <= 1:
        return min(new_gap_start + (pos - gap_start), new_gap_end)
    return None


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class NamingVoicesRederivedMigration(Migration):
    """The names the deployed naming code gives the flagged served episodes, voice by voice."""

    id = MIGRATION_ID
    to_version = "2.7.25"
    description = (
        "the 166 voices in 126 served episodes that the deployed naming code names differently "
        "(read in the 2026-10-17 replay), rewritten voice by voice across metadata, segments, "
        "transcripts, GI, KG, bridge, diagnostics, turns and context, instead of a relabel"
    )

    def _scan(self, root: Path) -> Tuple[List[_Episode], List[_Episode], Dict[str, int]]:
        write: List[_Episode] = []
        refused: List[_Episode] = []
        totals: Dict[str, int] = {}
        for meta in select_served_artifacts(root, ".metadata.json")[0]:
            ep = _Episode(meta, root, NAMING_VOICES)
            if ep.plan():
                write.append(ep)
            if ep.refused:
                ep._bump("refused")
                refused.append(ep)
            for k, v in ep.counts.items():
                totals[k] = totals.get(k, 0) + v
        return write, refused, totals

    def plan(self, ctx: MigrationContext) -> str:
        """Count what apply() would rewrite and what it refuses; pure read, no writes."""
        write, refused, totals = self._scan(ctx.corpus_root)
        return f"naming voices plan: {len(write)} episode(s), {len(refused)} refused; " + ", ".join(
            f"{k}={v}" for k, v in sorted(totals.items())
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Pass when no listed voice is left to write and none was refused."""
        write, refused, _totals = self._scan(ctx.corpus_root)
        if write or refused:
            names = [ep.meta.name for ep in write + refused]
            return (
                False,
                f"{len(write)} episode(s) still to write, {len(refused)} refused: {names[:5]}",
            )
        return True, "every listed voice carries the name the deployed naming code gives it"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Rewrite every episode with a listed voice still on its old name, with backups."""
        root = ctx.corpus_root
        write, refused, totals = self._scan(root)
        receipts: List[dict] = []
        if not ctx.dry_run:
            for ep in write:
                for path, payload in sorted(ep.json_files.items()):
                    receipts.append(write_with_backup(root, BACKUP_TAG, path, payload))
                for path, text in sorted(ep.text_files.items()):
                    receipts.append(_write_text_with_backup(root, path, text, BACKUP_TAG))
            append_receipts(root, RECEIPTS_FILE, {"migration": MIGRATION_ID}, receipts)
        verb = "would rewrite" if ctx.dry_run else "rewrote"
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=f"{verb} {len(write)} episode(s), refused {len(refused)}; "
            + ", ".join(f"{k}={v}" for k, v in sorted(totals.items())),
            details={
                "episodes": [
                    {"meta": str(ep.meta.relative_to(root)), "voices": ep.voice_names}
                    for ep in write
                ],
                "refused": [
                    {"meta": str(ep.meta.relative_to(root)), "why": ep.refused} for ep in refused
                ],
                "totals": totals,
                "files_written": len(receipts),
            },
        )


def dry_run_report(root: Path) -> Iterable[Dict[str, Any]]:
    """Read-only: one row per episode the migration would write or refuses."""
    write, refused, _totals = NamingVoicesRederivedMigration()._scan(Path(root))
    for ep in write + refused:
        yield {
            "meta": str(ep.meta.relative_to(root)),
            "voices": ep.voice_names,
            "refused": ep.refused,
            "files": sorted(
                str(p.relative_to(root)) for p in list(ep.json_files) + list(ep.text_files)
            ),
        }
