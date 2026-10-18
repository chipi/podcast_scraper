"""0029 — the voices this deploy's naming ladder names differently, on the records already written.

#2334 (S7 of #2304). The naming ladder merged in #2337 changes who some voices are on NEW episodes;
the served records still carry what the code before it wrote. This applies exactly those changes,
and nothing else.

Why a frozen list and not a re-run of naming. Measured on prod 2026-10-12, the stored records and a
replay of the merged code differ on 994 voices in 775 episodes, 486 of them names the replay
REMOVES: August-September arithmetic names (the feed's host forced onto a seat, a self-intro) that
today's rules withhold on purpose, many of them correct (Gergely Orosz, Lenny Rachitsky). A blanket
re-run would delete them. So the list is the voices whose stored record still equals what the
DEPLOYED code (sha-3e06151) produces AND that the merged code (ea1af916f) changes: 100 voices in 89
episodes, replayed on one snapshot and each read against the voice's own words. In: 83. Out: the
People I (Mostly) Admire announcer that the merged code would name Steve Levitt (a K13 known
issue), and every role-only change on a voice that stays unnamed (the voice-by-voice rewrite
carries a role only with a name; three of those were wrong, docs/guides/NAMING_KNOWN_ISSUES.md
K13).

``NAMING_ARC_VOICES``: ``(episode id, voice): (stored name, stored role, new name, new role,
evidence)``. ``None`` as the new name unnames the voice. An entry applies only while the voice still
carries the stored name and role; once it carries the new ones it is done (a second run writes
nothing); anything else REFUSES the episode, untouched and listed, because the record moved on since
the replay.

Every surface is rewritten voice by voice with m0025's machinery (metadata, segments, transcripts
with every offset carried, GI, KG, bridge, diagnostics, turns, context), all or none per episode.
``verify`` fails while anything is left to write or refused. UNDO restores from
``.podcast_scraper/upgrade-backups/0029/`` while each file is still what this migration wrote.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..corpus_selection import select_served_artifacts
from ..file_rewrite import append_receipts, undo_from_receipts, write_with_backup
from ..migration import Migration, MigrationContext, MigrationResult
from .m0023_transcript_speaker_prefixes_resynced import _write_text_with_backup
from .m0025_one_person_one_entry import _Episode as _VoiceEpisode, _load, _run_relative

MIGRATION_ID = "0029_naming_arc_voices_repaired"
RECEIPTS_FILE = "naming_arc_voices_repaired.jsonl"
BACKUP_TAG = "0029"
_DIARIZATION_LABEL = re.compile(r"^SPEAKER(_\d+)?$")

Entry = Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str]

#: Read on 2026-10-12 against each voice's own transcript lines (the evidence is its first line).
NAMING_ARC_VOICES: Dict[Tuple[str, str], Entry] = {
    ("brilliantexperience.podbean.com/bf193d4f-cce7-37f6-9337-67c8154daa8c", "SPEAKER_01"): (
        "John",
        "guest",
        "John Whalen",
        "host",
        "AI 4 UX with John Whalen, Ph 'AI Meets UX: Enhancing Surveys with Voice - "
        "Inte': says 'Hi, and welcome to AI for UX. This is the space where UX "
        "researchers and product leaders s'",
    ),
    ("brilliantexperience.podbean.com/bf193d4f-cce7-37f6-9337-67c8154daa8c", "SPEAKER_02"): (
        "Philip Brook",
        "host",
        "Philip Brook",
        "guest",
        "AI 4 UX with John Whalen, Ph 'AI Meets UX: Enhancing Surveys with Voice - "
        "Inte': says 'your product and the kinds of things it addresses? Definitely. "
        "Well, why don't I dive into'",
    ),
    ("brilliantexperience.podbean.com/0075814b-f7ea-3e09-978b-167c4c35eca0", "SPEAKER_00"): (
        "John",
        "guest",
        "John Whalen",
        "host",
        "AI 4 UX with John Whalen, Ph 'AI as Your Research Partner, Not a Replacement "
        "-': says 'Hi, and welcome to AI for UX. This is the space where UX researchers "
        "and product leaders s'",
    ),
    ("brilliantexperience.podbean.com/0075814b-f7ea-3e09-978b-167c4c35eca0", "SPEAKER_01"): (
        "Tanya Berlina",
        "host",
        "Tanya Berlina",
        "guest",
        "AI 4 UX with John Whalen, Ph 'AI as Your Research Partner, Not a Replacement "
        "-': says 'much for having me here. And thank you for everyone who is watching "
        "us. And I'm sending yo'",
    ),
    ("brilliantexperience.podbean.com/0f8246ce-5eb3-3ac1-92d7-a1e6636e4aed", "SPEAKER_01"): (
        "John",
        "guest",
        "John Whalen",
        "host",
        "AI 4 UX with John Whalen, Ph 'Can AI-moderated research best current methods? "
        "': says 'Welcome to AI for UX. This is a space where UX researchers and "
        "product leaders stay curren'",
    ),
    ("brilliantexperience.podbean.com/0f8246ce-5eb3-3ac1-92d7-a1e6636e4aed", "SPEAKER_02"): (
        "Alfred Wahlforss",
        "host",
        "Alfred Wahlforss",
        "guest",
        "AI 4 UX with John Whalen, Ph 'Can AI-moderated research best current methods? "
        "': says 'to do this interview and talk with us today. Yeah, thank you for "
        "having me. We're really'",
    ),
    ("067bc3cc-1fcc-4684-8e27-c0b4d72b2222", "SPEAKER_01"): (
        "Sophie Dionnet",
        "host",
        "Sophie Dionnet",
        "guest",
        "Analyse Podcast 'The Three Ingredients That Turn AI Into Value wi': says 'A "
        "lot of the changes that organizations need to do today actually don't require "
        "the latest'",
    ),
    ("067bc3cc-1fcc-4684-8e27-c0b4d72b2222", "SPEAKER_00"): (
        "Brennan Leung",
        "guest",
        "Bernard Leong",
        "host",
        "Analyse Podcast 'The Three Ingredients That Turn AI Into Value wi': says "
        "'know. Welcome to Analyze Podcast, the premier podcast dedicated to dissecting "
        "the pulse of'",
    ),
    ("6e692abb-56b7-4385-8bab-84652943c93d", "SPEAKER_01"): (
        None,
        None,
        "Tim Wu",
        "guest",
        "Capitalisn't 'Are Big Tech’s Regulators “Cowards”? ft. Tim Wu': says 'There "
        "was a cultural campaign to make people who believe in structural separation,'",
    ),
    ("162efd2f-7685-42f4-b927-f51e1830d632", "SPEAKER_00"): (
        None,
        None,
        "Dan Rasmussen",
        "guest",
        "Capitalisn't 'Trump's Great Private Equity Bailout, with Dan R': says 'The S&P "
        "500 is at all-time highs, and the stock market is booming, and the economy "
        "seems t'",
    ),
    ("fdfc0bbc-cc66-465c-bfd2-dfee6fdca2b5", "SPEAKER_02"): (
        None,
        None,
        "Carl Frey",
        "guest",
        "Capitalisn't 'Why Human Progress Is Not Inevitable - ft. Carl ': says 'I'm not "
        "of the view that every technology is good, but unfortunately, many times it's "
        "hard'",
    ),
    ("substack:post:190839917", "SPEAKER_00"): (
        None,
        None,
        "Dylan Patel",
        "guest",
        "Dwarkesh Podcast 'Dylan Patel — Deep dive on the 3 big bottlenecks': says 'All "
        "right. This is the episode of My Roommate Teaches Me Semiconductors.'",
    ),
    ("substack:post:191582481", "SPEAKER_00"): (
        None,
        None,
        "Terence Tao",
        "guest",
        "Dwarkesh Podcast 'Terence Tao – Kepler, Newton, and the true natur': says "
        "'Okay, yeah. So I've always had an amateur interest in astronomy,'",
    ),
    ("26c8b4a4-6a87-407b-a53c-845ef00225c0", "SPEAKER_01"): (
        None,
        None,
        "Mike Munger",
        "guest",
        "EconTalk 'Public Choice and Politics Without Romance (with': says 'The 52nd "
        "episode, I want to assure you that I feel all the honor that the situation "
        "merits'",
    ),
    ("751294f4-baee-4962-b2dc-7c42433c9f42", "SPEAKER_01"): (
        None,
        None,
        "Rowan Jacobsen",
        "guest",
        "EconTalk 'The Case for Sunshine (with Rowan Jacobsen)': says 'we are now at a "
        "point where enough science is filled in that we need to reconsider'",
    ),
    ("a8362633-c109-4342-a5d8-d7aa53ea48d5", "SPEAKER_01"): (
        None,
        None,
        "Rebecca Newberger Goldstein",
        "guest",
        "EconTalk 'The Mattering Instinct (with Rebecca Newberger G': says 'It's a "
        "pleasure to be here.'",
    ),
    ("ac146ab1-80d5-43ac-8256-5c9f8e752faa", "SPEAKER_01"): (
        None,
        None,
        "Rachel Cockerell",
        "guest",
        "EconTalk 'Zionism, the Melting Pot, and the Galveston Proj': says 'I really "
        "just wanted to write a nice, normal book.'",
    ),
    ("b8184db1-1299-435d-a66f-8dcf87b49b70", "SPEAKER_01"): (
        None,
        None,
        "Dr. Sarah Chan",
        "guest",
        "FUTURES Podcast 'Bioethics of Human & Animal Enhancement w/ Dr. S': says "
        "'bioethicist Dr. Sarah Chan. What is enhancement and what means do we have "
        "realistically, n'",
    ),
    ("substack:post:159080671", "SPEAKER_00"): (
        None,
        None,
        "Katelyn Jetelina",
        "guest",
        "Ground Truths 'Katelyn Jetelina: Public health in the US': says 'Yeah, you "
        "know, he kind of came out of nowhere.'",
    ),
    ("substack:post:172589898", "SPEAKER_01"): (
        None,
        None,
        "Peter Hotez",
        "guest",
        "Ground Truths 'Peter Hotez & Eric Topol - Discuss “Science Unde': says 'Well, "
        "first of all, Eric, thank you.'",
    ),
    ("0270f621-aa0b-4262-90d5-b9cdc4604dee", "SPEAKER_06"): (
        None,
        None,
        "Jensen Huang",
        "guest",
        "Hard Fork 'The Ezra Klein Show: Jensen Huang Thinks A.I. Al': says 'Well, "
        "first of all, it's a new industrial revolution.'",
    ),
    ("5c90db99-3390-4327-83c3-cb9a15507412", "SPEAKER_01"): (
        None,
        None,
        "Aishwarya Kekre",
        "guest",
        "Ideas of India 'Aishwarya Kekre on Land Fragmentation and Its Im': says 'Hi, "
        "Shruti. I am very excited to be here. Thank you for having me on the Job "
        "Market Series'",
    ),
    ("8ff9162f-9f56-46b1-9663-73fe4f90d885", "SPEAKER_01"): (
        None,
        None,
        "Chetana Sabnis",
        "guest",
        "Ideas of India 'Chetana Sabnis on The Intimacy Contract and the ': says 'Hi, "
        "Chetna. Welcome to the show. It is so lovely to have you here.'",
    ),
    ("b36d42b2-42fc-4723-be93-8187189dc793", "SPEAKER_00"): (
        None,
        None,
        "Kartik Srivastava",
        "guest",
        "Ideas of India 'Kartik Srivastava on Referral-Based Hiring, Cast': says 'Hi, "
        "Shruti. Great to be here. Thank you.'",
    ),
    ("9505619a-009e-11f1-869d-c7d5bc43b194", "SPEAKER_01"): (
        None,
        None,
        "Ben Horowitz",
        "guest",
        "Invest Like the Best with Pa 'Ben Horowitz - Backing America’s Future': says "
        "'Patrick O'Shaughnessy is the CEO of Positive Sum. All opinions expressed by "
        "Patrick and po'",
    ),
    ("d0acd199-0d4d-4618-8a34-fafe7182ad54", "SPEAKER_00"): (
        None,
        None,
        "Philip Ahn Cuddy",
        "guest",
        "Korea Deconstructed 'Who Was Ahn Chang Ho (안창호)? | Conversation with ': says "
        "'I think when you look at the people who really, you know, sacrificed their "
        "lives and were'",
    ),
    ("substack:post:198688585", "SPEAKER_01"): (
        "Ivan Burazin",
        "host",
        "Ivan Burazin",
        "guest",
        "Latent Space: The AI Engineer Podcast 'Giving Agents Computers — Ivan Burazin, "
        "Daytona': says 'you found like, we did you reach out or for Shift? Or I "
        "reached out to you. The reason was'",
    ),
    ("substack:post:199607874", "SPEAKER_01"): (
        "Walden Yan",
        "host",
        "Walden Yan",
        "guest",
        "Latent Space: The AI Engineer Podcast 'The Age of Async Agents — Cognition's Walden "
        "Yan': says 'one a coiner of context engineering? Yes. Yes. Although I I I "
        "think there are many people '",
    ),
    ("substack:post:199994886", "SPEAKER_00"): (
        "Carina Hong",
        "host",
        "Carina Hong",
        "guest",
        "Latent Space: The AI Engineer Podcast '🔬Scaling Past Informal AI - Carina Hong, Axiom "
        "M': says 'But it's for the first time now, I think, verified AI is to open up "
        "collaboration.'",
    ),
    ("substack:post:199994886", "SPEAKER_01"): (
        "Brandon Anderson",
        "guest",
        "Brandon Anderson",
        "host",
        "Latent Space: The AI Engineer Podcast '🔬Scaling Past Informal AI - Carina Hong, Axiom "
        "M': says 'He was already a really strong one, but verification helps him "
        "extend the brilliance, like'",
    ),
    ("substack:post:188846384", "SPEAKER_00"): (
        None,
        None,
        "Jenny Wen",
        "guest",
        "Lenny's Podcast: Product | C 'The design process is dead. Here’s what’s "
        "replac': says 'This design process that designers have been taught, we sort of "
        "treat it as gospel. That's'",
    ),
    ("substack:post:182877855", "SPEAKER_02"): (
        "Molly Graham",
        "guest",
        None,
        "unknown",
        "Lenny's Podcast: Product | C 'The high-growth handbook: Molly Graham’s "
        "framewo': says 'Today's episode is brought to you by DX, the developer "
        "intelligence platform designed by l'",
    ),
    ("substack:post:184167026", "SPEAKER_02"): (
        "Zevi Arnewitz",
        "guest",
        None,
        "unknown",
        "Lenny's Podcast: Product | C 'The non-technical PM’s guide to building with "
        "Cu': says 'This episode is brought to you by ten web, the company that "
        "pioneered AI website building '",
    ),
    ("substack:post:184167026", "SPEAKER_00"): (
        None,
        None,
        "Zevi Arnovitz",
        "guest",
        "Lenny's Podcast: Product | C 'The non-technical PM’s guide to building with "
        "Cu': says 'You are a product manager shipping product without knowing how to "
        "write code, barely knowi'",
    ),
    ("substack:post:189787695", "SPEAKER_01"): (
        "Jacob Warwick",
        "host",
        "Lenny Rachitsky",
        "host",
        "Lenny's Podcast: Product | C 'The tactical playbook for getting 20-40% more "
        "co': says 'Today, my guest is Jacob Warwick. Jacob is a professional "
        "negotiator.'",
    ),
    ("454a0a66-4955-11f1-adf9-2fa0e93c779d", "SPEAKER_00"): (
        None,
        None,
        "Harrison Chase",
        "guest",
        "NVIDIA AI Podcast 'Harrison Chase of LangChain on Deep Agents, Lang': says "
        "'And so I think, like, these always on asynchronous event driven agents, that "
        "will be a rea'",
    ),
    ("1683311e-1dcf-11f1-b1e2-fbd9c92e8464", "SPEAKER_00"): (
        None,
        None,
        "Simon Last",
        "guest",
        "No Priors: Artificial Intell 'From Coder to Manager: Navigating the Shift to "
        "A': says 'I I've been watching, you know, what's going on. In general, I've "
        "just been, like, super c'",
    ),
    ("9b87c7ba-12e6-11f1-b442-a76c7c1ba652", "SPEAKER_01"): (
        "Neil Tiwari",
        "host",
        None,
        "host",
        "No Priors: Artificial Intell 'How Capital is Powering the AI Infrastructure "
        "Bu': says 'listeners. Welcome back to No Priors. Today, I'm here with Neil "
        "Tewari of Magnetar Capital'",
    ),
    ("0215da30-a479-11f0-b972-17ab202e9eea", "SPEAKER_01"): (
        "Eric Zelikman",
        "host",
        None,
        "host",
        "No Priors: Artificial Intell 'Humans&: Bridging IQ and EQ in Machine Learning "
        "': says 'listeners. Welcome back to No Priors. Today, we're here with Eric "
        "Zelkman,'",
    ),
    ("4203e26a-af7b-11f0-82a3-ffff032a4199", "SPEAKER_01"): (
        "Elad Gil",
        "host",
        "Zach Lloyd",
        "guest",
        "No Priors: Artificial Intell 'Reinventing the Developer Terminal with Warp "
        "Co-': says 'societal impacts will be? The way I think about, like, the "
        "advances are it's kinda like we'",
    ),
    ("678b0abc-a94b-11f0-9c77-f7212adc0484", "SPEAKER_01"): (
        "Zach Dell",
        "host",
        "Zach Dell",
        "guest",
        "No Priors: Artificial Intell 'Unlocking the Road to Energy Abundance with "
        "Base': says 'a lot of the industry, the value chain of of energy'",
    ),
    ("2f8edaf4-9698-11f1-bdc1-eb068382bf86", "SPEAKER_00"): (
        None,
        "host",
        "Erik Allebest",
        "guest",
        "No Priors: Artificial Intell 'What Chess.com Teaches US About Superhuman "
        "Capab': says 'COVID happened. Queen's Gambit happened. Some news cycles "
        "happened. And we're like, oh, th'",
    ),
    ("2f8edaf4-9698-11f1-bdc1-eb068382bf86", "SPEAKER_01"): (
        "Erik Allebest",
        "host",
        None,
        "host",
        "No Priors: Artificial Intell 'What Chess.com Teaches US About Superhuman "
        "Capab': says 'Hi, listeners. Welcome back to No Priors.'",
    ),
    ("b1fa9369-e153-4071-8b62-b47400f4a30f", "SPEAKER_03"): (
        "Jill Wisenthal",
        "guest",
        "Joe Weisenthal",
        "host",
        "Odd Lots 'How Lenovo's CFO Is Allocating Capital During On': says 'And I'm "
        "Jill Wisenthal.'",
    ),
    ("gid://art19-episode-locator/V0/8wT46qX06HADJaFeOlyA2UZY5KB_zpC5JF3n_Q9sijs", "SPEAKER_01"): (
        None,
        None,
        "Catherine Liu",
        "guest",
        "Philosophy For Our Times 'Taking down the cultural elite | Catherine Liu': "
        "says 'What else is there if you're not paid a proper wage?'",
    ),
    ("0d6638fb-a3e7-4624-a4b1-b37ba4a88821", "SPEAKER_01"): (
        "Alexander Carpi",
        "guest",
        "Alexandra Karppi",
        "host",
        "Talk Eastern Europe 'Russia Escalating, Serbia's Election Test | Week': says "
        "'Germany accused Russia of some kind of involvement'",
    ),
    ("bccd2500-0223-4c8a-ac68-2e929a175cc4", "SPEAKER_00"): (
        "Alexander Carpi",
        "guest",
        "Alexandra Karppi",
        "host",
        "Talk Eastern Europe 'Russia’s Election & NATO’s Eastern Flank: What C': says "
        "'Welcome back to the Talk Eastern Europe weekly news roundup,'",
    ),
    ("d20f6d64-77ea-47ae-93ad-292bcc163537", "SPEAKER_03"): (
        None,
        None,
        "Deborah Kamen",
        "guest",
        "The Daily 'The Firestorm Over a Rape Allegation at Cornell': says 'You know, "
        "Rachel, I've been covering sexual assault and sexual harassment for many "
        "years,'",
    ),
    ("741c8a11-b0f7-48e0-bf40-76de35bc29ff", "SPEAKER_02"): (
        None,
        None,
        "Michael Gold",
        "guest",
        "The Daily 'The Landmark Housing Bill That Trump Refuses to ': says 'Michael "
        "Gold, always a pleasure. Happy to be here, Michael. Thanks.'",
    ),
    ("db9dbb14-4257-4c04-936f-359a0a978c67", "SPEAKER_02"): (
        None,
        None,
        "David E. Sanger",
        "guest",
        "The Daily 'Why the Cease-Fire With Iran Keeps Crumbling': says 'There's not "
        "much of it left.'",
    ),
    ("d66e786a-9431-4fb0-a332-f76779f51fd7", "SPEAKER_01"): (
        "New York City",
        "guest",
        None,
        "unknown",
        "The Daily 'Zohran Mamdani Knows He Has Political Capital. A': says 'some of my "
        "colleagues cheering in in I think there's about twenty minutes left in the "
        "Fran'",
    ),
    ("ccfba8b3-e1ab-4df5-9714-e570baff1961", "SPEAKER_01"): (
        None,
        "host",
        "Justin Norman",
        "host",
        "The Flip 'African Small Businesses Have Many Challenges. C': says 'Hi there. "
        "Justin here, the founder of The Flip. And before we start, I want to first say "
        "t'",
    ),
    ("fbc4b8e7-0333-488d-ab49-604ac46974b9", "SPEAKER_02"): (
        None,
        "host",
        "Justin Norman",
        "host",
        "The Flip 'I Went to Bolivia Where Businesses Survive on St': says 'This guy is "
        "about to sell me local currency for double the official rate, but he's not'",
    ),
    ("93ccb619-fd1e-4693-89da-da481dc36d3e", "SPEAKER_00"): (
        None,
        "host",
        "Justin Norman",
        "host",
        "The Flip 'The Future of Work Will Be Bootstrapped': says 'Hi there, Justin "
        "here, the founder of The Flip.'",
    ),
    ("5555e937-4e0a-4e17-abc3-68f5b77e5501", "SPEAKER_00"): (
        None,
        None,
        "Justin Norman",
        "host",
        "The Flip 'Unlocking Gender-Smart Capital At Scale (2X Glob': says 'The 2X "
        "Challenge has been incredibly successful in transforming mainstream fund "
        "managers t'",
    ),
    ("5125f8f2-9724-4f85-9a6d-f0a9c7a5e41a", "SPEAKER_07"): (
        None,
        None,
        "Justin Norman",
        "host",
        "The Flip 'What China Taught Me About Global Trade': says 'by being able to "
        "find markets overseas for its manufactured goods'",
    ),
    ("b769a5e0-4748-47f6-b9a3-230677e033a4", "SPEAKER_04"): (
        None,
        "host",
        "Justin Norman",
        "host",
        "The Flip 'What Vietnam’s Gold Shops Taught Me About Money': says 'The price of "
        "gold has soared to a new record, risen above $5,000 an hour.'",
    ),
    ("2cfba12e-e38e-4df9-a3d7-2a20565e1247", "SPEAKER_07"): (
        None,
        "host",
        "Justin Norman",
        "host",
        "The Flip 'What We Get Wrong About Jobs in Africa': says 'Hi there. Justin "
        "here, the founder of The Flip. And before we start, I want to first say t'",
    ),
    ("e52aca03-ed61-40e1-88e8-7b27006fd4a1", "SPEAKER_02"): (
        None,
        "host",
        "Justin Norman",
        "host",
        "The Flip 'Why the NFL is Bringing American Football to Afr': says 'Here in "
        "Lagos, Nigeria, these athletes are vying for a shot at the NFL to become the "
        "next '",
    ),
    ("252b2cf8-98d3-4a07-aa67-d218dfe74a22", "SPEAKER_00"): (
        None,
        None,
        "Layman Pascal",
        "guest",
        "The Living Philosophy '#9 Layman Pascal - Metashamanic Nietzsche': says "
        "'That's all right. I don't hate that.'",
    ),
    ("45581c22-65b4-4d2b-9abe-87d975d5c144", "SPEAKER_01"): (
        None,
        None,
        "Michael Montgomery",
        "guest",
        "The Living Philosophy '🎙️#10 Michael Montgomery: Psychophobia and Bridg': says "
        "'Hi. Great to be here.'",
    ),
    ("9d0e0f69-af6c-4f05-8f2a-3139bb38cafc", "SPEAKER_01"): (
        "Dr. Sharon Blackie",
        "host",
        "Dr. Sharon Blackie",
        "guest",
        "The Living Philosophy '🎙️#17 Dr. Sharon Blackie: Can We Re-enchant a Di': says "
        "'Thank you for inviting me. It's a pleasure. That sounds like a lot of "
        "ologies.'",
    ),
    ("557e200ecdc72f5714593ffe1bde", "SPEAKER_01"): (
        None,
        None,
        "Simon Shuster",
        "guest",
        "The Naked Pravda 'Simon Shuster on the fall of Andriy Yermak': says 'I was "
        "surprised, even after speaking to him exactly a week ago, last Thursday.'",
    ),
    ("6507b402-caaa-43e2-9e1e-23073b3b51de", "SPEAKER_01"): (
        None,
        None,
        "Brian Grosberg",
        "guest",
        "The Peter Attia Drive '#406 ‒ Migraine, cluster headache, and tension h': says "
        "'Yes. It makes me a unicorn.'",
    ),
    ("substack:post:168964444", "SPEAKER_00"): (
        None,
        None,
        "Laura Tacho",
        "guest",
        "The Pragmatic Engineer 'Measuring the impact of AI on software engineeri': "
        "says 'What happens with the time saving? As a developer, what would that mean "
        "for me as a develo'",
    ),
    ("substack:post:178618026", "SPEAKER_00"): (
        "Elizabeth Stone",
        "host",
        "Elizabeth Stone",
        "guest",
        "The Pragmatic Engineer 'Netflix’s Engineering Culture': says 'What is the "
        "scale of the company from an engineer? When you add that up, we have more "
        "than'",
    ),
    ("substack:post:165580001", "SPEAKER_01"): (
        None,
        None,
        "Kent Beck",
        "guest",
        "The Pragmatic Engineer 'TDD, AI agents and coding with Kent Beck': says "
        "'things that Genie missed in terms of tests. Today, I was working on the small "
        "talk parser '",
    ),
    ("substack:post:170111910", "SPEAKER_00"): (
        None,
        None,
        "Peter Walker",
        "guest",
        "The Pragmatic Engineer 'The state of VC within software and AI startups ': "
        "says 'far fewer people than they used to. In January 2022,'",
    ),
    ("4d691819-3665-4d16-aee1-8655c9322f9b", "SPEAKER_00"): (
        None,
        None,
        "Adam Neumann",
        "guest",
        "The a16z Show 'Adam Neumann: This Is How You Build Iconic Compa': says 'You'd "
        "said, Adam, tell me how you're doing, and I jumped straight into my lessons "
        "learned,'",
    ),
    ("1f41bb8a-925a-4d3e-95fc-f3790db1619e", "SPEAKER_00"): (
        None,
        None,
        "Robin Hanson",
        "guest",
        "The a16z Show 'Robin Hanson on Prediction Markets, Gambling, an': says 'start "
        "at the beginning. The basic vision is that speculative markets are shown to be "
        "a unm'",
    ),
    ("7ef41fcf-3127-4992-ad89-e3ea46e3ee78", "SPEAKER_00"): (
        None,
        None,
        "Ruby Thelot",
        "guest",
        "The a16z Show 'Ruby Thelot on Internet Culture, AI, and the Fut': says "
        "'Americans don't like AI. Mhmm. AI is a very broad term. They all use chat.'",
    ),
    ("916c5e0a-b2ab-4935-ae27-7dc4eeb8612a", "SPEAKER_00"): (
        None,
        None,
        "Samo Burja",
        "guest",
        "The a16z Show 'Samo Burja on Growth, Energy, and AI': says 'So I actually "
        "think a big macro story that we've been exploring at Bismarck Analysis and B'",
    ),
    ("f785186b-12b9-4336-95e5-4242cc0b6ddb", "SPEAKER_00"): (
        None,
        None,
        "Dileep Thazhmon",
        "guest",
        "The a16z Show 'Stablecoins, AI Agents, and The Future of Global': says 'The "
        "goal of Jeeves is really to build a global business bank that can function in "
        "multiple'",
    ),
    ("ef54f007-9c46-4d15-b039-1183f989a26d", "SPEAKER_00"): (
        None,
        None,
        "Steven Sinofsky",
        "guest",
        "The a16z Show 'Steven Sinofsky: AI Doesn't Need New Rules Yet': says 'The "
        "whole topic of regulation for me just seems'",
    ),
    ("2eba7469-414c-4103-a091-bbdbf12bd835", "SPEAKER_02"): (
        "Consumer AI Ads",
        "guest",
        None,
        "unknown",
        "The a16z Show 'The Top 100 Consumer AI Apps: Who’s Actually Pay': says 'Half "
        "of Americans now report using AI, but only a small fraction are actually "
        "paying for i'",
    ),
    ("turkeybooktalk.podbean.com/ab440b57-8e89-36fb-bd0b-e29ab8237637", "SPEAKER_01"): (
        None,
        None,
        "Brett Wilson",
        "guest",
        "Turkey Book Talk 'Brett Wilson on the myth and reality of modern T': says "
        "'were its implications? The 1925 law is one of the hallmark Kemalist reforms "
        "of the 1920s a'",
    ),
    ("turkeybooktalk.podbean.com/0c63be09-6442-3539-ac8f-a197de02934c", "SPEAKER_01"): (
        None,
        None,
        "Faisal Devji",
        "guest",
        "Turkey Book Talk 'Faisal Devji on the crisis of 'global Islam' in ': says 'In "
        "a way, the subtitle goes with the title, obviously, and the title, Waning "
        "Crescent, is '",
    ),
    ("turkeybooktalk.podbean.com/775677b0-5a19-38d0-a37d-7315c8996f5b", "SPEAKER_01"): (
        None,
        None,
        "Hannah Lucinda Smith",
        "guest",
        "Turkey Book Talk 'Hannah Lucinda Smith on Turkey, Russia and Europ': says 'her "
        "to write it. This was a book that actually took a really long time to get "
        "commissioned'",
    ),
    ("turkeybooktalk.podbean.com/70f1424a-edfb-3777-bff9-7feeb4dadcd7", "SPEAKER_01"): (
        None,
        None,
        "I\u015f\u0131lay G\u00fcrsu",
        "guest",
        "Turkey Book Talk 'Işılay Gürsu on archaeology and public perceptio': says "
        "'There was a lot of interest towards archaeology in the early republican "
        "period,'",
    ),
    ("turkeybooktalk.podbean.com/3fdee49b-80cd-39e2-8217-9ab3cb585c16", "SPEAKER_01"): (
        None,
        None,
        "Jennifer Hattam",
        "guest",
        "Turkey Book Talk 'Jennifer Hattam on writing the Lonely Planet gui': says "
        "'Yeah, sure. Thanks for having me, Will. Always a pleasure. And it was fun to "
        "kind of go ba'",
    ),
    ("turkeybooktalk.podbean.com/f9412b5a-a368-36ed-ae9d-43146d42f609", "SPEAKER_01"): (
        None,
        None,
        "Suzy Hansen",
        "guest",
        "Turkey Book Talk 'Suzy Hansen on Turkey and an Istanbul neighborho': says "
        "'explore the issues that she addresses in the book? So as you know, Karagumruk "
        "is in Fati. '",
    ),
    ("6f808a20-9589-11f1-bde4-97c443817848", "SPEAKER_01"): (
        "Peter Hitchens",
        "host",
        "Peter Hitchens",
        "guest",
        "Unbelievable? 'Peter Hitchens on Britain’s decline, Christian f': says 'I "
        "wrote it for a lot of reasons. I wrote it because it seemed to me that you "
        "couldn't real'",
    ),
    ("97facf74-c6fc-11f0-b555-cf872a2d78a0", "SPEAKER_06"): (
        "Bird Pinkerton",
        "guest",
        "Byrd Pinkerton",
        "host",
        "Unexplainable 'The bodies buried in the bog': says 'The word bog has never had "
        "a particularly charming connotation to me.'",
    ),
}


def _stored_by_voice(meta: Path, rel: str) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """``{voice: (its named label or None, its segment role)}`` from the episode's segments."""
    payload = _load(meta.parent.parent / rel.replace(".txt", ".segments.json"))
    rows = payload if isinstance(payload, list) else (payload or {}).get("segments")
    out: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for r in rows or []:
        if isinstance(r, dict) and r.get("speaker") and str(r["speaker"]) not in out:
            out[str(r["speaker"])] = (r.get("speaker_label") or None, r.get("speaker_role") or None)
    return out


class _Episode(_VoiceEpisode):
    """m0025's voice-by-voice rewrite, decided by ``NAMING_ARC_VOICES`` alone."""

    def __init__(self, meta: Path, root: Path, voices: Optional[Dict] = None) -> None:
        super().__init__(meta, root)
        self._map = NAMING_ARC_VOICES if voices is None else voices

    def decide(self) -> bool:
        meta = _load(self.meta)
        if not isinstance(meta, dict):
            return False
        episode_id = str((meta.get("episode") or {}).get("episode_id") or "")
        mine = {v: e for (eid, v), e in self._map.items() if eid == episode_id}
        if not mine:
            return False
        rel = _run_relative(
            (meta.get("content") or {}).get("transcript_file_path"), self.meta.parent.parent
        )
        if not rel.endswith(".txt"):
            self.refused = "no transcript"
            return True
        self._load_seg_voices(rel)
        stored = _stored_by_voice(self.meta, rel)
        for voice, (old_name, old_role, new_name, new_role, _why) in sorted(mine.items()):
            name, role = stored.get(voice, (None, None))
            name = None if not name or _DIARIZATION_LABEL.match(name) else name
            if (name, role if new_name else None) == (new_name, new_role if new_name else None):
                continue  # already done
            if (name, role) != (old_name, old_role):
                self.refused = (
                    f"{voice}: stored {name!r}/{role!r}, the replay read {old_name!r}/{old_role!r}"
                )
                return True
            self.voice_names[voice] = new_name
            if new_name and new_role:
                self.voice_roles[voice] = new_role
        return bool(self.voice_names)


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class NamingArcVoicesRepairedMigration(Migration):
    """The voices #2304's naming ladder names differently, applied to the served records."""

    id = MIGRATION_ID
    to_version = "2.7.23"
    #: 0030a carries this list again through the fixed rewrite, with its own decision and files.
    _episode_class: type = _Episode
    _receipts_file = RECEIPTS_FILE
    _backup_tag = BACKUP_TAG
    description = (
        "the 83 voices the #2304 naming ladder names differently from the code that wrote them "
        "(read in the 2026-10-12 replay), rewritten voice by voice across metadata, segments, "
        "transcripts, GI, KG, bridge, diagnostics, turns and context"
    )

    def _scan(self, root: Path) -> Tuple[List[_Episode], List[_Episode], Dict[str, int]]:
        write: List[_Episode] = []
        refused: List[_Episode] = []
        totals: Dict[str, int] = {}
        for meta in select_served_artifacts(root, ".metadata.json")[0]:
            ep = self._episode_class(meta, root)
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
        return (
            f"naming arc voices plan: {len(write)} episode(s), {len(refused)} refused; "
            + ", ".join(f"{k}={v}" for k, v in sorted(totals.items()))
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
        return True, "every listed voice carries the name the naming ladder gives it"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Rewrite every episode with a listed voice still on its old name, with backups."""
        root = ctx.corpus_root
        write, refused, totals = self._scan(root)
        receipts: List[dict] = []
        if not ctx.dry_run:
            for ep in write:
                for path, payload in sorted(ep.json_files.items()):
                    receipts.append(write_with_backup(root, self._backup_tag, path, payload))
                for path, text in sorted(ep.text_files.items()):
                    receipts.append(_write_text_with_backup(root, path, text, self._backup_tag))
            append_receipts(root, self._receipts_file, {"migration": self.id}, receipts)
        verb = "would rewrite" if ctx.dry_run else "rewrote"
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=f"{verb} {len(write)} episode(s); "
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
    write, refused, _totals = NamingArcVoicesRepairedMigration()._scan(Path(root))
    for ep in write + refused:
        yield {
            "meta": str(ep.meta.relative_to(root)),
            "voices": ep.voice_names,
            "refused": ep.refused,
            "files": sorted(
                str(p.relative_to(root)) for p in list(ep.json_files) + list(ep.text_files)
            ),
        }
