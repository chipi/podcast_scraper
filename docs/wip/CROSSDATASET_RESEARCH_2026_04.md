# Cross-Dataset Research — Summarization Benchmarks (April 2026)

Research into publicly available datasets for cross-domain summarization evaluation.
Our pipeline produces **300-600 word abstractive summaries from 5-50k character inputs**.
Datasets ranked by fit for that profile.

---

## 🎯 Top picks for Phase 2.2 — podcast-domain (best domain match)

### 1. `potsawee/podcast_summary_assessment` (TREC 2020 surviving artifact)

- **Source:** [huggingface.co/datasets/potsawee/podcast_summary_assessment](https://huggingface.co/datasets/potsawee/podcast_summary_assessment)
- **Size:** 3,580 transcript-summary pairs
- **Content:** actual podcast transcripts + summaries from the TREC 2020 podcast
  summarization track, with human quality assessment scores
- **License:** CC-BY-4.0 (freely downloadable)
- **Input length:** TBD — need to inspect; likely 10-100k chars (full podcast episodes)
- **Reference summaries:** yes — system-generated summaries that were human-judged for
  quality. May not be "gold" in the traditional sense (they're model outputs, not
  human-written), but the human quality scores let us filter for the best ones.
- **Status:** needs exploration. Key questions:
  - Are the "summaries" human-written or model-generated?
  - What quality scores are available and can we use top-scored ones as references?
  - Do transcripts come with the dataset or need separate download?
- **Why interesting:** closest domain match possible. If transcripts + usable reference
  summaries exist, this is the Phase 2.2 dataset.

### 2. RHAPSODY (2025, YouTube podcasts)

- **Paper:** [arxiv.org/html/2505.19429v2](https://arxiv.org/html/2505.19429v2)
- **Size:** 13K YouTube podcast episodes
- **Content:** YouTube podcast highlights + transcripts
- **License:** TBD — need to check paper
- **Input length:** TBD — YouTube podcast episodes vary (5 min to 3+ hrs)
- **Reference summaries:** focuses on highlight detection, not generic summarization.
  May need silver generation if no generic summaries exist.
- **Status:** needs exploration. Key questions:
  - Does it have generic episode summaries or only highlight timestamps?
  - Is the data accessible (GitHub repo, HF dataset)?
  - What's the transcript quality (ASR or manual)?
- **Why interesting:** 13K episodes is massive scale. If it has usable summaries,
  it's the largest podcast summarization dataset currently available.

---

## 📊 Best-fit non-podcast datasets (broader domain validation)

Ranked by match to our pipeline's input/output profile.

| Rank | Dataset | HF URL | Input | Gold ref | Size | License | Fit |
|------|---------|--------|-------|----------|------|---------|-----|
| 1 | **GovReport** | `ccdv/govreport-summarization` | 30-100k chars | 600-1200 words (analyst-written) | 19.5k | Apache 2.0 | **Best length match** — professional abstractive summaries, government reports |
| 2 | **BookSum** | `kmfoda/booksum` | 5-40k chars/chapter | 300-600 words (study guides) | 12.6k | BSD-3 | **Exact output-length match** — narrative domain (fiction + nonfiction) |
| 3 | **SQuALITY** | `GEM/squality` | 15-30k chars | 250-400 words, **4 refs/story** | 625 | CC BY 4.0 | Multi-reference gold (rare, reduces noise). Small but very high quality |
| 4 | **ArXiv** | `ccdv/arxiv-summarization` | 20-80k chars | 150-300 words (abstracts) | 215k | CC0 | Massive scale. Refs slightly short but volume compensates |
| 5 | **BillSum** | `FiscalNote/billsum` | 5-50k chars | 200-400 words | 23.5k | CC0 | Legislative domain, good input range |
| 6 | **CaseSumm** | `ChicagoHAI/CaseSumm` | 20-200k chars | 300-800 words (SCOTUS syllabuses) | 25.6k | CC BY-NC 4.0 | Very long inputs + matching summary length. NC license. |
| 7 | **Multi-LexSum** | `allenai/multi_lexsum` | 100-500 pages/case | 3 granularities (25/130/500+ words) | 9.3k | ODbL | Long summaries (~500w) at fine granularity. Multi-doc. |
| 8 | **Multi-News** | `alexfabbri/multi_news` | 3-15k chars | 250-400 words | 56k | Non-commercial | Shorter inputs, but large scale. Multi-doc. |

---

## ❌ Investigated and dropped

| Dataset | Source | Why dropped |
|---------|--------|-------------|
| **MeetingBank** | `huuuyeah/MeetingBank` | Per-agenda-item, not full meetings. Gold summaries median 60 words — structural mismatch with our 300-600w output. |
| **DialogSum** | `knkarthick/dialogsum` | Too short (300-2k chars). Below our pipeline's 5k+ char sweet spot. |
| **AMI Corpus** | `edinburghcstr/ami` | Only 137 meetings. Overlaps QMSum's domain (both academic meetings). |
| **MediaSum** | `ccdv/mediasum` | Gold refs 1-2 sentences — too short for ROUGE against our output. |
| **Spotify Podcast Dataset** | Was at podcastsdataset.byspotify.com | **Offline since Dec 2023.** No official mirrors. Access requests no longer accepted. See `podcast_summary_assessment` above for surviving artifact. |
| **SummScreen** | TV/movie scripts | Narrower value than QMSum for spoken-word validation. |
| **XSum / CNN-DM** | News articles | Written text, not spoken. Too short for our pipeline. |
| **ArXiv-summarization** (as primary) | Academic papers | Ref summaries (abstracts) are 150-300 words — slightly below our target. Useful as secondary. |
| **SAMSum** | Chat dialogues | Very short (avg 100 words input). Not a fit. |

---

## 📋 Recommendation for Phase 2.2

**First priority:** explore `potsawee/podcast_summary_assessment`. If it has usable
transcripts + reference summaries of reasonable length, it's the Phase 2.2 dataset
(exact domain match, free, 3.5k examples).

**If podcast_summary_assessment doesn't work out:** pivot to **GovReport** (best
non-podcast fit) or **BookSum** (best output-length match). Both are freely available
on HF with proper abstractive gold references.

**Scale play if needed later:** ArXiv (215k examples, CC0) for massive-scale validation,
accepting that reference summaries (abstracts) are shorter than our target.

---

## Open actions

- [ ] Explore `potsawee/podcast_summary_assessment` — inspect structure, transcript
  availability, summary quality scores, reference usability
- [ ] Explore RHAPSODY — check if generic summaries exist, access method, transcript quality
- [ ] After QMSum Phase 2.1 results: decide Phase 2.2 dataset
