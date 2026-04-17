# Home AI Hardware — Planning Note

Scratch thinking on what a self-hosted indie AI setup would look like to run the workloads
we've been evaluating (Ollama LLMs + HF transformers ML models + potential training/LoRA)
with better performance than your current M4 MBP 48GB.

**This is a working draft** — things to think through before spending money, not a shopping list.

---

## What our current workloads actually need

Based on the v2 eval work, the workloads break into four size tiers:

| Tier | Workload example | Working set needed | MBP M4 48GB today |
|------|-------------------|:------------------:|:-----------------:|
| Small | phi3:mini, llama3.2:3b, BART, LongT5-base | <8GB | Fast, trivial |
| Mid | qwen3.5:9b, SummLlama3.2-3B, LongT5-xl, mistral:7b | 8-20GB | Workable, 20-60s/ep |
| Large | qwen3.5:27b, qwen3.5:35b, mistral-small:22b | 20-40GB | Borderline, 100+s/ep |
| XL | 70B-class (llama3.3:70b Q4) | 40-70GB | Cannot fit |
| Training | LoRA on 3-9B models | 30-60GB + | Cannot realistically train |

You already cover small + mid + most of large. The gap is XL inference and any training.

**Concrete question:** are those gaps valuable enough to buy hardware for?

- XL inference quality gains: ~2-5 pp ROUGE-L vs top mid-tier (based on v1 numbers, flattening under v2). Marginal.
- Training/LoRA: potentially transformative for small models (SummLlama showed what DPO does to a 3B). **This is the real reason to upgrade.**

---

## Hardware options (Feb-Apr 2026 landscape)

### Option A: Bigger Mac Studio / Mac Pro

- **M3 Ultra / M4 Ultra Mac Studio**, 64/128/192/256GB unified memory
- Unified memory = what MPS sees. Much more than current 48GB.
- Same tooling you're already using (MLX, MPS-compatible transformers)
- Same software gaps (no CUDA → no bitsandbytes for QLoRA)

**Cost/fit:**

- M3 Ultra Studio 64GB: ~$3,999 baseline
- M3 Ultra Studio 128GB: ~$4,800
- M3 Ultra Studio 192GB: ~$5,600
- M3 Ultra Studio 256GB: ~$6,400

**Pros:**

- Drop-in replacement for your current Apple workflow
- Silent, low-power, no fans roaring
- Can coexist with other Apple kit (file sharing, continuity)
- MLX ecosystem growing fast; Ollama already has MLX backend preview

**Cons:**

- No CUDA — QLoRA via bitsandbytes still not available
- MLX training is newer, fewer curated recipes than PyTorch+CUDA
- Can't run the NVIDIA-native training frameworks (torchtune, DeepSpeed, etc.)
- Premium pricing per GB of unified memory

**Verdict:** most seamless path but caps your training options.

### Option B: NVIDIA workstation / desktop

- RTX 4090 (24GB), RTX 5090 (32GB as of 2025), RTX 6000 Ada (48GB), or datacenter refurb
- Mix-and-match with 64-128GB DDR5 system RAM
- CUDA ecosystem: bitsandbytes/QLoRA, DeepSpeed, FlashAttention, unsloth, axolotl, etc.

**Cost/fit:**

- DIY desktop + RTX 4090 + 64GB DDR5: ~$3,500-4,500
- DIY desktop + RTX 5090 + 64GB DDR5: ~$4,500-6,000
- Used RTX 3090 (24GB) build: ~$2,000-2,800 (lower sustained quality)
- RTX 6000 Ada (48GB): ~$7,000+ for the GPU alone
- Dual-4090 workstation: ~$8,000+ (supports 70B Q4 inference + 13B LoRA)

**Pros:**

- **Full training ecosystem.** QLoRA, unsloth, axolotl all work. This is where SummLlama was trained.
- Can run 70B models at Q4 (48GB VRAM) or Q8 (dual-24GB)
- Much larger model zoo works out of the box
- Faster training (3-10x vs Apple Silicon for equivalent size)
- Reusable for gaming, 3D, other CUDA workloads

**Cons:**

- Loud, hot, power-hungry (350-600W under load)
- Separate box from your work laptop (SSH into it, set up remote access)
- Needs Linux for best ecosystem support — or Windows+WSL2
- VRAM is strict cap; can't spill to system RAM cleanly
- Per-GB of VRAM is expensive vs Apple unified

**Verdict:** the right answer if you want to do training seriously.

### Option C: Cloud-when-needed (rental model)

- Use current M4 for everything you already can (inference, small-scale tests)
- Rent a GPU by the hour for training / 70B+ inference:
  - RunPod RTX 4090: ~$0.40-0.60/hr
  - RunPod A100 80GB: ~$1.40/hr
  - Lambda H100: ~$2.50/hr

**Pros:**

- $0 capex; pay only when you use
- Always-latest GPUs available
- No home cooling/power infrastructure
- Fire-and-forget: train overnight, download weights, release instance

**Cons:**

- Network-bound: upload datasets, download checkpoints (slow for 50GB+ models)
- Latency to instance (SSH, Jupyter, VS Code remote)
- Data-residency concerns if running on someone else's cloud
- You do NOT own the tools when the instance stops; always setting up
- Long-running agents / dev iteration feels painful through SSH

**Verdict:** complements Mac well. Fine for occasional training. Not a daily-driver solution.

#### Cloud GPU providers — links + comparison

- **RunPod** — [runpod.io](https://runpod.io) — pod-based, very indie-friendly, best docs
  for PyTorch/HF workflows. Also has "serverless GPU" for short jobs.
- **Lambda Cloud** — [lambdalabs.com/service/gpu-cloud](https://lambdalabs.com/service/gpu-cloud)
  — simpler / more business-y, H100 and A100 availability good, fewer frills than RunPod.
- **Vast.ai** — [vast.ai](https://vast.ai) — marketplace (random people's idle GPUs).
  Cheapest (~30-50% below RunPod) but reliability varies and security posture depends on
  the host. Fine for non-sensitive experiments.
- **Modal** — [modal.com](https://modal.com) — different paradigm: you write a Python
  function with a decorator (`@app.function(gpu="A100")`), Modal spins up a container on
  each call. Best when you want "GPU as a Python library" rather than a persistent pod.
- **Paperspace Gradient** — [paperspace.com/gradient](https://paperspace.com/gradient) —
  Jupyter-focused, fine for notebook tinkering, less suited for long training jobs.

**Rough pricing (2026 ballpark):**

| GPU | VRAM | RunPod | Lambda | When to pick |
|-----|------|--------|--------|--------------|
| RTX 4090 | 24GB | ~$0.40/hr | — | 3-9B LoRA, single-GPU inference |
| A100 40GB | 40GB | ~$0.80/hr | ~$1.10/hr | 13B models, faster LoRA |
| A100 80GB | 80GB | ~$1.40/hr | ~$1.80/hr | 30B inference, full fine-tuning mid-size |
| H100 80GB | 80GB | ~$2.50-4/hr | ~$2.50/hr | Top speed, fast training |

#### RunPod setup flow (what it actually looks like)

About 5-10 min from "I want a GPU" to "training is running." Step-by-step:

1. **Sign up** at runpod.io, add card, deposit $10-20 to start (they charge per-minute
   against the deposit)
2. **Deploy a pod** — click "Deploy," pick:
   - Hardware: e.g. "RTX 4090 (24GB)" or "A100 80GB"
   - Template: "RunPod PyTorch 2.4" (Python + CUDA + PyTorch pre-installed)
   - Disk: ~50GB persistent volume if you want to save cache between sessions; otherwise
     ephemeral
3. **Wait ~60 seconds** for it to provision. You get:
   - A Jupyter URL (click to open notebook in browser)
   - SSH key + host (`ssh root@...`)
   - Direct VS Code Remote SSH target (`code --remote ssh-remote+<host>`)
4. **Get your code onto the pod**, 3 options:
   - `git clone` the repo inside the pod (cleanest for public code)
   - `scp -r` / `rsync` from your MBP (works but slow for big repos)
   - Mount a persistent volume (`/workspace`) and keep the repo there between sessions
5. **Run your workload** — for our case: `pip install -r requirements.txt`, then
   `python scripts/eval/run_summllama_v2.py ...` or a LoRA training command. Exactly the
   same command you'd run locally.
6. **Download results** — `scp -r` the predictions file / adapter weights back to your MBP
7. **Stop the pod** (button in UI or `runpodctl stop`). Billing ends. If you keep a volume,
   state survives for next session.

#### Practical gotchas

- **First pod takes longer** — signup, template pull, first download of the HF model
  (e.g. SummLlama ~6GB) each take a few minutes. After that, relaunching with the same
  volume is fast.
- **Use persistent volumes** for anything you don't want to re-download every session
  (HF model cache, datasets, venv). Without a volume, you re-`pip install` and
  re-download every time.
- **Upload speed is your home bandwidth** — our materialized datasets are small (~10MB),
  but LoRA training checkpoints are 500MB-2GB. Download from pod is fast; upload from
  your home line is typically 20-50 Mbps.
- **Security** — RunPod/Lambda: OK for normal work, standard Linux VM. Vast.ai: random
  host runs your code, don't put secrets there.
- **Cost discipline** — set a "max spend" alarm in RunPod settings. The failure mode is
  "forgot to stop the pod, 72hrs × $1.40 = $100 surprise." Cheap guard: turn off the pod
  the moment you're done.

#### When this actually makes sense for our workload

Two realistic cases:

1. **LoRA training** (3-9B model, 4-8 hrs on A100 = $5-10 per run). Run maybe 3-5 times
   during a tuning campaign. Total: ~$50.
2. **70B inference smoke tests** (A100 80GB, 1hr to run a 5-episode eval = $1.40). Once-off
   kind of use.

The "friction" is mainly: remembering to start the pod at the start of a session and stop
it at the end. Everything in between feels exactly like working on a remote Linux box — VS
Code Remote SSH makes it seamless. A training weekend might look like 2-3 sessions of
~4hrs each, total spend $20-40, total wall-clock 8-12hrs.

**Scripts worth writing when this becomes real** (not yet needed):

- `scripts/cloud/bootstrap_runpod.sh` — clone repo, install venv, cache HF models
  (one-time per pod)
- `scripts/cloud/run_and_download.sh` — drive a training/eval, download artifacts, stop pod

Each is ~50 lines; defer until the first real cloud run.

---

## What actually makes sense for your situation

Given what we've built (podcast_scraper + v2 eval + WIP LoRA plan):

**If production serving is the near-term goal:**

- **M4 MBP 48GB stays the primary dev machine** (fast, integrated workflow, covers mid-tier inference)
- Add a **Mac mini M4 Pro 64GB** (~$2,000) as a "home server" for overnight batch jobs
  - Runs Ollama + autoresearch ratchet loops while you work on the laptop
  - Can host silvers corpus, serve inference to apps on your LAN
  - Coexists with your dev Mac seamlessly (same OS, same tools)
- Use **cloud GPU rental** for any occasional training

**If serious training / LoRA / research is the near-term goal:**

- Keep MBP for dev + inference
- **Build or buy an NVIDIA desktop with RTX 4090 or 5090 + 64GB RAM** (~$4,000-6,000)
- Put it in a closet with a wired ethernet; SSH/VS Code remote in
- This is where DPO, LoRA, fine-tuning, 70B inference all become first-class
- Run as a Linux server (Ubuntu 24.04 LTS); your M4 stays the friendly dev surface

**Hybrid (what I'd actually suggest):**

- Mac mini M4 Pro 64GB home server: $2,000, silent, runs Ollama + batch eval
- When/if training becomes a real need: rent RunPod A100 for a weekend, see if it justifies owning a 4090 box
- Own the NVIDIA box only after you've rented 3-4 times and know you want it weekly

---

## What "run an actual service next to it" looks like

You mentioned running a service box separately. If the ML box is the "AI lab" and you want a separate "production service" machine:

**Production serving:**

- **Mac mini M4 Pro** (base or 16GB) as a production box is plausible — it runs Ollama + your podcast_scraper CLI fine. ~$600-1,000.
- Or **Raspberry Pi 5 8GB** if the service just orchestrates external APIs (doesn't run models). ~$80.
- Or a **small NUC / Framework Desktop** on Linux if you want x86 Docker ecosystem. ~$400-800.

**Networking:**

- 1GbE minimum between boxes (transferring model weights, streaming audio for transcription)
- If you upgrade the ML box to have 10GbE, wire both boxes that way for fast checkpoint moves

**Architecture**:

```text
┌─────────────────────┐       ┌────────────────────────┐
│ Dev MBP M4 48GB     │       │  ML Lab Box            │
│ (laptop, portable)  │◄─────►│  (Mac Studio OR        │
│ autoresearch work   │ LAN   │   NVIDIA desktop)      │
│ code, docs, eval    │       │  Ollama daemon         │
└─────────────────────┘       │  HF model cache        │
         │                    │  LoRA training         │
         │ API calls          └────────────────────────┘
         ▼
┌─────────────────────┐
│ Production box      │
│ (Mac mini M4 Pro    │
│  or NUC)            │
│ podcast_scraper svc │
└─────────────────────┘
```

---

## Cost bands to think in

| Budget | What you can do |
|--------|-----------------|
| **$0 extra** | Current M4 48GB + cloud GPU rental for training. Covers 90% of what we've been doing. |
| **$2k** | Add Mac mini M4 Pro 64GB as home server. 7B-20B inference always-on, batch evals overnight. |
| **$5k** | Mac Studio M3 Ultra 128GB OR NVIDIA RTX 5090 workstation. 70B inference, meaningful LoRA training. |
| **$10k+** | Dual-4090 workstation or Mac Studio 192GB. Research-grade. Overkill unless you're doing real ML research. |

My honest prior: **$2k (Mac mini home server)** is the right first step if you want to free the laptop up for work. **$5k (either Mac Studio 128GB or 4090 box)** is the right step if you want to do LoRA training seriously. Going higher than $5k requires a specific use case justifying it.

---

## Software ecosystem to track

Whichever path you take, these are the names to follow:

- **Ollama** — your current LLM serving path. Improving fast, MLX backend coming.
- **MLX-LM** — Apple's LLM training + inference on MPS. Best-in-class for Apple Silicon.
- **unsloth** — QLoRA training library, CUDA only but 2-5x faster than vanilla HF Trainer.
- **axolotl** — YAML-driven fine-tuning, CUDA.
- **torchtune** — Meta's PyTorch-native fine-tuning, works on MPS and CUDA.
- **vLLM / SGLang** — production LLM serving (throughput focus), CUDA primary.
- **llama.cpp** — CPU + Metal + CUDA, what Ollama uses under the hood.
- **RunPod / Lambda / Vast.ai** — GPU rental providers for occasional jobs.

---

## Concrete next-step question (for when you have a free evening)

Answer this before spending any money:

1. In the next 6 months, will you want to do **real LoRA training** (not just inference)? If yes → consider NVIDIA. If no → stay Apple.
2. Do you want the laptop to **stay responsive during long eval runs**? If yes → any home server helps.
3. Is your workload growing past the M4 48GB comfortably, or are you still inside it? Current answer: inside it. So the urgent need is batch-job offloading, not raw horsepower.

Given those: the **most leveraged first upgrade is a Mac mini M4 Pro 64GB as a home server**, not a new compute monster. Buy it when you catch yourself wanting to run something overnight that's blocking the laptop.

---

## Not in scope of this note

- Storage (NAS / SSD arrays) — separate concern, depends on silver corpus size
- Power / cooling — NVIDIA box needs a dedicated ~800W circuit and air
- Noise — NVIDIA towers are loud; Mac Studio / mini are silent
- UPS — $200-400 addition for any always-on box

Happy to revisit once you've mulled on this.
