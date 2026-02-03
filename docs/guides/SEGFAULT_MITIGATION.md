# Segfault Mitigation Guide

This guide provides strategies for diagnosing and mitigating segmentation faults that occur during pipeline execution, particularly at process shutdown.

## Common Causes

Segfaults at the end of successful pipeline runs are typically caused by:

1. **PyTorch MPS (Metal Performance Shaders) teardown issues**
   - MPS backend cleanup can trigger segfaults during interpreter shutdown
   - Especially common with Transformers models on Apple Silicon

2. **Native extension cleanup order**
   - PyTorch, Transformers, spaCy, and Thinc all have native extensions
   - Destructor ordering during shutdown can cause double-free or use-after-free

3. **Threading + native library interactions**
   - Worker threads holding references to native objects
   - Cleanup happening in wrong order across threads

## Diagnostic Tools

### Enable Faulthandler (Automatic)

Faulthandler is automatically enabled when running the CLI. It provides native backtraces when crashes occur.

To enable manually:

```bash
export PYTHONFAULTHANDLER=1
python -m podcast_scraper.cli ...
```

Or in code:

```python
import faulthandler
faulthandler.enable(all_threads=True)
```

### Check Crash Dump

If a crash occurs, check for `crash_dump_<pid>.log` in the current directory for a backtrace.

## Mitigation Strategies (Try in Order)

### Option 0: Enable MPS Exclusive Mode (Prevent Memory Contention)

If both Whisper and summarization use MPS, enable exclusive mode to serialize GPU work and prevent memory contention:

```yaml
# config.yaml
mps_exclusive: true  # Default: true
```

Or via environment variable:

```bash
export MPS_EXCLUSIVE=1
```

This ensures transcription completes before summarization starts on MPS, preventing both models from competing for GPU memory. I/O operations (downloads, parsing) remain parallel. This is enabled by default and can help prevent crashes from memory pressure.

**When to use**: Always enabled by default. Disable (`mps_exclusive: false`) only if you have sufficient GPU memory and want maximum throughput with concurrent GPU operations.

### Option 1: Disable MPS for Summarization (Most Common Fix)

Keep Whisper on MPS if stable, but move Transformers summarization to CPU:

```yaml
# config.yaml
summary_device: cpu
```

If the segfault disappears, it's almost certainly MPS + Transformers teardown.

### Option 2: Run Everything on CPU (Sanity Check)

This is the "is it MPS?" test:

```yaml
# config.yaml
whisper_device: cpu
summary_device: cpu
```

If CPU run exits cleanly: MPS is involved.

### Option 3: Force Safer Threading Settings

Set environment variables before running:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
python -m podcast_scraper.cli ...
```

This often helps when native libs + threads die at exit.

### Option 4: Don't Cleanup Models Explicitly

Sometimes explicit `del model; gc.collect() + torch.mps.empty_cache()` at the very end triggers unstable finalizers.

**Try:**

- Cleanup after each episode (or after each stage) rather than at process shutdown
- Skip cleanup entirely and let the process exit (counterintuitive, but avoids double-free / destructor ordering bugs)

### Option 5: Isolate Summarization into a Subprocess

If stability matters more than performance:

- Run the summarization step in a separate Python process
- Return only the summary text to the main process

If it segfaults, it won't take down the main run (and you can retry).

## Getting Actionable Crash Information

### Enable Faulthandler with File Output

```bash
PYTHONFAULTHANDLER=1 python -m podcast_scraper.cli ... 2>&1 | tee run.log
```

Check `crash_dump_<pid>.log` for native backtrace.

### Check Last Log Lines

Paste the very last few log lines before the segfault (right after cleanup starts) to identify the most likely culprit:

- **Whisper teardown**: Look for Whisper model cleanup logs
- **spaCy/thinc**: Look for NER model cleanup logs
- **Transformers/MPS**: Look for summary model cleanup logs

## Environment Variables Summary

```bash
# Threading limits (reduces teardown weirdness)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Faulthandler (crash diagnostics)
export PYTHONFAULTHANDLER=1

# Run pipeline
python -m podcast_scraper.cli ...
```

## Related Documentation

- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Development Guide](DEVELOPMENT_GUIDE.md)
