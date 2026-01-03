# Whisper Progress Bar Analysis & Proposal

## Current Situation

**Problem**: Whisper's internal progress bar (using `tqdm`) creates 30-40 new lines instead of
updating in place, even with `verbose=False`.

**Current Solution**: We suppress Whisper's progress output by redirecting `stdout`/`stderr` to a null stream.

**Current Progress Bar**: Our own progress bar shows "Transcribing: elapsed" but provides no actual
progress information (just elapsed time).

## What is Whisper's Internal Progress Bar?

1. **Technology**: Whisper uses `tqdm` library internally to show transcription progress
2. **Information**: Shows percentage complete, speed (tokens/sec), and ETA
3. **Control**: Controlled by `verbose` parameter:
   - `verbose=True`: Shows detailed progress with text being decoded
   - `verbose=False`: Should show minimal progress, but still emits multiple lines (bug)
   - `verbose=None`: Should suppress all output, but doesn't always work
4. **No Callback API**: Whisper doesn't expose a progress callback - it uses tqdm directly

## Analysis: Should We Use It?

### Option A: Keep Suppressing (Current Approach)

**Pros:**

- ✅ Clean output, no clutter
- ✅ Simple implementation
- ✅ No conflicts with our progress system

**Cons:**

- ❌ No real-time progress during transcription
- ❌ Users can't see how long transcription will take
- ❌ Poor UX for long transcriptions (can take minutes)

### Option B: Use Whisper's Progress Bar Directly

**Pros:**

- ✅ Real-time progress with percentage, speed, ETA
- ✅ Useful information for users

**Cons:**

- ❌ Multiple lines issue (the bug we're trying to fix)
- ❌ Less control over formatting
- ❌ Conflicts with our own progress bars
- ❌ Can't integrate with our progress system

### Option C: Intercept Whisper's Progress (RECOMMENDED)

**Approach**: Override `tqdm` globally during transcription to capture Whisper's progress updates and
feed them to our own progress bar.

**Pros:**

- ✅ Real-time progress information
- ✅ Single progress bar (no multiple lines)
- ✅ Integrates with our existing progress system
- ✅ Consistent UI with rest of application
- ✅ Can show episode-specific information

**Cons:**

- ⚠️ More complex implementation
- ⚠️ Requires careful tqdm monkey-patching

## Recommendation: Option C - Intercept Whisper's Progress

### Implementation Strategy

1. **Create a custom tqdm wrapper** that captures Whisper's progress updates
2. **Temporarily override `tqdm.tqdm`** during transcription
3. **Update our progress bar** with Whisper's progress information
4. **Restore original tqdm** after transcription completes

### Example Implementation

````python
class WhisperProgressInterceptor:
    """Intercepts Whisper's tqdm calls and forwards to our progress bar."""

    def __init__(self, our_progress_bar):
        self.our_bar = our_progress_bar
        self.original_tqdm = None

    def __enter__(self):
        import tqdm
        self.original_tqdm = tqdm.tqdm

        class InterceptedTqdm(tqdm.tqdm):
            def __init__(self, *args, **kwargs):

                # Suppress tqdm's own output

                kwargs['file'] = open(os.devnull, 'w')
                kwargs['disable'] = True  # Disable tqdm's display
                super().__init__(*args, **kwargs)

```python

            def update(self, n=1):
                super().update(n)

```
                # Forward to our progress bar

```

                if self.our_bar:
                    self.our_bar.update(n)

```

```python

    def __exit__(self, *args):
        import tqdm
        tqdm.tqdm = self.original_tqdm

```
2. **Clean Output**: Single progress bar, no multiple lines
3. **Consistent UX**: Matches our other progress bars
4. **Episode Context**: Can show which episode is being transcribed

### Alternative: Simpler Approach

If intercepting is too complex, we could:

- Set `verbose=True` and configure tqdm globally to prevent multiple lines
- Use tqdm's `disable` parameter conditionally
- Accept Whisper's progress bar but configure it better

## Decision

**Recommendation**: Implement Option C (intercept Whisper's progress) because:

1. Provides real progress information users need
2. Maintains clean output (no multiple lines)
3. Integrates well with existing progress system
4. Better UX for long transcriptions

**Fallback**: If Option C proves too complex or fragile, keep current suppression (Option A) but improve our placeholder progress bar to show more useful information (e.g., "Transcribing episode 5/10...").

````
