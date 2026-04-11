<script setup lang="ts">
/**
 * Short prerequisites for vector search (RFC-062 / PRD-021 parity with CLI).
 * Uses semantic tokens only (UXS-001).
 */
</script>

<template>
  <aside
    class="rounded border border-border border-l-4 border-l-primary bg-elevated px-2.5 py-2 text-xs leading-relaxed text-muted"
    aria-label="Semantic search tips"
  >
    <p class="font-medium text-surface-foreground">
      How semantic search works
    </p>
    <ul class="mt-1.5 list-disc space-y-1.5 pl-4 marker:text-muted">
      <li>
        The server reads the FAISS index under
        <code class="whitespace-nowrap rounded bg-canvas px-1 py-0.5 font-mono text-[11px] text-canvas-foreground">
          &lt;corpus&gt;/search/
        </code>
        — build it with
        <code class="rounded bg-canvas px-1 py-0.5 font-mono text-[11px] text-canvas-foreground">
          podcast index
        </code>
        (or your pipeline’s vector step). Dashboard → Vector index shows whether it
        loaded.
      </li>
      <li>
        Queries are embedded with the same model as the index; the server needs that
        model available locally (same idea as
        <code class="rounded bg-canvas px-1 py-0.5 font-mono text-[11px] text-canvas-foreground">
          podcast search
        </code>
        without network downloads).
      </li>
      <li>
        Use filters to narrow doc types, feed, date, speaker, or grounded insights — same
        metadata the indexer stores.
      </li>
      <li>
        Result cards use <strong>G</strong> to focus a graph node when the hit maps to your loaded
        <code class="rounded bg-canvas px-1 py-0.5 font-mono text-[11px] text-canvas-foreground">
          .gi.json
        </code>
        /
        <code class="rounded bg-canvas px-1 py-0.5 font-mono text-[11px] text-canvas-foreground">
          .kg.json
        </code>
        . <strong>L</strong> opens the episode in Library when the vector index stores
        <code class="rounded bg-canvas px-1 py-0.5 font-mono text-[11px] text-canvas-foreground">
          source_metadata_relative_path
        </code>
        (run a fresh <code class="rounded bg-canvas px-1 py-0.5 font-mono text-[11px] text-canvas-foreground">podcast index</code>).
      </li>
    </ul>
  </aside>
</template>
