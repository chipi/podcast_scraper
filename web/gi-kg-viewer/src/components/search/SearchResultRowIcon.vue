<script setup lang="ts">
/**
 * Leading icon on every search-result row (2026-07-22 UX cleanup). Two
 * shapes:
 *   - Letter avatar in a colored badge (matches the letter chrome the
 *     graph node detail panel uses, so entities are recognizable across
 *     surfaces). Used for Insight / Quote / Topic / Person / Summary and
 *     the transcript-cluster episode row.
 *   - Small episode artwork thumbnail when the hit resolves to an
 *     episode with a locally-cached cover (mirrors the digest / library
 *     row treatment).
 *
 * Replaces the prior text labels at the top of each result card
 * (``docType`` chip + tier badge). Their information now lives in the
 * icon's ``title`` tooltip.
 */
import { computed } from 'vue'
import PodcastCover from '../shared/PodcastCover.vue'
import { useShellStore } from '../../stores/shell'

type IconKind = 'letter' | 'image'

interface LetterIcon {
  kind: 'letter'
  letter: string
  /** Tailwind bg-* / text-* classes for the badge. */
  classes: string
  title: string
}
interface ImageIcon {
  kind: 'image'
  title: string
  episodeImageLocalRelpath: string | null
  feedImageLocalRelpath: string | null
  feedImageUrl: string | null
  episodeImageUrl: string | null
  alt: string
}
type IconMeta = LetterIcon | ImageIcon

const shell = useShellStore()

const props = defineProps<{
  /** ``doc_type`` from the hit's metadata (or a synthetic value for
   *  the transcript-cluster row, e.g. ``episode``). */
  docType: string
  /** ``source_tier`` from the hit — folded into the tooltip. */
  sourceTier?: string | null
  /** Extra tooltip context, e.g. speaker name for a Quote. */
  subtitle?: string | null
  /** When the row IS an episode (transcript cluster, kg_episode etc.),
   *  pass any cached artwork so we render a thumbnail instead of a
   *  letter. */
  episodeImageLocalRelpath?: string | null
  feedImageLocalRelpath?: string | null
  episodeImageUrl?: string | null
  feedImageUrl?: string | null
}>()

const TIER_LABEL: Record<string, string> = {
  insight: 'Insight',
  segment: 'Transcript',
  aux: 'Reference',
}

function tierBit(): string {
  const t = props.sourceTier ?? ''
  const label = TIER_LABEL[t]
  return label ? `${label} · ` : ''
}

const meta = computed<IconMeta>((): IconMeta => {
  const dt = props.docType || 'unknown'
  const subtitleBit = props.subtitle ? ` — ${props.subtitle}` : ''
  const hasArtwork = Boolean(
    props.episodeImageLocalRelpath ??
      props.feedImageLocalRelpath ??
      props.episodeImageUrl ??
      props.feedImageUrl,
  )
  const isEpisodeLike = dt === 'episode' || dt === 'transcript'
  if (isEpisodeLike && hasArtwork) {
    return {
      kind: 'image',
      title: `${tierBit()}Episode${subtitleBit}`,
      episodeImageLocalRelpath: props.episodeImageLocalRelpath ?? null,
      feedImageLocalRelpath: props.feedImageLocalRelpath ?? null,
      feedImageUrl: props.feedImageUrl ?? null,
      episodeImageUrl: props.episodeImageUrl ?? null,
      alt: `${tierBit()}Episode${subtitleBit}`.trim() || 'Episode',
    }
  }
  switch (dt) {
    case 'insight':
      return {
        kind: 'letter',
        letter: 'I',
        classes: 'bg-primary/15 text-primary',
        title: `${tierBit()}Insight${subtitleBit}`,
      }
    case 'quote':
      return {
        kind: 'letter',
        letter: 'Q',
        classes: 'bg-warning/15 text-warning',
        title: `${tierBit()}Quote${subtitleBit}`,
      }
    case 'kg_topic':
      return {
        kind: 'letter',
        letter: 'T',
        classes: 'bg-accent/15 text-accent',
        title: `${tierBit()}Topic${subtitleBit}`,
      }
    case 'kg_entity':
      return {
        kind: 'letter',
        letter: 'P',
        classes: 'bg-success/15 text-success',
        title: `${tierBit()}Person / Entity${subtitleBit}`,
      }
    case 'summary':
      return {
        kind: 'letter',
        letter: 'S',
        classes: 'bg-overlay text-muted',
        title: `${tierBit()}Summary bullet${subtitleBit}`,
      }
    case 'summary_short':
      return {
        kind: 'letter',
        letter: 'S',
        classes: 'bg-overlay text-muted',
        title: `${tierBit()}Summary${subtitleBit}`,
      }
    case 'episode_title':
      return {
        kind: 'letter',
        letter: 'E',
        classes: 'bg-primary/15 text-primary',
        title: `${tierBit()}Episode title${subtitleBit}`,
      }
    case 'episode_description':
      return {
        kind: 'letter',
        letter: 'E',
        classes: 'bg-primary/15 text-primary',
        title: `${tierBit()}Episode description${subtitleBit}`,
      }
    case 'transcript':
    case 'episode':
      return {
        kind: 'letter',
        letter: 'E',
        classes: 'bg-primary/15 text-primary',
        title: `${tierBit()}Episode${subtitleBit}`,
      }
    default: {
      const letter = (dt[0] || '?').toUpperCase()
      return {
        kind: 'letter',
        letter,
        classes: 'bg-overlay text-muted',
        title: `${tierBit()}${dt}${subtitleBit}`,
      }
    }
  }
})

const kind = computed<IconKind>(() => meta.value.kind)
</script>

<template>
  <span
    class="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded overflow-hidden"
    data-testid="search-result-row-icon"
    :data-doc-type="docType"
    :title="meta.title"
    :aria-label="meta.title"
  >
    <template v-if="kind === 'image'">
      <PodcastCover
        :episode-image-local-relpath="(meta as ImageIcon).episodeImageLocalRelpath"
        :feed-image-local-relpath="(meta as ImageIcon).feedImageLocalRelpath"
        :feed-image-url="(meta as ImageIcon).feedImageUrl"
        :episode-image-url="(meta as ImageIcon).episodeImageUrl"
        :corpus-path="shell.corpusPath"
        :alt="(meta as ImageIcon).alt"
        size-class="h-6 w-6"
        frameless
      />
    </template>
    <span
      v-else
      class="inline-flex h-full w-full items-center justify-center text-[10px] font-semibold leading-none"
      :class="(meta as LetterIcon).classes"
    >
      {{ (meta as LetterIcon).letter }}
    </span>
  </span>
</template>
