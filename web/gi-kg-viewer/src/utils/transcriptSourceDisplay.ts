/** Build `/api/corpus/text-file` URL for opening a transcript in a new browser tab. */
export function corpusTextFileViewUrl(corpusRoot: string, relpath: string): string {
  const path = encodeURIComponent(corpusRoot.trim())
  const rel = encodeURIComponent(relpath.trim())
  return `/api/corpus/text-file?path=${path}&relpath=${rel}`
}

/**
 * Derive ``media/<stem>.mp3`` from a ``transcripts/…/*.txt`` corpus relpath.
 *
 * The backend flattens persisted audio into a single ``media/`` directory keyed by
 * the transcript **stem** (subdirectories dropped), so we flatten the same way —
 * preserving subdirs produced a path the backend never writes (404). The ``.mp3``
 * is a hint only; the media route resolves the real extension by stem.
 */
export function audioRelpathFromTranscriptRelpath(transcriptRelpath: string): string {
  const norm = transcriptRelpath.trim().replace(/\\/g, '/').replace(/^\/+/, '')
  const stem = (norm.replace(/\.txt$/i, '').split('/').pop() || '').trim()
  return `media/${stem}.mp3`
}

/** Build `/api/corpus/media` URL for local episode playback (Wave 3). */
export function corpusMediaFileViewUrl(corpusRoot: string, relpath: string): string {
  const path = encodeURIComponent(corpusRoot.trim())
  const rel = encodeURIComponent(relpath.trim())
  return `/api/corpus/media?path=${path}&relpath=${rel}`
}
