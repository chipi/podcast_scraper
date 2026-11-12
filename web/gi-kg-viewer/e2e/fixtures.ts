import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

/** Bundled GI sample (aligned with tests/fixtures/gil_kg_ci_enforce/metadata/ci_sample.gi.json). */
export const GI_SAMPLE_FIXTURE = path.join(__dirname, 'fixtures', 'ci_sample.gi.json')

/**
 * The graph the offline specs load through the file picker: one episode mentioning one topic and
 * one person. Its ids match the GI sample's (`episode:ci-fixture`, `topic:ci-policy`), so a downstream
 * build can load either.
 */
export const KG_SAMPLE_FIXTURE = path.join(__dirname, 'fixtures', 'ci_sample.kg.json')
