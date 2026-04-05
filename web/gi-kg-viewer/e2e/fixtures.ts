import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

/** Bundled GI sample (aligned with tests/fixtures/gil_kg_ci_enforce/metadata/ci_sample.gi.json). */
export const GI_SAMPLE_FIXTURE = path.join(__dirname, 'fixtures', 'ci_sample.gi.json')
