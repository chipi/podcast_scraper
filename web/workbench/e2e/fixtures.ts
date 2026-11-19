import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

/**
 * The graph the offline specs load through the file picker: one episode mentioning one topic and
 * one person. Its ids match the GI sample's (`episode:ci-fixture`, `topic:ci-policy`), so a downstream
 * build can load either.
 */
export const KG_SAMPLE_FIXTURE = path.join(__dirname, 'fixtures', 'ci_sample.kg.json')
