import { describe, expect, it } from 'vitest'
import {
  mergeOperatorYamlProfile,
  splitOperatorYamlProfile,
  yamlScalarForProfileLine,
} from './operatorYamlProfile'

describe('operatorYamlProfile', () => {
  it('splits top-level profile line', () => {
    expect(splitOperatorYamlProfile('profile: cloud_balanced\nmax_episodes: 2\n')).toEqual({
      profile: 'cloud_balanced',
      body: 'max_episodes: 2',
    })
  })

  it('returns empty profile when absent', () => {
    expect(splitOperatorYamlProfile('batch_size: 1\n')).toEqual({
      profile: '',
      body: 'batch_size: 1',
    })
  })

  it('merges profile and body', () => {
    expect(mergeOperatorYamlProfile('local', 'max_episodes: 3')).toBe('profile: local\nmax_episodes: 3\n')
  })

  it('merges profile only', () => {
    expect(mergeOperatorYamlProfile('airgapped', '')).toBe('profile: airgapped\n')
  })

  it('omits profile when none', () => {
    expect(mergeOperatorYamlProfile('', 'x: 1')).toBe('x: 1\n')
  })

  it('quotes profile names that need YAML quoting', () => {
    expect(yamlScalarForProfileLine('cloud_balanced')).toBe('cloud_balanced')
    expect(yamlScalarForProfileLine('weird name')).toBe('"weird name"')
    expect(mergeOperatorYamlProfile('weird name', 'x: 1')).toBe('profile: "weird name"\nx: 1\n')
  })

  it('parses quoted profile value from file', () => {
    expect(splitOperatorYamlProfile('profile: "weird name"\nx: 1\n')).toEqual({
      profile: 'weird name',
      body: 'x: 1',
    })
  })
})
