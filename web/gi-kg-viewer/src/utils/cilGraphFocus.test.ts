import { describe, expect, it, vi } from 'vitest'
import {
  applyGraphFocusPlan,
  cameraIncludeRawIdsFromCilPill,
  graphFocusPlanFromCilPill,
  requestGraphFocusFromCilPill,
} from './cilGraphFocus'

describe('cameraIncludeRawIdsFromCilPill', () => {
  it('returns compound id when clustered', () => {
    expect(
      cameraIncludeRawIdsFromCilPill({
        topic_id: 'topic:x',
        in_topic_cluster: true,
        topic_cluster_compound_id: 'tc:parent',
      }),
    ).toEqual(['tc:parent'])
  })

  it('returns empty when not clustered', () => {
    expect(
      cameraIncludeRawIdsFromCilPill({
        topic_id: 'topic:x',
        in_topic_cluster: false,
        topic_cluster_compound_id: 'tc:ignored',
      }),
    ).toEqual([])
  })

  it('returns empty when clustered but compound missing', () => {
    expect(
      cameraIncludeRawIdsFromCilPill({
        topic_id: 'topic:x',
        in_topic_cluster: true,
        topic_cluster_compound_id: null,
      }),
    ).toEqual([])
  })
})

describe('graphFocusPlanFromCilPill', () => {
  it('topic + episode includes camera when clustered', () => {
    expect(
      graphFocusPlanFromCilPill(
        {
          topic_id: 'topic:a',
          in_topic_cluster: true,
          topic_cluster_compound_id: 'tc:c1',
        },
        'e99',
      ),
    ).toEqual({
      kind: 'topic',
      primary: 'topic:a',
      fallback: 'e99',
      cameraInclude: ['tc:c1'],
    })
  })

  it('topic + episode omits third arg when not clustered', () => {
    expect(
      graphFocusPlanFromCilPill(
        {
          topic_id: 'topic:b',
          in_topic_cluster: false,
          topic_cluster_compound_id: null,
        },
        'e1',
      ),
    ).toEqual({
      kind: 'topic',
      primary: 'topic:b',
      fallback: 'e1',
      cameraInclude: undefined,
    })
  })

  it('episode only when no pill topic', () => {
    expect(graphFocusPlanFromCilPill(undefined, 'e2')).toEqual({
      kind: 'episode_only',
      primary: 'e2',
    })
  })

  it('topic only passes camera for clustered pill', () => {
    expect(
      graphFocusPlanFromCilPill(
        {
          topic_id: 'topic:z',
          in_topic_cluster: true,
          topic_cluster_compound_id: 'tc:z',
        },
        null,
      ),
    ).toEqual({
      kind: 'topic',
      primary: 'topic:z',
      fallback: null,
      cameraInclude: ['tc:z'],
    })
  })

  it('none when no ids', () => {
    expect(graphFocusPlanFromCilPill(null, null)).toEqual({ kind: 'none' })
    expect(
      graphFocusPlanFromCilPill(
        { topic_id: '', in_topic_cluster: false, topic_cluster_compound_id: null },
        '',
      ),
    ).toEqual({ kind: 'none' })
  })
})

describe('applyGraphFocusPlan', () => {
  it('dispatches requestFocusNode with camera for topic + fallback', () => {
    const requestFocusNode = vi.fn()
    const clearPendingFocus = vi.fn()
    applyGraphFocusPlan(
      { requestFocusNode, clearPendingFocus },
      {
        kind: 'topic',
        primary: 'topic:t',
        fallback: 'ep1',
        cameraInclude: ['tc:x'],
      },
    )
    expect(requestFocusNode).toHaveBeenCalledWith('topic:t', 'ep1', ['tc:x'])
    expect(clearPendingFocus).not.toHaveBeenCalled()
  })

  it('dispatches topic-only with undefined fallback', () => {
    const requestFocusNode = vi.fn()
    const clearPendingFocus = vi.fn()
    applyGraphFocusPlan(
      { requestFocusNode, clearPendingFocus },
      {
        kind: 'topic',
        primary: 'topic:t',
        fallback: null,
        cameraInclude: undefined,
      },
    )
    expect(requestFocusNode).toHaveBeenCalledWith('topic:t', undefined, undefined)
  })
})

describe('requestGraphFocusFromCilPill', () => {
  it('clears when plan is none', () => {
    const requestFocusNode = vi.fn()
    const clearPendingFocus = vi.fn()
    requestGraphFocusFromCilPill({ requestFocusNode, clearPendingFocus }, null, null)
    expect(clearPendingFocus).toHaveBeenCalled()
    expect(requestFocusNode).not.toHaveBeenCalled()
  })
})
