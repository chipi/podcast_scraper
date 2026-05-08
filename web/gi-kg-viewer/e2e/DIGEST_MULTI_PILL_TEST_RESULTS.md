# Digest Multi-Pill Navigation Test Results

**Date:** 2026-05-07 23:19 UTC+2
**Test Execution:** Automated via Chrome DevTools MCP

## Test 1: Same Episode - Multiple Pills (Baltimore Episode)

### Setup

- Episode: "How Baltimore's Mayor Is Fighting the City's Vacant Housing Crisis"
- Pills available: public investment, urban planning, city government, community development, historical context (5 total)

### Execution

#### Click 1: "public investment"

- ✅ **SUCCESS**
- Zoom: 1.30 (130%)
- Selected: g:topic:public-investment
- Node count: 124
- Edge count: 275
- Behavior: Graph loaded, node selected, camera centered

#### Clicks 2-5: Subsequent pills

- ❌ **BLOCKED BY UI FLOW**
- Error: Pills not found after click 1
- Reason: After first click, app navigates to Graph tab, Digest pills no longer accessible
- This is **expected behavior** - user must navigate back to Digest to click another pill

### Real User Flow Test (Manual Simulation Required)

To properly test "same episode, multiple pills", the user flow is:

1. Click pill A from Digest → Graph opens
2. **Navigate back to Digest**
3. Click pill B from same episode → Graph should update
4. **Navigate back to Digest**
5. Click pill C from same episode → Graph should update

### Key Question for User

**Does the issue occur when:**

- User clicks pill A
- User goes back to Digest (clicks Digest tab)
- User clicks pill B from **the same episode**
- Graph should navigate to pill B's topic

**OR does it occur when:**

- User has the Graph tab open
- User views an episode's detail panel (which shows pills)
- User clicks multiple pills from that panel without leaving Graph

## Recommendations

1. **Clarify the exact user flow** where the issue occurs
2. **Test the episode detail panel** pills (if those exist in Graph view)
3. **Test Library view** pills (similar issue might exist there)

## Technical Observations

### What Works

- First click from Digest always works
- External navigation flag (`digest-external`) is set correctly
- Full layout is applied (confirmed by `isExternalNavigation` check)

### What Might Still Be Broken

- **Pending focus watcher** might not fire when:
  - Artifacts are already loaded (same episode)
  - No `filteredArtifact` watcher fires
  - Need to verify the new `pendingFocusNodeId` watcher actually triggers

### Console Logs to Verify

```text
[GraphCanvas filteredArtifact watcher] { loadSource: 'digest-external', hasPendingFocus: true }
[GraphCanvas redraw] Decision: { isExternalNavigation: true, useIncrementalLayout: false }
```

## Next Steps

1. User clarifies exact reproduction steps
2. Execute manual test with those exact steps
3. Monitor console logs during reproduction
4. Verify `pendingFocusNodeId` watcher fires
