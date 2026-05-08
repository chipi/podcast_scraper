# Graph Incremental Loading - E2E Test Evaluation Criteria

## Overview

This document defines the comprehensive test scenarios and validation criteria for graph incremental loading behavior, including the fixes for:

- **Fix #1**: Topic cluster "Load" button path tracking
- **Fix #2**: Digest topic cluster selection and focus
- **Fix #3**: Camera centering and zoom behavior

## Pre-Fix Validation (Scenarios Tested Before Issues Were Found)

These scenarios were tested and **passed** before the 3 issues were identified, confirming the foundational incremental loading behavior works correctly:

### Pre-Fix Scenario 1: Digest Topic Pills (Individual Episodes)

**Entry Point**: Digest → Click 10 different episode pills alternating between clusters and normal pills
**Result**: ✓ PASSED

- Episodes loaded incrementally (8 → 11 → 14 → ...)
- No full graph redraw/flash
- Each episode correctly selected and focused
- Episode detail panels opened correctly
- Camera centered on each selected episode

### Pre-Fix Scenario 2: Dashboard TopicClusters

**Entry Point**: Dashboard/Intelligence → Click multiple topic cluster cards
**Result**: ✓ PASSED

- Clusters loaded with their members
- NodeDetail panel showed cluster information
- Incremental loading worked correctly
- Selection state correct for each cluster

### Pre-Fix Scenario 3: Graph Node Neighbourhood

**Entry Point**: Graph → Select node → Expand neighbourhood
**Result**: ✓ PASSED

- Connected nodes appeared incrementally
- Selected node remained selected
- No full graph wipe
- Camera adjusted appropriately

### Pre-Fix Scenario 4: Graph Node Lists (in Detail Panel)

**Entry Point**: Graph → NodeDetail panel → Click related node in a list
**Result**: ✓ PASSED

- Navigation to related node worked
- New node selected and focused
- Detail panel updated correctly
- Incremental navigation

### Pre-Fix Scenario 5: Search Results

**Entry Point**: Semantic search → Click "Open in graph" on result
**Result**: ✓ PASSED

- Graph activated and result node selected
- Detail panel opened for search result
- Camera focused on result
- Incremental load

### Pre-Fix Scenario 6: Graph Internal Links

**Entry Point**: Graph → Click links within the graph visualization
**Result**: ✓ PASSED

- Node selection updated
- Detail panel changed to show clicked node
- Camera followed selection
- No unnecessary redraws

### Pre-Fix Scenario 7: Library Episode "Open in Graph"

**Entry Point**: Library → Episode row → Click graph icon
**Result**: ✓ PASSED

- Episode appeared in graph and was selected
- Episode detail panel opened
- Camera centered on episode
- Incremental load

### Pre-Fix Scenario 8: Mixed Entry Point Sequence

**Entry Point**: Digest → Dashboard → Graph internal → Search
**Result**: ✓ PASSED (with load source tracking)

- Each entry point worked independently
- No state contamination between entry points
- Incremental loading maintained throughout
- `lastLoadSource` correctly tracked and cleared

### Pre-Fix Scenario 9: Rapid Sequential Clicks (Same Source)

**Entry Point**: Digest → Click 5 episode pills rapidly (< 2s apart)
**Result**: ✓ PASSED

- Last click won (correct race condition handling)
- All episodes loaded incrementally
- Final state matched last click
- No console errors

### Pre-Fix Scenario 10: Feed Scope Navigation

**Entry Point**: Library → Feed scope → "Open graph"
**Result**: ✓ PASSED

- All feed episodes loaded
- Graph showed all episodes from feed
- Camera fit to show feed episodes
- Incremental load

### Pre-Fix Scenario 11: Dashboard "Show More" Members

**Entry Point**: Dashboard → Cluster card → "Show more members"
**Result**: ✓ PASSED

- Additional cluster members loaded incrementally
- Node count increased correctly
- Cluster remained selected
- No redraw

### Pre-Fix Scenario 12: Cross-Tab Navigation Preservation

**Entry Point**: Graph (with nodes selected) → Switch to Digest → Switch back to Graph
**Result**: ✓ PASSED

- Graph state preserved across tab switches
- Selected nodes remained selected
- Camera position preserved
- No unnecessary reload

### Pre-Fix Scenario 13: Auto-Merge Behavior (Graph-Internal Loads)

**Entry Point**: Graph → Load cluster → Sibling merge triggers
**Result**: ✓ PASSED (with `loadSource = 'graph-internal'`)

- Auto-merge executed as expected
- Sibling topics merged with loaded episodes
- Incremental addition of merged nodes
- Load source tracking enabled this behavior

### Pre-Fix Scenario 14: No Auto-Merge (External Loads)

**Entry Point**: Digest → Load category band
**Result**: ✓ PASSED (with `loadSource = 'digest-external'`)

- Auto-merge correctly skipped
- Only requested episodes loaded
- No unwanted topic merges
- Load source tracking prevented contamination

### Pre-Fix Scenario 15: Multiple Sequential Digest Category Bands

**Entry Point**: Digest → "Science & research" → "Technology" → "Business & markets"
**Result**: ✓ PASSED

- Each category loaded incrementally
- Episodes from all categories accumulated in graph
- No cross-category contamination
- Load source correctly managed

## Issues Found During Validation

After the above 15+ scenarios passed, manual deep inspection revealed 3 issues:

1. **Issue #1**: When clicking "Load" in NodeDetail panel for TopicCluster members, the shell footer path wasn't updating to show newly loaded episodes
2. **Issue #2**: When clicking a Digest topic cluster band, the cluster wasn't being selected/focused in the graph, and the detail panel wasn't showing cluster information
3. **Issue #3**: Camera was defaulting to 14% zoom in the upper-left corner instead of centering at an appropriate zoom level

These issues were **not caught by the initial pass** because they required specific inspection of:

- Path tracking in the shell footer
- Detail panel content (vs just presence)
- Exact camera zoom percentage and position

## Test Scenarios

### Category 1: Digest Entry Points

#### Scenario 1.1: Digest Category Band (without cluster)

**Entry Point**: Digest → "Open graph for topic Science & research"
**Expected Behavior**:

- Graph loads 8 episodes from the category
- Camera centers at **50% zoom** (not 14% in upper-left)
- Right panel shows "Select an episode..." (no detail panel)
- No node is selected
- Incremental loading (no full wipe)

**Validation**:

```typescript
- cy.zoom() >= 0.45 && cy.zoom() <= 0.55
- cy.nodes(':selected').length === 0
- rightPanel contains "Select an episode"
- graph does not flash/blank during load
```

#### Scenario 1.2: Digest Topic Cluster (with cluster)

**Entry Point**: Digest → Click topic cluster pill that maps to a TopicCluster (e.g., specific AI topic)
**Expected Behavior**:

- Graph navigates to Graph tab
- TopicCluster compound node (tc:...) is selected and focused
- NodeDetail panel opens showing TopicCluster details
- Camera focuses on the cluster node
- Incremental loading

**Validation**:

```typescript
- cy.nodes(':selected').length === 1
- cy.nodes(':selected').id().startsWith('tc:')
- NodeDetail panel is visible
- subject.currentGraphNodeId === clusterId
```

#### Scenario 1.3: Digest Episode Pill

**Entry Point**: Digest → Click "Open graph and episode details" for an individual episode
**Expected Behavior**:

- Graph navigates to Graph tab
- Episode node is selected
- Episode detail panel opens
- Camera centers on episode
- Incremental loading

**Validation**:

```typescript
- cy.nodes(':selected').length === 1
- cy.nodes(':selected').id().startsWith('e:')
- Episode detail panel shows episode title
- camera is centered on episode node
```

### Category 2: Dashboard Entry Points

#### Scenario 2.1: Dashboard Topic Cluster

**Entry Point**: Dashboard/Intelligence → Click topic cluster card
**Expected Behavior**:

- Graph loads cluster and members
- TopicCluster is selected
- NodeDetail shows cluster info
- Camera focuses on cluster
- Incremental loading

**Validation**:

```typescript
- tc: node is selected
- NodeDetail shows cluster members list
- incremental load (nodes increase, not reset)
```

#### Scenario 2.2: Dashboard → Load More Members

**Entry Point**: Dashboard → Click cluster → "Show more" in cluster members list
**Expected Behavior**:

- Additional episodes load incrementally
- Node count increases
- No graph redraw/flash
- Cluster remains selected

**Validation**:

```typescript
- before_count < after_count
- cy.nodes(':selected').id() unchanged
- no full redraw event
```

### Category 3: Graph Internal Navigation

#### Scenario 3.1: Node Detail "Load" Button (Fix #1)

**Entry Point**: Graph → Select TopicCluster → Click "Load" in NodeDetail panel
**Expected Behavior**:

- `artifacts.setLoadSource('graph-internal')` is set
- Additional cluster member episodes load
- Path in shell footer updates to show new episodes
- `artifacts.clearLoadSource()` is called
- Node count increases
- Cluster remains selected

**Validation**:

```typescript
- artifacts.lastLoadSource === 'graph-internal' (during load)
- shell.corpusPath includes newly loaded episodes
- before_node_count < after_node_count
- cy.nodes(':selected').id() === cluster_id
```

#### Scenario 3.2: Node Neighbourhood Expansion

**Entry Point**: Graph → Select node → Click "Show connections" or similar
**Expected Behavior**:

- Connected nodes appear incrementally
- Selected node remains selected
- Camera adjusts to show new nodes
- No full redraw

**Validation**:

```typescript
- before_node_count < after_node_count
- cy.nodes(':selected').id() unchanged
- incremental layout
```

### Category 4: Library Entry Points

#### Scenario 4.1: Library Episode "Open in Graph"

**Entry Point**: Library → Episode row → "Open in graph" icon
**Expected Behavior**:

- Graph tab activates
- Episode node is selected and focused
- Episode detail panel opens
- Camera centers on episode
- Incremental load

**Validation**:

```typescript
- cy.nodes(':selected').id() === episode_id
- episode detail panel visible
- camera centered on episode
```

#### Scenario 4.2: Library Feed Scope "Open Graph"

**Entry Point**: Library → Feed scope button → "Open graph"
**Expected Behavior**:

- All episodes from feed load into graph
- Camera fits to show all feed episodes
- No specific node selected initially
- Incremental load

**Validation**:

```typescript
- cy.nodes().filter(n => n.data('feedId') === feed_id).length === expected_count
- no specific node selected
- all feed episodes visible
```

### Category 5: Search Entry Points

#### Scenario 5.1: Search Result "Open in Graph"

**Entry Point**: Semantic search → Result → "Open in graph"
**Expected Behavior**:

- Graph activates
- Result's episode/topic node is selected
- Detail panel opens
- Camera focuses on result
- Incremental load

**Validation**:

```typescript
- cy.nodes(':selected').length === 1
- selected node matches search result
- detail panel matches result type
```

### Category 6: Sequential Multi-Entry (Cross-Contamination Test)

#### Scenario 6.1: Digest → Dashboard → Graph Internal

**Entry Point**: Sequence of 3 different entry points
**Expected Behavior**:

- Each entry point correctly sets/clears load source
- No "leftover" state from previous entry
- Each load is incremental
- Camera behavior appropriate to each entry type

**Validation**:

```typescript
// After each step:
- artifacts.lastLoadSource is correct or null
- node count increases (never resets unless explicit context switch)
- camera behavior matches entry type
- selected node matches intent
```

#### Scenario 6.2: Rapid Sequential Clicks

**Entry Point**: Click multiple Digest pills rapidly (within 2s)
**Expected Behavior**:

- Last click wins
- No race conditions
- Graph settles on last clicked item
- Incremental loads complete correctly

**Validation**:

```typescript
- final selected node matches last click
- node count reflects all loads
- no errors in console
```

### Category 7: Auto-Merge Behavior (Deterministic Control)

#### Scenario 7.1: Graph-Internal Load Triggers Auto-Merge

**Entry Point**: Graph → Load cluster members → Sibling topics appear
**Expected Behavior**:

- `lastLoadSource === 'graph-internal'`
- Auto-merge runs (sibling topics merge with loaded episodes)
- Merged topics appear in graph

**Validation**:

```typescript
- auto-merge executed
- merged topic nodes visible
- loadSource was 'graph-internal'
```

#### Scenario 7.2: Digest-External Load Does NOT Trigger Auto-Merge

**Entry Point**: Digest → Load category band
**Expected Behavior**:

- `lastLoadSource === 'digest-external'`
- Auto-merge does NOT run
- Only explicitly requested episodes load

**Validation**:

```typescript
- auto-merge skipped
- only requested episodes in graph
- loadSource was 'digest-external'
```

## Cross-Cutting Validation Points

### For ALL Scenarios

#### Camera Behavior

- **With specific focus target (episode/cluster)**: Camera centers on target, appropriate zoom
- **Without specific focus (category band)**: Camera at **50% zoom**, centered on graph center
- **Never**: 14% zoom in upper-left corner

#### Incremental Loading

- Graph does not flash/blank during load
- Node count increases (or stays same for context switches)
- Selected node remains selected when appropriate
- Cytoscape uses incremental layout (`useIncrementalLayout = true`)

#### Load Source Tracking

- `artifacts.lastLoadSource` is set correctly during load
- Always cleared after load completes (in `finally` block)
- Null between loads

#### Path Tracking

- Shell footer `corpusPath` updates when new artifacts load
- Path shows all loaded episodes
- "Load" buttons update path

#### Right Panel Behavior

- Correct detail panel opens (NodeDetail vs Episode vs TopicEntityView)
- "Select an episode..." shows only when no focus target
- Panel content matches selected node

#### Selection State

- `cy.nodes(':selected').length` is correct
- Selected node ID matches intent
- Selection classes applied (`graph-focused`, `graph-neighbour`, `graph-dimmed`)

## Test Execution Notes

### Test Data Requirements

- Corpus with TopicClusters and category bands
- At least 3 category bands with varying episode counts (8, 12, 20+)
- Multiple TopicClusters with members
- Feeds with multiple episodes

### Test Environment

- Clean browser state before each test category
- No cached viewport state
- API and UI servers running
- Corpus loaded

### Failure Modes to Watch For

1. **14% zoom in upper-left** - indicates Fix #3 regression
2. **Empty or wrong detail panel** - indicates Fix #2 regression  
3. **Path not updating** - indicates Fix #1 regression
4. **Full graph redraw** - indicates incremental loading broken
5. **Wrong node selected** - indicates focus logic broken
6. **Auto-merge when it shouldn't** - indicates load source tracking broken

## Success Criteria

A test scenario **passes** when ALL of:

- Camera behavior is correct
- Selection state is correct
- Right panel content is correct
- Incremental loading works (no flash)
- Load source is tracked correctly
- Path updates (when applicable)
- No console errors

## Automation Mapping

Each scenario above should map to one or more Playwright test cases in:

- `web/gi-kg-viewer/e2e/graph-incremental-loading.spec.ts` (new file)

Tests should use:

- `page.locator()` for UI elements
- `page.evaluate()` for Cytoscape state inspection
- Expect assertions for all validation points
- Consistent test data fixtures

## Related Documentation

- `E2E_SURFACE_MAP.md` - UI element selectors
- GitHub Issue #750 - E2E test suite tracking
- This document - Evaluation criteria (what to test)
- `AGENT_BROWSER_LOOP_GUIDE.md` - Manual browser testing with MCP
