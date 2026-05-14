# Digest Multi-Pill Navigation Test Plan

## Test Scenarios

### Test 1: Same Episode - Multiple Pills (CRITICAL)

**Baltimore episode has 5 pills:**

1. Click "public investment"
2. Click "urban planning" (same episode, different topic)
3. Click "city government" (same episode, different topic)
4. Click "community development" (same episode, different topic)
5. Click "historical context" (same episode, different topic)

**Expected:** ALL 5 clicks should navigate to their respective topics in the graph with:

- Camera centered at ~130% zoom
- Topic node selected
- Detail panel opened
- Full layout applied (sparse spacing)

### Test 2: Different Episodes

**Health Revolution episode has 5 pills:**

1. Click "biotech investing" (episode 2)
2. Click "public investment" (episode 1 - different episode)
3. Click "cardiovascular health" (episode 2 - back to same)

**Expected:** ALL clicks work across episode boundaries

### Test 3: Mixed Cluster/Regular Pills

1. Click "public investment" (cluster pill)
2. Click "biotech investing" (regular pill)
3. Click "AI industry" (cluster pill from different episode)

**Expected:** Both cluster and regular pills work

### Test 4: Rapid Clicks (Same Episode)

1. Rapidly click 3 pills from Baltimore episode with <1s delay
2. Last click should "win" (focus on last clicked topic)

### Test 5: Layout Quality

**After each click, verify:**

- Nodes are evenly distributed (not packed in dense clusters)
- New nodes are integrated into existing layout
- Zoom level is appropriate (~130% for focused nodes)

## Validation Criteria

For each click:

- [ ] Graph tab activates
- [ ] Node exists in graph
- [ ] Node is selected (highlighted)
- [ ] Detail panel opens showing correct topic
- [ ] Camera centers on selected node
- [ ] Zoom level is ~130% (1.3)
- [ ] Layout is sparse (nodes not densely packed)
- [ ] Console shows no errors

## Console Logs to Monitor

- `[GraphCanvas redraw] Decision: { isExternalNavigation: true }`
- `[GraphCanvas filteredArtifact watcher] { loadSource: 'digest-external' }`
- NO "Skipping disruptive ops" when clicking different pills
