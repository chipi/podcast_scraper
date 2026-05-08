# Camera Positioning Bug - Root Cause Analysis

## Bug Confirmed ✓

**Issue:** When clicking multiple topic pills from the same episode in Digest, the node selection changes correctly, but the camera does NOT center on the newly selected node.

## Test Evidence

### Automated Test Results (2026-05-07 23:44 UTC+2)

**Test Flow:**

1. Graph tab with "historical-context" selected
2. Switch to Digest tab
3. Click "community development" pill (same Baltimore episode)

**Results:**

| Step | Selected Node | Node Position | Camera Pan | Zoom |
| --- | --- | --- | --- | --- |
| 1. Initial | historical-context | (2498, 858) | (-2608, -596) | 1.16 |
| 3. After click | **community-development** | **(2110, 1135)** | **(-2608, -596)** | **1.30** |

### Analysis

✅ **Working:**

- Node selection changes (`historical-context` → `community-development`)
- Zoom changes (1.16 → 1.30)
- Detail panel opens with correct topic
- `tryApplyPendingFocus()` IS being called (evidenced by zoom change)

❌ **Broken:**

- **Camera pan does NOT change** (stays at -2608, -596)
- Should have moved to center on node at (2110, 1135)
- `animateCameraToFocusedNode()` zoom works, but center doesn't

## Root Cause Hypothesis

The `animateCameraToFocusedNode()` function calls:

```javascript
core.animate({
  center: { eles: centerEles },
  zoom: targetZoom,
  duration: 320,
})
```

The `zoom` part works (changes to 1.30), but the `center` part doesn't (camera pan unchanged).

**Possible causes:**

1. Animation is being interrupted/cancelled by another operation
2. `pendingViewportPreserve` is overriding the camera position after animation
3. The animation completes but something immediately resets the pan
4. When called from the `pendingFocusNodeId` watcher (not during layout), the center animation is somehow skipped

## Next Steps

1. Add console logging to `animateCameraToFocusedNode` to verify it's called
2. Add logging to check if animation completes or gets interrupted
3. Check if any other code is resetting camera pan after the animation
4. Investigate if `pendingViewportPreserve` restoration is interfering

## Files Involved

- `web/gi-kg-viewer/src/components/graph/GraphCanvas.vue`
  - Line 1604: `animateCameraToFocusedNode()`
  - Line 1654: `tryApplyPendingFocus()`
  - Line 2769: New `pendingFocusNodeId` watcher
  - Line 1313: `applyViewportPreserveOrFit()` - might be interfering?

## Test Document Created

- `web/gi-kg-viewer/e2e/DIGEST_MULTI_PILL_TEST_PLAN.md`
- `web/gi-kg-viewer/e2e/DIGEST_MULTI_PILL_TEST_RESULTS.md`
- `web/gi-kg-viewer/e2e/CAMERA_POSITIONING_BUG_ANALYSIS.md` (this file)
