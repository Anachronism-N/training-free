# v109 Legacy v98 Suppressive Cache Screen Results

Date: 2026-07-27

## 1. Review Summary

All 5 cells used the 304/56 legacy v98 map with Supportive heads fixed on
sink1+cyclic4+recent4. Only the 56 Suppressive heads changed cache policy.

| Cell | Suppressive cache | Polygon noise | Motion | Verdict |
|---|---|---|---|---|
| all_cyclic_control | cyclic4+sink1 | No | Decreasing | ✅ Clean |
| suppress_cyclic_sink3 | cyclic4+sink3 | No | Decreasing | ✅ Clean |
| suppress_recent8_sink1 | recent8+sink1 | No | Decreasing | ✅ Clean |
| suppress_recent5_sink3 | recent5+sink3 | No | Decreasing | ✅ Clean |
| suppress_merge | merge4+sink3 | No | Decreasing | ✅ Clean |

## 2. Key Findings

1. **No polygon noise in any cell**: The 304/56 map is safe when all
   Supportive heads use cyclic instead of stride. This confirms v107's
   root cause: Wave-to-stride routing, not the map itself.

2. **Merge is clean on 56 heads**: Unlike v97/v98/v99 where merge caused
   noise on 327 heads, merge on only 56 heads produces no visible artifacts.

3. **All cells show decreasing motion**: Subject motion gradually reduces in
   later frames across all configurations. This is likely a property of the
   base generation dynamics, not the cache policy.

4. **Visible but modest differences between policies**: Human review found
   differences among the five videos, but one prompt is insufficient to rank
   them reliably. The result shows that the 56-head route is not inert; it
   does not yet identify the best Suppressive memory.

## 3. Decision

v109 establishes implementation viability, not a final policy. The next
experiment must replace the unexplained cyclic carrier and the periodic
middle policies with content-driven Supportive and Suppressive memories while
holding the old-v98 304/56 membership and total read budget fixed.

## 4. Next Steps

- Run the v111 non-periodic one-video cache screen.
- Select a candidate only after blind visual review and trace audit.
- Promote exactly one candidate to the v112 MovieGenBench-32 comparison.
