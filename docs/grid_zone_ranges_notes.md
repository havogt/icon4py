# ICON4Py Grid Zone Ranges — Findings Summary

Reference notes on how horizontal grid zones (lateral boundary, nudging, interior, halo) are laid out in index space for ICON grids. Compiled from exploration of the `icon4py` codebase.

## 1. Zone enum — key values

Defined in `model/common/src/icon4py/model/common/grid/horizontal.py`.

| Zone | Level | Applies to |
|---|---|---|
| `END` | 0 | num entries in local grid |
| `INTERIOR` | 0 | interior unordered prognostic entries |
| `LOCAL` | 0 | all entries owned on local grid (excludes halo) |
| `HALO` | 1 | 1st halo line (distributed only) |
| `HALO_LEVEL_2` | 2 | 2nd halo line (distributed only) |
| `LATERAL_BOUNDARY` | 1 | boundary row 1 (LAM only) |
| `LATERAL_BOUNDARY_LEVEL_2..4` | 2-4 | further boundary rows (LAM only) |
| `LATERAL_BOUNDARY_LEVEL_5..8` | 5-8 | **EdgeDim only** |
| `NUDGING` | 1 | 1st nudging level (LAM only) |
| `NUDGING_LEVEL_2` | 2 | 2nd nudging level (**EdgeDim only**) |

## 2. Boundary/nudging counts per dimension

| Dimension | Boundary levels | Nudging levels | `_GRID_REFINEMENT_BOUNDARY_WIDTH` | `_MAX_BOUNDARY_DISTANCE` |
|---|---|---|---|---|
| CellDim | 4 | 1 | 4 | 14 |
| EdgeDim | 8 | 2 | 9 | 28 |
| VertexDim | 4 | 1 | 4 | 14 |

Constants live in `model/common/src/icon4py/model/common/grid/grid_refinement.py`:
- `_GRID_REFINEMENT_BOUNDARY_WIDTH`: CellDim=4, EdgeDim=9, VertexDim=4
- `_MAX_BOUNDARY_DISTANCE`: CellDim=14, EdgeDim=28, VertexDim=14
- `DEFAULT_GRF_NUDGEZONE_WIDTH`: 8

## 3. Index-space ordering (invariant)

For every dimension, indices are always laid out in this order:

```
LATERAL_BOUNDARY (LB1) → LB2 → ... → LBn
  → NUDGING [→ NUDGING_LEVEL_2]
  → INTERIOR (unordered)
  → [END = size of LOCAL]
  → HALO → HALO_LEVEL_2   (distributed only)
```

`LOCAL` spans `[0, END)` on the local rank. `END = HALO_LEVEL_2.end` when distributed.

## 4. The four cases

### Case A: Global grid, single rank
- Lateral boundary, nudging, halo zones all empty (start == end).
- Entire grid is INTERIOR = LOCAL.
- Example: `R02B04_GLOBAL` (20480 cells, 30720 edges, 10242 vertices).

### Case B: Global grid, multi rank
- Lateral boundary and nudging empty.
- INTERIOR is a subset (the rank's owned points), followed by HALO and HALO_LEVEL_2.

### Case C: Regional (LAM) grid, single rank
- Full set of boundary rows + nudging + interior.
- Halo empty.
- Example: `MCH_OPR_R04B07_DOMAIN01` (10700 cells, 16209 edges, 5510 vertices).

### Case D: Regional (LAM) grid, multi rank
- All boundary + nudging points are present on each rank (replicated, owned by some rank).
- Plus a fraction of interior, plus halo.
- Example: 4-rank decomposition of `MCH_CH_R04B09` — ~5230 cells/rank.

## 5. Ground-truth data sources

| File | What it contains |
|---|---|
| `model/common/tests/common/grid/unit_tests/test_grid_refinement.py` | Exact bounds for `MCH_OPR_R04B07_DOMAIN01` (regional/single-rank) |
| `model/common/tests/common/grid/mpi_tests/test_parallel_icon.py` | 2-rank and 4-rank `LOCAL_IDX` and `HALO_IDX` tuples for `MCH_CH_R04B09` |
| `model/common/src/icon4py/model/common/grid/horizontal.py` | `Zone` enum, `_ZONE_TO_INDEX_MAPPING`, ICON constant mappings |
| `model/common/src/icon4py/model/common/grid/grid_refinement.py` | `compute_domain_bounds()`, boundary/nudging width constants |
| `model/testing/src/icon4py/model/testing/definitions.py` | Grid descriptions (R02B04_GLOBAL, MCH_OPR_R04B07_DOMAIN01, etc.) |

## 6. Reference data: MCH_OPR_R04B07_DOMAIN01 (regional, single-rank)

### cell_bounds (10700 total)
```python
Zone.LATERAL_BOUNDARY:         (0,    629)
Zone.LATERAL_BOUNDARY_LEVEL_2: (629,  1244)
Zone.LATERAL_BOUNDARY_LEVEL_3: (1244, 1843)
Zone.LATERAL_BOUNDARY_LEVEL_4: (1843, 2424)
Zone.NUDGING:                  (2424, 2989)
Zone.INTERIOR:                 (2989, 10700)
Zone.LOCAL:                    (0,    10700)
Zone.END, HALO, HALO_LEVEL_2:  (10700, 10700)
```

### edge_bounds (16209 total)
```python
Zone.LATERAL_BOUNDARY:         (0,    318)
Zone.LATERAL_BOUNDARY_LEVEL_2: (318,  947)
Zone.LATERAL_BOUNDARY_LEVEL_3: (947,  1258)
Zone.LATERAL_BOUNDARY_LEVEL_4: (1258, 1873)
Zone.LATERAL_BOUNDARY_LEVEL_5: (1873, 2177)
Zone.LATERAL_BOUNDARY_LEVEL_6: (2177, 2776)
Zone.LATERAL_BOUNDARY_LEVEL_7: (2776, 3071)
Zone.LATERAL_BOUNDARY_LEVEL_8: (3071, 3652)
Zone.NUDGING:                  (3652, 3938)
Zone.NUDGING_LEVEL_2:          (3938, 4503)
Zone.INTERIOR:                 (4503, 16209)
Zone.LOCAL:                    (0,    16209)
Zone.END, HALO, HALO_LEVEL_2:  (16209, 16209)
```

### vertex_bounds (5510 total)
```python
Zone.LATERAL_BOUNDARY:         (0,    318)
Zone.LATERAL_BOUNDARY_LEVEL_2: (318,  629)
Zone.LATERAL_BOUNDARY_LEVEL_3: (629,  933)
Zone.LATERAL_BOUNDARY_LEVEL_4: (933,  1228)
Zone.NUDGING:                  (1228, 1514)
Zone.INTERIOR:                 (1514, 5510)
Zone.LOCAL:                    (0,    5510)
Zone.END, HALO, HALO_LEVEL_2:  (5510, 5510)
```

## 7. Multi-rank halo data: MCH_CH_R04B09, 4 ranks

From `test_parallel_icon.py`:

**LOCAL_IDX_4** (owned size per rank):
- CellDim:   {0: 5238, 1: 5222, 2: 5231, 3: 5230}
- EdgeDim:   {0: 7929, 1: 7838, 2: 7955, 3: 7889}
- VertexDim: {0: 2688, 1: 2612, 2: 2723, 3: 2656}

**HALO_IDX_4** as `(halo_start, halo_level_2_start, end)`:
- CellDim rank 0: (5238, 5340, 5446)
- EdgeDim rank 0: (7929, 7966, 8173)
- VertexDim rank 0: (2688, 2727, 2834)

## 8. Typical zone fractions (dict form)

```python
from icon4py.model.common.grid.horizontal import Zone

cell_zone_fractions: dict[Zone, float] = {
    Zone.LATERAL_BOUNDARY:         0.059,
    Zone.LATERAL_BOUNDARY_LEVEL_2: 0.057,
    Zone.LATERAL_BOUNDARY_LEVEL_3: 0.056,
    Zone.LATERAL_BOUNDARY_LEVEL_4: 0.054,
    Zone.NUDGING:                  0.053,
    Zone.INTERIOR:                 0.721,
    Zone.HALO:                     0.019,  # distributed only
    Zone.HALO_LEVEL_2:             0.020,  # distributed only
}

edge_zone_fractions: dict[Zone, float] = {
    Zone.LATERAL_BOUNDARY:         0.020,
    Zone.LATERAL_BOUNDARY_LEVEL_2: 0.039,
    Zone.LATERAL_BOUNDARY_LEVEL_3: 0.019,
    Zone.LATERAL_BOUNDARY_LEVEL_4: 0.038,
    Zone.LATERAL_BOUNDARY_LEVEL_5: 0.019,
    Zone.LATERAL_BOUNDARY_LEVEL_6: 0.037,
    Zone.LATERAL_BOUNDARY_LEVEL_7: 0.018,
    Zone.LATERAL_BOUNDARY_LEVEL_8: 0.036,
    Zone.NUDGING:                  0.018,
    Zone.NUDGING_LEVEL_2:          0.035,
    Zone.INTERIOR:                 0.722,
    Zone.HALO:                     0.005,  # distributed only
    Zone.HALO_LEVEL_2:             0.026,  # distributed only
}

vertex_zone_fractions: dict[Zone, float] = {
    Zone.LATERAL_BOUNDARY:         0.058,
    Zone.LATERAL_BOUNDARY_LEVEL_2: 0.056,
    Zone.LATERAL_BOUNDARY_LEVEL_3: 0.055,
    Zone.LATERAL_BOUNDARY_LEVEL_4: 0.054,
    Zone.NUDGING:                  0.052,
    Zone.INTERIOR:                 0.725,
    Zone.HALO:                     0.015,  # distributed only
    Zone.HALO_LEVEL_2:             0.040,  # distributed only
}
```

**Edge boundary small/large pattern:** odd rows (~1.9%) contain new edges at the ring perimeter; even rows (~3.7%) add edges connecting back to the adjacent cell boundary row.

## 9. Deliverables produced

- `docs/grid_zone_visualization.html` — interactive HTML visualization with tabs for all 4 cases (global/regional × single/multi-rank). Features: stacked color-coded bars for cells/edges/vertices, hover tooltips showing zone name/index range/percentage, LOCAL bracket beneath each bar, legend.
- Committed and pushed to branch `claude/define-grid-zone-ranges-9zc6S`.

## 10. Caveats

- Regional proportions come from the actual test data in `test_grid_refinement.py` — these are exact.
- Halo proportions come from the `MCH_CH_R04B09` 4-rank decomposition in `test_parallel_icon.py` — exact for that case but vary with rank count and grid.
- Global multi-rank proportions in the HTML are **approximate** (1/4 split); no test fixture with explicit global halo bounds was found.
- No grid files were downloaded or executed; all data is from source/test code.
