# Mapping ICON Fortran to ICON4Py / GT4Py

This document provides a comprehensive mapping between ICON Fortran code and its ICON4Py / GT4Py translation. It serves two purposes:

1. **Human reference**: Look up how a specific Fortran variable, loop nest, or subroutine maps to ICON4Py.
2. **Translation guide**: Understand the systematic patterns for translating new Fortran modules to GT4Py.

The mappings are derived from the two modules that have been manually translated so far:

| ICON Fortran module | ICON4Py Python package |
|---|---|
| `src/atm_dyn_iconam/mo_solve_nonhydro.f90` | `model/atmosphere/dycore` |
| `src/atm_dyn_iconam/mo_nh_diffusion.f90` | `model/atmosphere/diffusion` |

---

## Table of Contents

- [Part 1: Translation Guide — Fortran to GT4Py](#part-1-translation-guide--fortran-to-gt4py)
  - [1.1 Overall Architecture](#11-overall-architecture)
  - [1.2 Grid and Dimensions](#12-grid-and-dimensions)
  - [1.3 Field Types](#13-field-types)
  - [1.4 Translating Loop Nests to Stencils](#14-translating-loop-nests-to-stencils)
  - [1.5 Neighbor Access and Connectivity](#15-neighbor-access-and-connectivity)
  - [1.6 Vertical Offsets](#16-vertical-offsets)
  - [1.7 Conditionals and Branching](#17-conditionals-and-branching)
  - [1.8 Reductions (Sums over Neighbors)](#18-reductions-sums-over-neighbors)
  - [1.9 Scan Operations (Tridiagonal Solvers)](#19-scan-operations-tridiagonal-solvers)
  - [1.10 Precision Handling](#110-precision-handling)
  - [1.11 Domain Specification](#111-domain-specification)
  - [1.12 Program Composition and Fusing](#112-program-composition-and-fusing)
  - [1.13 Orchestration Layer](#113-orchestration-layer)
  - [1.14 Naming Conventions](#114-naming-conventions)
- [Part 2: Field Name Mapping — ICON Fortran to ICON4Py](#part-2-field-name-mapping--icon-fortran-to-icon4py)
  - [2.1 Prognostic Variables](#21-prognostic-variables)
  - [2.2 Diagnostic Variables (Dycore)](#22-diagnostic-variables-dycore)
  - [2.3 Metric State Variables (Dycore)](#23-metric-state-variables-dycore)
  - [2.4 Interpolation State Variables](#24-interpolation-state-variables)
  - [2.5 Prep-Advection Variables](#25-prep-advection-variables)
  - [2.6 Intermediate / Local Variables (Dycore)](#26-intermediate--local-variables-dycore)
  - [2.7 Diagnostic Variables (Diffusion)](#27-diagnostic-variables-diffusion)
  - [2.8 Metric State Variables (Diffusion)](#28-metric-state-variables-diffusion)
  - [2.9 Interpolation State Variables (Diffusion)](#29-interpolation-state-variables-diffusion)
- [Part 3: Loop Nest to Stencil Mapping](#part-3-loop-nest-to-stencil-mapping)
  - [3.1 Dycore — mo_solve_nonhydro.f90](#31-dycore--mo_solve_nonhydrof90)
  - [3.2 Diffusion — mo_nh_diffusion.f90](#32-diffusion--mo_nh_diffusionf90)

---

## Part 1: Translation Guide — Fortran to GT4Py

### 1.1 Overall Architecture

An ICON Fortran module (e.g. `mo_nh_diffusion.f90`) containing a main subroutine with many loop nests maps to ICON4Py as follows:

```
Fortran module (e.g. mo_nh_diffusion.f90)
  └── SUBROUTINE diffusion(...)
        ├── DO jb loop nest 1  ──►  stencils/stencil_name_1.py  (@field_operator + @program)
        ├── DO jb loop nest 2  ──►  stencils/stencil_name_2.py  (@field_operator + @program)
        ├── ...
        └── DO jb loop nest N  ──►  stencils/stencil_name_N.py  (@field_operator + @program)

Maps to ICON4Py:
  └── Python package (e.g. model/atmosphere/diffusion/)
        ├── stencils/            # individual @field_operator / @program per loop nest
        │   ├── stencil_name_1.py
        │   ├── stencil_name_2.py
        │   └── ...
        ├── diffusion.py         # orchestration: class Diffusion with run() method
        ├── diffusion_states.py  # dataclass definitions for state containers
        └── diffusion_utils.py   # helper computations
```

**Key principle**: Each independent loop nest (a `DO jb` block loop with inner `DO jk`/`DO jc`/`DO je` loops) in Fortran typically maps to **one `@field_operator` + `@program` pair** in GT4Py. Multiple adjacent loop nests that are tightly coupled may be fused into a single larger program.

### 1.2 Grid and Dimensions

ICON uses an unstructured triangular grid. The three primary horizontal entity types and their ICON4Py dimensions are:

| Fortran entity | Loop variable | ICON4Py Dimension | Description |
|---|---|---|---|
| Cells | `jc` | `dims.CellDim` | Triangle cell centers |
| Edges | `je` | `dims.EdgeDim` | Triangle edges |
| Vertices | `jv` | `dims.VertexDim` | Triangle vertices |
| Vertical levels | `jk` | `dims.KDim` | Full model levels |
| Half levels | `jk` (nlevp1) | `dims.KDim` (extended by 1) | Interface levels between full levels |

In Fortran, fields are stored in blocked layout: `field(nproma, nlev, nblks)` where `nproma` is the block size and `nblks` the number of blocks. The `DO jb` loop iterates over blocks, and inner loops over `jc`/`je` iterate within a block. In GT4Py, the blocking is abstracted away — fields are simply `Field[Dims[CellDim, KDim], float]` and the framework handles parallelization.

### 1.3 Field Types

ICON4Py defines field type aliases in `icon4py.model.common.field_type_aliases`:

| Alias | GT4Py type | Fortran equivalent |
|---|---|---|
| `fa.CellField[T]` | `Field[Dims[CellDim], T]` | `field(nproma, nblks_c)` |
| `fa.EdgeField[T]` | `Field[Dims[EdgeDim], T]` | `field(nproma, nblks_e)` |
| `fa.VertexField[T]` | `Field[Dims[VertexDim], T]` | `field(nproma, nblks_v)` |
| `fa.CellKField[T]` | `Field[Dims[CellDim, KDim], T]` | `field(nproma, nlev, nblks_c)` |
| `fa.EdgeKField[T]` | `Field[Dims[EdgeDim, KDim], T]` | `field(nproma, nlev, nblks_e)` |
| `fa.VertexKField[T]` | `Field[Dims[VertexDim, KDim], T]` | `field(nproma, nlev, nblks_v)` |
| `fa.KField[T]` | `Field[Dims[KDim], T]` | `field(nlev)` (1D vertical profile) |

Precision types:
- `ta.wpfloat` / `wpfloat`: working precision (typically `float64`)
- `ta.vpfloat` / `vpfloat`: variable precision (can be `float32` in mixed precision mode, corresponds to Fortran `REAL(vp)`)

### 1.4 Translating Loop Nests to Stencils

A Fortran loop nest like:

```fortran
!$OMP DO PRIVATE(...) ICON_OMP_DEFAULT_SCHEDULE
DO jb = i_startblk, i_endblk
  CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, rl_start, rl_end)
  !$ACC PARALLEL DEFAULT(PRESENT) ASYNC(1)
  !$ACC LOOP GANG VECTOR COLLAPSE(2)
  DO jk = 1, nlev
    DO jc = i_startidx, i_endidx
      result_field(jc,jk,jb) = input_a(jc,jk,jb) + scalar * input_b(jc,jk,jb)
    ENDDO
  ENDDO
  !$ACC END PARALLEL
ENDDO
```

translates to:

```python
@gtx.field_operator
def _compute_result(
    input_a: fa.CellKField[wpfloat],
    input_b: fa.CellKField[wpfloat],
    scalar: wpfloat,
) -> fa.CellKField[wpfloat]:
    return input_a + scalar * input_b


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_result(
    input_a: fa.CellKField[wpfloat],
    input_b: fa.CellKField[wpfloat],
    scalar: wpfloat,
    result_field: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_result(
        input_a, input_b, scalar,
        out=result_field,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
```

**Key rules:**
- The `@field_operator` (prefixed with `_`) contains the pure computation. It takes input fields and scalars, returns result fields.
- The `@program` wraps the field_operator, adding the output field as parameter and specifying the execution domain.
- The Fortran block loop (`DO jb`) and index extraction (`get_indices_c/e/v`) are replaced by the `domain` specification with `horizontal_start`/`horizontal_end`.
- The inner loops (`DO jk`, `DO jc/je`) are implicit — GT4Py applies the computation to all points in the domain.
- `!$OMP`, `!$ACC` directives are not needed — GT4Py handles parallelization.

### 1.5 Neighbor Access and Connectivity

ICON Fortran accesses neighbors through index arrays:

```fortran
! Access cell neighbors of an edge
icidx => p_patch%edges%cell_idx
icblk => p_patch%edges%cell_blk
value = cell_field(icidx(je,jb,1), jk, icblk(je,jb,1))  ! first neighbor cell
value = cell_field(icidx(je,jb,2), jk, icblk(je,jb,2))  ! second neighbor cell
```

In GT4Py, this becomes a connectivity offset:

```python
from icon4py.model.common.dimension import E2C, E2CDim

# Inside a field_operator operating on EdgeDim:
cell_value_0 = cell_field(E2C[0])  # first neighbor cell of edge
cell_value_1 = cell_field(E2C[1])  # second neighbor cell of edge
```

Available connectivity offsets and their Fortran equivalents:

| GT4Py Offset | Fortran index arrays | Description |
|---|---|---|
| `E2C` | `p_patch%edges%cell_idx/blk` | Edge → neighboring cells (2 neighbors) |
| `E2V` | `p_patch%edges%vertex_idx/blk` | Edge → neighboring vertices (2 neighbors) |
| `C2E` | `p_patch%cells%edge_idx/blk` | Cell → neighboring edges (3 neighbors) |
| `C2V` | `p_patch%cells%vertex_idx/blk` | Cell → neighboring vertices (3 neighbors) |
| `V2E` | `p_patch%verts%edge_idx/blk` | Vertex → neighboring edges (6 neighbors) |
| `V2C` | `p_patch%verts%cell_idx/blk` | Vertex → neighboring cells (3 neighbors) |
| `E2C2E` | Diamond edges | Edge → edges sharing a cell (4 neighbors) |
| `E2C2V` | `p_patch%edges%vertex_idx` (diamond) | Edge → vertices of diamond (4 neighbors) |
| `C2E2C` | `p_patch%cells%neighbor_idx/blk` | Cell → neighbor cells (3 neighbors) |
| `C2E2CO` | Cell + its neighbors (with self) | Cell → self + neighbor cells (4 entries) |
| `E2C2EO` | Edge + diamond edges (with self) | Edge → self + diamond edges (5 entries) |
| `Koff` | `jk+1`, `jk-1` | Vertical offset by integer |

### 1.6 Vertical Offsets

Fortran vertical neighbor access:

```fortran
field(jc, jk-1, jb)   ! level above
field(jc, jk+1, jb)   ! level below
```

GT4Py equivalent using `Koff`:

```python
from icon4py.model.common.dimension import Koff

field(Koff[-1])   # level above (jk-1)
field(Koff[1])    # level below (jk+1)
```

### 1.7 Conditionals and Branching

Fortran `IF` inside loops:

```fortran
IF (condition(jc,jk,jb)) THEN
  result(jc,jk,jb) = value_a
ELSE
  result(jc,jk,jb) = value_b
ENDIF
```

GT4Py equivalent:

```python
from gt4py.next import where
result = where(condition, value_a, value_b)
```

For conditionals that depend on the position in the horizontal domain (e.g. distinguishing boundary vs. interior), GT4Py uses `concat_where`:

```python
from gt4py.next.experimental import concat_where

result = concat_where(
    dims.EdgeDim >= boundary_index,
    interior_computation(...),
    boundary_computation(...),
)
```

### 1.8 Reductions (Sums over Neighbors)

Fortran sums over neighbors:

```fortran
! Sum over cell-to-edge neighbors
result(jc,jk,jb) = 0
DO je = 1, 3
  result(jc,jk,jb) = result(jc,jk,jb) + &
    coeff(jc,je,jb) * edge_field(ieidx(jc,jb,je), jk, ieblk(jc,jb,je))
ENDDO
```

GT4Py equivalent:

```python
from gt4py.next import neighbor_sum
from icon4py.model.common.dimension import C2E, C2EDim

result = neighbor_sum(coeff * edge_field(C2E), axis=C2EDim)
```

### 1.9 Scan Operations (Tridiagonal Solvers)

Fortran sequential vertical loops (e.g. tridiagonal solvers):

```fortran
! Forward sweep
DO jk = 2, nlev
  z_q(jc,jk) = -z_c / (z_b + z_a * z_q(jc,jk-1))
  w(jc,jk) = (d(jc,jk) - z_a * w(jc,jk-1)) / (z_b + z_a * z_q(jc,jk-1))
ENDDO
```

GT4Py equivalent using `@scan_operator`:

```python
@gtx.scan_operator(axis=dims.KDim, forward=True, init=(vpfloat("0.0"), 0.0))
def tridiagonal_forward_sweep(
    state_kminus1: tuple[vpfloat, float],
    a: vpfloat, b: vpfloat, c: vpfloat, d: wpfloat,
):
    c_prev = state_kminus1[0]
    d_prev = state_kminus1[1]
    norm = vpfloat("1.0") / (b + a * c_prev)
    c_new = -c * norm
    d_new = (d - astype(a, wpfloat) * d_prev) * astype(norm, wpfloat)
    return c_new, d_new
```

The `@scan_operator` replaces sequential vertical DO loops where each level depends on the result from the previous level. The `init` parameter provides the boundary condition at the top (for `forward=True`) or bottom (for `forward=False`).

### 1.10 Precision Handling

Fortran uses `REAL(wp)` (working precision, double) and `REAL(vp)` (variable precision, single in mixed-precision mode).

In GT4Py, use `astype()` for explicit precision conversions:

```python
from gt4py.next import astype
from icon4py.model.common.type_alias import vpfloat, wpfloat

# Convert from vpfloat to wpfloat for computation
value_wp = astype(value_vp, wpfloat)

# Convert result back to vpfloat for storage
result_vp = astype(result_wp, vpfloat)
```

Multiple fields can be converted at once:

```python
a_wp, b_wp, c_wp = astype((a_vp, b_vp, c_vp), wpfloat)
```

### 1.11 Domain Specification

Fortran uses `rl_start`/`rl_end` to determine block ranges, then `get_indices_c/e/v` to get index ranges within blocks. Different halo regions are accessed by different `rl_start`/`rl_end` values.

In GT4Py, the domain is specified in the `@program` call:

```python
_stencil(
    ...,
    out=output_field,
    domain={
        dims.CellDim: (horizontal_start, horizontal_end),  # replaces jb/jc loops + rl_start/rl_end
        dims.KDim: (vertical_start, vertical_end),          # replaces jk loop bounds
    },
)
```

The `horizontal_start` and `horizontal_end` are computed from the grid's zone definitions (lateral boundary, nudging zone, halo, interior) in the orchestration layer.

### 1.12 Program Composition and Fusing

Multiple related loop nests can be fused into a single `@program` that calls several `@field_operator`s:

```python
@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_cell_diagnostics_for_dycore(
    # all input fields ...
    # all output fields ...
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _stencil_a(..., out=output_a, domain={...})
    _stencil_b(..., out=output_b, domain={...})
    _stencil_c(..., out=output_c, domain={...})
```

This is used when multiple loop nests operate on the same domain and have data dependencies. For example, `compute_cell_diagnostics_for_dycore` fuses stencils 1-13 of the nonhydrostatic solver.

### 1.13 Orchestration Layer

The orchestration layer is a Python class (e.g. `Diffusion`, `SolveNonhydro`) that:

1. **Holds configuration and pre-computed state** (metric fields, interpolation coefficients, grid parameters).
2. **Implements the `run()` method** that calls the GT4Py programs in the correct order, mirroring the Fortran subroutine's control flow.
3. **Handles conditional logic** (predictor/corrector steps, initialization vs. runtime) that cannot be expressed in GT4Py stencils.

```python
class Diffusion:
    def __init__(self, ...):
        # Pre-compute derived parameters, allocate temporaries
        ...

    def run(self, prognostic_state, diagnostic_state, dtime, ...):
        # Step 1: Compute u,v at vertices
        self._rbf_vec_interpol_vertex(...)
        # Step 2: Calculate Smagorinsky coefficients
        self._calculate_nabla2_and_smag_coefficients_for_vn(...)
        # Step 3: Apply diffusion to vn
        self._apply_diffusion_to_vn(...)
        # ...etc...
```

### 1.14 Naming Conventions

See also: `docs/stencil_naming_convention.md` and `docs/variable_naming_convention.md`.

**Stencil names:**
- `@field_operator`: prefixed with underscore, e.g. `_compute_tangential_wind`
- `@program`: same without underscore, e.g. `compute_tangential_wind`
- Start with a verb: `compute`, `apply`, `add`, `interpolate`, `extrapolate`, `solve`, `update`, `accumulate`, `set`, `init`, `copy`

**Variable names:**
- Descriptive English names replacing terse Fortran abbreviations
- Location suffixes when needed: `_at_cells`, `_at_edges`, `_at_vertices`
- Level suffixes when ambiguous: `_on_model_levels`, `_on_half_levels`
- Prognostic variables (`rho`, `w`, `vn`, `exner`, `theta_v`) keep their short names

---

## Part 2: Field Name Mapping — ICON Fortran to ICON4Py

### 2.1 Prognostic Variables

Defined in `icon4py.model.common.states.prognostic_state.PrognosticState` (corresponds to `t_nh_prog`).

| Fortran name | ICON4Py name | Type | Description |
|---|---|---|---|
| `p_nh%prog(nnow/nnew)%rho` | `rho` | `CellKField[wpfloat]` | Air density [kg/m³] |
| `p_nh%prog(nnow/nnew)%w` | `w` | `CellKField[wpfloat]` | Vertical wind speed [m/s], nlevp1 levels |
| `p_nh%prog(nnow/nnew)%vn` | `vn` | `EdgeKField[wpfloat]` | Normal wind component at edges [m/s] |
| `p_nh%prog(nnow/nnew)%exner` | `exner` | `CellKField[wpfloat]` | Exner function [-] |
| `p_nh%prog(nnow/nnew)%theta_v` | `theta_v` | `CellKField[wpfloat]` | Virtual potential temperature [K] |

### 2.2 Diagnostic Variables (Dycore)

Defined in `icon4py.model.atmosphere.dycore.dycore_states.DiagnosticStateNonHydro` (corresponds to `p_nh%diag`).

| Fortran name | ICON4Py name | Type | Description |
|---|---|---|---|
| `p_nh%diag%vt` | `tangential_wind` | `EdgeKField[vpfloat]` | Tangential wind at edges |
| `p_nh%diag%vn_ie` | `vn_on_half_levels` | `EdgeKField[vpfloat]` | Normal wind interpolated to half levels |
| `p_nh%diag%w_concorr_c` | `contravariant_correction_at_cells_on_half_levels` | `CellKField[vpfloat]` | Contravariant vertical correction at cells |
| `p_nh%diag%theta_v_ic` | `theta_v_at_cells_on_half_levels` | `CellKField[wpfloat]` | Theta_v interpolated to half levels |
| `p_nh%diag%exner_pr` | `perturbed_exner_at_cells_on_model_levels` | `CellKField[wpfloat]` | Perturbation Exner pressure |
| `p_nh%diag%rho_ic` | `rho_at_cells_on_half_levels` | `CellKField[wpfloat]` | Density interpolated to half levels |
| `p_nh%diag%ddt_exner_phy` | `exner_tendency_due_to_slow_physics` | `CellKField[vpfloat]` | Physics tendency of Exner pressure |
| `p_nh%diag%mass_fl_e` | `mass_flux_at_edges_on_model_levels` | `EdgeKField[wpfloat]` | Mass flux at edges |
| `p_nh%diag%ddt_vn_phy` | `normal_wind_tendency_due_to_slow_physics_process` | `EdgeKField[vpfloat]` | Physics tendency of normal wind |
| `p_nh%diag%ddt_vn_apc_pc` | `normal_wind_advective_tendency` | `PredictorCorrectorPair[EdgeKField[vpfloat]]` | Advective+Coriolis tendency of vn |
| `p_nh%diag%ddt_w_adv_pc` | `vertical_wind_advective_tendency` | `PredictorCorrectorPair[CellKField[vpfloat]]` | Advective tendency of w |
| `p_nh%diag%max_vcfl_dyn` | `max_vertical_cfl` | `ScalarLikeArray[wpfloat]` | Maximum vertical CFL number |
| `p_nh%diag%vn_incr` | `normal_wind_iau_increment` | `EdgeKField[vpfloat]` | IAU increment for normal wind |
| `p_nh%diag%rho_incr` | `rho_iau_increment` | `CellKField[vpfloat]` | IAU increment for density |
| `p_nh%diag%exner_incr` | `exner_iau_increment` | `CellKField[vpfloat]` | IAU increment for Exner pressure |
| `p_nh%diag%exner_dyn_incr` | `exner_dynamical_increment` | `CellKField[vpfloat]` | Dynamics increment for Exner pressure |
| `p_nh%diag%grf_tend_rho` | `grf_tend_rho` | `CellKField[wpfloat]` | Grid refinement tendency for rho |
| `p_nh%diag%grf_tend_thv` | `grf_tend_thv` | `CellKField[wpfloat]` | Grid refinement tendency for theta_v |
| `p_nh%diag%grf_tend_w` | `grf_tend_w` | `CellKField[wpfloat]` | Grid refinement tendency for w |
| `p_nh%diag%grf_tend_vn` | `grf_tend_vn` | `EdgeKField[wpfloat]` | Grid refinement tendency for vn |

### 2.3 Metric State Variables (Dycore)

Defined in `icon4py.model.atmosphere.dycore.dycore_states.MetricStateNonHydro` (corresponds to `p_nh%metrics`).

| Fortran name | ICON4Py name | Type | Description |
|---|---|---|---|
| `p_nh%metrics%rayleigh_w` | `rayleigh_w` | `KField[wpfloat]` | Rayleigh damping coefficient profile |
| `p_nh%metrics%wgtfac_c` | `wgtfac_c` | `CellKField[vpfloat]` | Weighting factor for interpolation full→half levels at cells |
| `p_nh%metrics%wgtfacq_c` | `wgtfacq_c` | `CellKField[vpfloat]` | Quadratic weighting factor at cells |
| `p_nh%metrics%wgtfac_e` | `wgtfac_e` | `EdgeKField[vpfloat]` | Weighting factor at edges |
| `p_nh%metrics%wgtfacq_e` | `wgtfacq_e` | `EdgeKField[vpfloat]` | Quadratic weighting factor at edges |
| `p_nh%metrics%exner_exfac` | `time_extrapolation_parameter_for_exner` | `CellKField[vpfloat]` | Exner pressure temporal extrapolation factor |
| `p_nh%metrics%exner_ref_mc` | `reference_exner_at_cells_on_model_levels` | `CellKField[vpfloat]` | Reference Exner pressure at cells |
| `p_nh%metrics%rho_ref_mc` | `reference_rho_at_cells_on_model_levels` | `CellKField[vpfloat]` | Reference density at cells |
| `p_nh%metrics%theta_ref_mc` | `reference_theta_at_cells_on_model_levels` | `CellKField[vpfloat]` | Reference theta at cells |
| `p_nh%metrics%rho_ref_me` | `reference_rho_at_edges_on_model_levels` | `EdgeKField[vpfloat]` | Reference density at edges |
| `p_nh%metrics%theta_ref_me` | `reference_theta_at_edges_on_model_levels` | `EdgeKField[vpfloat]` | Reference theta at edges |
| `p_nh%metrics%theta_ref_ic` | `reference_theta_at_cells_on_half_levels` | `CellKField[vpfloat]` | Reference theta at half levels |
| `p_nh%metrics%d_exner_dz_ref_ic` | `ddz_of_reference_exner_at_cells_on_half_levels` | `CellKField[vpfloat]` | Vertical derivative of reference Exner |
| `p_nh%metrics%ddqz_z_half` | `ddqz_z_half` | `CellKField[vpfloat]` | Layer thickness at half levels |
| `p_nh%metrics%inv_ddqz_z_full` | `inv_ddqz_z_full` | `CellKField[vpfloat]` | Inverse layer thickness at full levels |
| `p_nh%metrics%d2dexdz2_fac1_mc` | `d2dexdz2_fac1_mc` | `CellKField[vpfloat]` | Factor for 2nd vertical derivative of Exner |
| `p_nh%metrics%d2dexdz2_fac2_mc` | `d2dexdz2_fac2_mc` | `CellKField[vpfloat]` | Factor for 2nd vertical derivative of Exner |
| `p_nh%metrics%ddxn_z_full` | `ddxn_z_full` | `EdgeKField[vpfloat]` | Normal derivative of terrain height |
| `p_nh%metrics%ddxt_z_full` | `ddxt_z_full` | `EdgeKField[vpfloat]` | Tangential derivative of terrain height |
| `p_nh%metrics%ddqz_z_full_e` | `ddqz_z_full_e` | `EdgeKField[vpfloat]` | Layer thickness at edges |
| `p_nh%metrics%vwind_expl_wgt` | `exner_w_explicit_weight_parameter` | `CellField[wpfloat]` | Explicit weight for vertically implicit solver |
| `p_nh%metrics%vwind_impl_wgt` | `exner_w_implicit_weight_parameter` | `CellField[wpfloat]` | Implicit weight for vertically implicit solver |
| `p_nh%metrics%hmask_dd3d` | `horizontal_mask_for_3d_divdamp` | `EdgeField[wpfloat]` | Mask for 3D divergence damping |
| `p_nh%metrics%scalfac_dd3d` | `scaling_factor_for_3d_divdamp` | `KField[wpfloat]` | Vertical profile for 3D div. damping |
| `p_nh%metrics%coeff1_dwdz` | `coeff1_dwdz` | `CellKField[vpfloat]` | Coefficient for dw/dz computation |
| `p_nh%metrics%coeff2_dwdz` | `coeff2_dwdz` | `CellKField[vpfloat]` | Coefficient for dw/dz computation |
| `p_nh%metrics%coeff_gradekin` | `coeff_gradekin` | `Field[EdgeDim, E2CDim, vpfloat]` | Coefficient for gradient of kinetic energy |
| `p_nh%metrics%bdy_halo_c` | `bdy_halo_c` | `CellField[bool]` | Boundary halo mask for cells |
| `p_nh%metrics%mask_prog_halo_c` | `mask_prog_halo_c` | `CellKField[bool]` | Prognostic halo mask at cells |
| `p_nh%metrics%pg_edgeidx_dsl` | `pg_edgeidx_dsl` | `EdgeKField[bool]` | Edge index for pressure gradient |
| `p_nh%metrics%pg_exdist` | `pg_exdist` | `EdgeKField[vpfloat]` | Extrapolation distance for pressure gradient |
| `p_nh%metrics%vertoffset_gradp` | `vertoffset_gradp` | `Field[EdgeDim, E2CDim, KDim, int32]` | Vertical offset for pressure gradient |
| `p_nh%metrics%zdiff_gradp` | `zdiff_gradp` | `Field[EdgeDim, E2CDim, KDim, vpfloat]` | Height difference for pressure gradient |

### 2.4 Interpolation State Variables

Defined in `icon4py.model.atmosphere.dycore.dycore_states.InterpolationState` (corresponds to `p_int`).

| Fortran name | ICON4Py name | Type | Description |
|---|---|---|---|
| `p_int%e_bln_c_s` | `e_bln_c_s` | `Field[CellDim, C2EDim, wpfloat]` | Bilinear interpolation edge→cell |
| `p_int%rbf_coeff_1` | `rbf_coeff_1` | `Field[VertexDim, V2EDim, wpfloat]` | RBF vector coefficient (component 1) |
| `p_int%rbf_coeff_2` | `rbf_coeff_2` | `Field[VertexDim, V2EDim, wpfloat]` | RBF vector coefficient (component 2) |
| `p_int%geofac_div` | `geofac_div` | `Field[CellDim, C2EDim, wpfloat]` | Geometric factor for divergence |
| `p_int%geofac_n2s` | `geofac_n2s` | `Field[CellDim, C2E2CODim, wpfloat]` | Geometric factor for nabla² scalar |
| `p_int%geofac_grg_x/y` | `geofac_grg_x` / `geofac_grg_y` | `Field[CellDim, C2E2CODim, wpfloat]` | Green-Gauss gradient factors |
| `p_int%nudgecoeff_e` | `nudgecoeff_e` | `EdgeField[wpfloat]` | Nudging coefficients at edges |
| `p_int%c_lin_e` | `c_lin_e` | `Field[EdgeDim, E2CDim, wpfloat]` | Linear interpolation cell→edge |
| `p_int%geofac_grdiv` | `geofac_grdiv` | `Field[EdgeDim, E2C2EODim, wpfloat]` | Geometric factor for grad(div) |
| `p_int%rbf_vec_coeff_e` | `rbf_vec_coeff_e` | `Field[EdgeDim, E2C2EDim, wpfloat]` | RBF vector coefficient at edges |
| `p_int%c_intp` | `c_intp` | `Field[VertexDim, V2CDim, wpfloat]` | Interpolation cell→vertex |
| `p_int%geofac_rot` | `geofac_rot` | `Field[VertexDim, V2EDim, wpfloat]` | Geometric factor for curl/rotation |
| `p_int%pos_on_tplane_e_1/2` | `pos_on_tplane_e_1` / `pos_on_tplane_e_2` | `Field[EdgeDim, E2CDim, wpfloat]` | Position on tangent plane (for backward trajectory) |
| `p_int%e_flx_avg` | `e_flx_avg` | `Field[EdgeDim, E2C2EODim, wpfloat]` | Averaging coefficients for edge fluxes |

### 2.5 Prep-Advection Variables

Defined in `icon4py.model.atmosphere.dycore.dycore_states.PrepAdvection` (corresponds to `prep_adv`).

| Fortran name | ICON4Py name | Type | Description |
|---|---|---|---|
| `prep_adv%vn_traj` | `vn_traj` | `EdgeKField[wpfloat]` | Trajectory-averaged normal wind |
| `prep_adv%mass_flx_me` | `mass_flx_me` | `EdgeKField[wpfloat]` | Mass flux at edges for advection |
| `prep_adv%mass_flx_ic` | `dynamical_vertical_mass_flux_at_cells_on_half_levels` | `CellKField[wpfloat]` | Vertical mass flux at cells |
| `prep_adv%vol_flx_ic` | `dynamical_vertical_volumetric_flux_at_cells_on_half_levels` | `CellKField[wpfloat]` | Vertical volume flux at cells |

### 2.6 Intermediate / Local Variables (Dycore)

Defined in `icon4py.model.atmosphere.dycore.solve_nonhydro.IntermediateFields`. These are local (`z_`-prefixed) arrays in Fortran.

| Fortran name | ICON4Py name | Type | Description |
|---|---|---|---|
| `z_gradh_exner` | `horizontal_pressure_gradient` | `EdgeKField[vpfloat]` | Horizontal gradient of Exner pressure |
| `z_rho_e` | `rho_at_edges_on_model_levels` | `EdgeKField[wpfloat]` | Density interpolated to edges |
| `z_theta_v_e` | `theta_v_at_edges_on_model_levels` | `EdgeKField[wpfloat]` | Theta_v interpolated to edges |
| `z_kin_hor_e` | `horizontal_kinetic_energy_at_edges_on_model_levels` | `EdgeKField[vpfloat]` | Horizontal kinetic energy at edges |
| `z_vt_ie` | `tangential_wind_on_half_levels` | `EdgeKField[vpfloat]` | Tangential wind at half levels |
| `z_graddiv_vn` | `horizontal_gradient_of_normal_wind_divergence` | `EdgeKField[vpfloat]` | Grad(div(vn)) |
| `z_dwdz_dd` | `dwdz_at_cells_on_model_levels` | `CellKField[vpfloat]` | dw/dz for divergence damping |

### 2.7 Diagnostic Variables (Diffusion)

Defined in `icon4py.model.atmosphere.diffusion.diffusion_states.DiffusionDiagnosticState`.

| Fortran name | ICON4Py name | Type | Description |
|---|---|---|---|
| `p_nh_diag%hdef_ic` | `hdef_ic` | `CellKField[float]` | Horizontal wind deformation at half levels |
| `p_nh_diag%div_ic` | `div_ic` | `CellKField[float]` | Divergence at half levels |
| `p_nh_diag%dwdx` | `dwdx` | `CellKField[float]` | Zonal gradient of w |
| `p_nh_diag%dwdy` | `dwdy` | `CellKField[float]` | Meridional gradient of w |

### 2.8 Metric State Variables (Diffusion)

Defined in `icon4py.model.atmosphere.diffusion.diffusion_states.DiffusionMetricState`.

| Fortran name | ICON4Py name | Type | Description |
|---|---|---|---|
| `p_nh_metrics%theta_ref_mc` | `theta_ref_mc` | `CellKField[float]` | Reference theta at cells |
| `p_nh_metrics%wgtfac_c` | `wgtfac_c` | `CellKField[float]` | Weighting factor full→half at cells |
| `p_nh_metrics%mask_hdiff` | `mask_hdiff` | `CellKField[bool]` | Mask for horizontal diffusion |
| `p_nh_metrics%zd_vertoffset` | `zd_vertoffset` | `Field[CellDim, C2E2CDim, KDim, int32]` | Vertical offset for truly horizontal diffusion |
| `p_nh_metrics%zd_diffcoef` | `zd_diffcoef` | `CellKField[float]` | Diffusion coefficient for truly horizontal diffusion |
| `p_nh_metrics%zd_intcoef` | `zd_intcoef` | `Field[CellDim, C2E2CDim, KDim, float]` | Interpolation coefficient for truly horizontal diffusion |

### 2.9 Interpolation State Variables (Diffusion)

Defined in `icon4py.model.atmosphere.diffusion.diffusion_states.DiffusionInterpolationState`.

| Fortran name | ICON4Py name | Type | Description |
|---|---|---|---|
| `p_int%e_bln_c_s` | `e_bln_c_s` | `Field[CellDim, C2EDim, float]` | Bilinear interpolation edge→cell |
| `p_int%rbf_coeff_1` | `rbf_coeff_1` | `Field[VertexDim, V2EDim, float]` | RBF vector coefficient 1 |
| `p_int%rbf_coeff_2` | `rbf_coeff_2` | `Field[VertexDim, V2EDim, float]` | RBF vector coefficient 2 |
| `p_int%geofac_div` | `geofac_div` | `Field[CellDim, C2EDim, float]` | Geometric factor for divergence |
| `p_int%geofac_n2s` | `geofac_n2s` | `Field[CellDim, C2E2CODim, float]` | Geometric factor for nabla² scalar |
| `p_int%geofac_grg_x/y` | `geofac_grg_x` / `geofac_grg_y` | `Field[CellDim, C2E2CODim, float]` | Green-Gauss gradient factors |
| `p_int%nudgecoeff_e` | `nudgecoeff_e` | `EdgeField[float]` | Nudging coefficients |

---

## Part 3: Loop Nest to Stencil Mapping

### 3.1 Dycore — mo_solve_nonhydro.f90

The `solve_nh` subroutine contains a predictor-corrector time stepping loop (`DO istep = 1, 2`). Each step executes a sequence of loop nests that have been translated to GT4Py stencils. The ICON4Py orchestration lives in `solve_nonhydro.py` (class `SolveNonhydro`).

Below, the original Fortran stencil numbering is preserved. The "Formerly known as" column shows the internal stencil number used during development.

#### Cell-based diagnostics (predictor/corrector)

| Stencil # | GT4Py stencil | Description |
|---|---|---|
| 02 | `extrapolate_temporally_exner_pressure` | Temporal extrapolation of perturbation Exner pressure |
| 04 | `interpolate_to_surface` | Interpolate theta_v to surface |
| 07, 13 | `compute_perturbation_of_rho_and_theta` | Perturbation of rho and theta (subtract reference) |
| 08 | `compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers` | Perturbation + rho at half levels |
| 09 | `compute_virtual_potential_temperatures_and_pressure_gradient` | Compute theta_v primes and pressure gradient prep |
| 10 | `compute_rho_virtual_potential_temperatures_and_pressure_gradient` | Rho-weighted theta_v and pressure gradient |
| 11 | `set_theta_v_prime_ic_at_lower_boundary` | Set theta_v perturbation at lower boundary |
| 12 | `compute_approx_of_2nd_vertical_derivative_of_exner` | Approximate d²exner/dz² |
| 1-13 (fused) | `compute_cell_diagnostics_for_dycore` | Fused program for all cell diagnostics |

#### Edge-based diagnostics and vn update

| Stencil # | GT4Py stencil | Description |
|---|---|---|
| 14, 15, 33 | `init_two_edge_kdim_fields_with_zero_wp` | Zero-initialize edge fields |
| 16 | `compute_horizontal_advection_of_rho_and_theta` | Horizontal advection with backward trajectory |
| 17 | `add_vertical_wind_derivative_to_divergence_damping` | Add dw/dz to divergence damping |
| 18 | `compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates` | Pressure gradient (flat terrain) |
| 19 | `compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates` | Pressure gradient (terrain-following) |
| 20 | `compute_horizontal_gradient_of_exner_pressure_for_multiple_levels` | Pressure gradient (multiple levels) |
| 21 | `compute_hydrostatic_correction_term` | Hydrostatic correction for pressure gradient |
| 22 | `apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure` | Apply hydrostatic correction |
| 23 | `add_temporal_tendencies_to_vn_by_interpolating_between_time_levels` | Temporal interpolation of tendencies |
| 24 | `add_temporal_tendencies_to_vn` | Add tendencies to vn |
| 25 | `compute_graddiv2_of_vn` | 2nd order grad(div) of vn |
| 26 | `apply_2nd_order_divergence_damping` | Apply 2nd order divergence damping |
| 27 | `apply_weighted_2nd_and_4th_order_divergence_damping` | Combined 2nd/4th order div. damping |
| 28 | `add_analysis_increments_to_vn` | Add IAU increments to vn |
| 15-28 (fused) | `compute_edge_diagnostics_for_dycore_and_update_vn` | Fused program for edge diagnostics |

#### Lateral boundary and averaging

| Stencil # | GT4Py stencil | Description |
|---|---|---|
| 29 | `compute_vn_on_lateral_boundary` | Interpolate vn at lateral boundary |
| 30 | `compute_avg_vn_and_graddiv_vn_and_vt` | Average vn, compute grad(div), tangential wind |
| 31 | `spatially_average_flux_or_velocity` | Spatial averaging of edge quantities |
| 32 | `compute_mass_flux` | Compute mass flux at edges |

#### Vertical solver

| Stencil # | GT4Py stencil | Description |
|---|---|---|
| 34 | `accumulate_prep_adv_fields` | Accumulate prep-advection fields |
| 35 | `compute_contravariant_correction` | Contravariant correction of vertical wind |
| 36 | `interpolate_vn_and_vt_to_ie_and_compute_ekin_on_edges` | Interpolate vn,vt to half levels + kinetic energy |
| 37 | `compute_horizontal_kinetic_energy` | Horizontal kinetic energy at edges |
| 38 | `extrapolate_at_top` | Extrapolation at model top |
| 39 | `compute_contravariant_correction_of_w` | Contravariant correction of w |
| 40 | `compute_contravariant_correction_of_w_for_lower_boundary` | Contravariant correction at lower boundary |
| 41 | `compute_divergence_of_fluxes_of_rho_and_theta` | Divergence of mass and theta fluxes |
| 42 | `compute_explicit_vertical_wind_from_advection_and_vertical_wind_density` | Explicit w from advection |
| 43 | `compute_explicit_vertical_wind_speed_and_vertical_wind_times_density` | Explicit w speed + w*rho |
| 44 | `compute_solver_coefficients_matrix` | Coefficients for tridiagonal solver |
| 47 | `set_lower_boundary_condition_for_w_and_contravariant_correction` | Lower boundary condition |
| 48, 49 | `compute_explicit_part_for_rho_and_exner` | Explicit part of rho and exner update |
| 50 | `add_analysis_increments_from_data_assimilation` | Add IAU increments (rho, exner) |
| 52 | `solve_tridiagonal_matrix_for_w_forward_sweep` | Tridiagonal forward sweep for w |
| 53 | `solve_tridiagonal_matrix_for_w_back_substitution` | Tridiagonal back substitution for w |
| 54 | `apply_rayleigh_damping_mechanism` | Rayleigh damping of w |
| 55 | `compute_results_for_thermodynamic_variables` | Update rho, exner, theta from solver |
| 56, 63 | `compute_dwdz_for_divergence_damping` | dw/dz for 3D divergence damping |
| 57, 64 | `init_cell_kdim_field_with_zero_wp` | Zero-initialize cell field |
| 58 | `update_mass_volume_flux` | Update mass and volume flux |
| 59 | `copy_cell_kdim_field_to_vp` | Copy cell field with precision conversion |
| 60 | `update_dynamical_exner_time_increment` | Update exner dynamics increment |
| 61 | `update_density_exner_wind` | Update density, exner, and wind |
| 62 | `update_wind` | Update vertical wind |
| 65 | `update_mass_flux_weighted` | Mass-flux weighted update |
| 66 | `compute_theta_and_exner` | Compute theta_v and exner |
| 67 | `compute_exner_from_rhotheta` | Compute exner from rho*theta |
| 68 | `update_theta_v` | Update theta_v |

#### 4th-order divergence damping

| Stencil # | GT4Py stencil | Description |
|---|---|---|
| 4th-order | `apply_4th_order_divergence_damping` | Apply nabla⁴ divergence damping |

#### Velocity advection (mo_velocity_advection)

| Stencil # | GT4Py stencil | Description |
|---|---|---|
| vel_adv 01 | `compute_tangential_wind` | Tangential wind from RBF interpolation |
| vel_adv 02 | `interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges` | vn at half levels + kinetic energy |
| vel_adv 03 | `interpolate_vt_to_interface_edges` | Tangential wind at half levels |
| vel_adv 04 | `compute_contravariant_correction` | Contravariant correction (shared with stencil 35) |
| vel_adv 05 | `compute_horizontal_kinetic_energy` | Horizontal kinetic energy (shared with stencil 37) |
| vel_adv 06 | `extrapolate_at_top` | Extrapolation at top (shared with stencil 38) |
| vel_adv 07 | `compute_horizontal_advection_term_for_vertical_velocity` | Horizontal advection of w |
| vel_adv 11 | `copy_cell_kdim_field_to_vp` | Copy field (shared with stencil 59) |
| vel_adv 14 | `compute_maximum_cfl_and_clip_contravariant_vertical_velocity` | CFL check and clipping |
| vel_adv 15 | `interpolate_contravariant_vertical_velocity_to_full_levels` | Contravariant w to full levels |
| vel_adv 16 | `compute_advective_vertical_wind_tendency` | Advective tendency of w |
| vel_adv 17 | `add_interpolated_horizontal_advection_of_w` | Add horizontal advection of w |
| vel_adv 18 | `add_extra_diffusion_for_w_con_approaching_cfl` | Extra diffusion near CFL limit for w |
| vel_adv 20 | `add_extra_diffusion_for_normal_wind_tendency_approaching_cfl` | Extra diffusion near CFL limit for vn |
| vel_adv 1-7 (fused) | `compute_diagnostics_from_normal_wind` | Fused edge and vertex diagnostics |
| vel_adv 19-20 (fused) | `compute_advection_in_horizontal_momentum_equation` | Fused horizontal momentum advection |

### 3.2 Diffusion — mo_nh_diffusion.f90

The `diffusion` subroutine applies horizontal diffusion to the prognostic variables. The ICON4Py orchestration lives in `diffusion.py` (class `Diffusion`).

#### Velocity reconstruction at vertices and cells

| Step | GT4Py stencil | Description |
|---|---|---|
| RBF interpolation | `mo_intp_rbf_rbf_vec_interpol_vertex` | Reconstruct u,v at vertices from vn via RBF |

#### Smagorinsky diffusion coefficients

| Step | GT4Py stencil | Description |
|---|---|---|
| Smag + nabla² | `calculate_nabla2_and_smag_coefficients_for_vn` | Compute Smagorinsky coefficient and nabla²(vn) |

#### Enhanced diffusion for cold pools

| Step | GT4Py stencil | Description |
|---|---|---|
| Cold pool enhancement | `calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools` | Enhanced diffusion in topographic depressions |

#### Diffusion of vn (normal wind)

| Step | GT4Py stencil | Description |
|---|---|---|
| Apply to vn | `apply_diffusion_to_vn` | Apply nabla² + nabla⁴ + Smagorinsky diffusion to vn |
| Sub-stencils | `calculate_nabla4` | Compute nabla⁴(vn) from nabla²(vn) |
| | `apply_nabla2_and_nabla4_to_vn` | Apply combined nabla²+nabla⁴ (interior) |
| | `apply_nabla2_and_nabla4_global_to_vn` | Apply combined nabla²+nabla⁴ (global domain) |
| | `apply_nabla2_to_vn_in_lateral_boundary` | Boundary diffusion for vn |

#### Diffusion of w (vertical wind)

| Step | GT4Py stencil | Description |
|---|---|---|
| Apply to w | `apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence` | Fused: nabla²(w) + nabla⁴(w) + turbulence diagnostics |
| Sub-stencils | `calculate_nabla2_for_w` | Compute nabla²(w) |
| | `apply_nabla2_to_w` | Apply nabla²(w) damping |
| | `apply_nabla2_to_w_in_upper_damping_layer` | Extra nabla² damping in upper layers |
| | `calculate_horizontal_gradients_for_turbulence` | Compute dw/dx, dw/dy for turbulence |

#### Diffusion of theta (temperature)

| Step | GT4Py stencil | Description |
|---|---|---|
| Apply to theta+exner | `apply_diffusion_to_theta_and_exner` | Diffusion on theta_v and Exner pressure |
| Sub-stencils | `calculate_nabla2_for_theta` | Compute nabla²(theta) |
| | `calculate_nabla2_of_theta` | nabla²(theta) via Green-Gauss |
| | `update_theta_and_exner` | Update theta_v and recompute exner |
| | `truly_horizontal_diffusion_nabla_of_theta_over_steep_points` | Truly horizontal diffusion over steep terrain |

#### Turbulence diagnostics

| Step | GT4Py stencil | Description |
|---|---|---|
| Diagnostics | `calculate_diagnostic_quantities_for_turbulence` | Compute div_ic and hdef_ic for turbulence parameterization |
| | `calculate_diagnostics_for_turbulence` | Additional turbulence diagnostics |

---

## Appendix: Quick Reference for Common Translation Patterns

### Fortran → GT4Py Cheat Sheet

| Fortran pattern | GT4Py equivalent |
|---|---|
| `DO jb = ...; DO jk = ...; DO jc = ...` | `@field_operator` + `@program` with `domain={CellDim: ..., KDim: ...}` |
| `field(jc,jk,jb)` | `field` (implicit point-wise access) |
| `field(jc,jk+1,jb)` | `field(Koff[1])` |
| `field(jc,jk-1,jb)` | `field(Koff[-1])` |
| `field(icidx(je,jb,1),jk,icblk(je,jb,1))` | `field(E2C[0])` |
| `field(icidx(je,jb,2),jk,icblk(je,jb,2))` | `field(E2C[1])` |
| `SUM over neighbors` | `neighbor_sum(coeff * field(OFFSET), axis=OFFSETDim)` |
| `IF (cond) THEN a ELSE b` | `where(cond, a, b)` |
| `REAL(vp)` → `REAL(wp)` | `astype(field, wpfloat)` |
| Sequential vertical DO (tridiag) | `@scan_operator(axis=KDim, forward=True/False, init=...)` |
| `SQRT(x)` | `sqrt(x)` (from `gt4py.next`) |
| `MAX(a,b)` | `maximum(a, b)` (from `gt4py.next`) |
| `MIN(a,b)` | `minimum(a, b)` (from `gt4py.next`) |
| `1.0_wp` (literal) | `wpfloat("1.0")` |
| `0.0_vp` (literal) | `vpfloat("0.0")` |
| `broadcast(K_field, (CellDim, KDim))` | `broadcast(k_field, (dims.CellDim, dims.KDim))` |
