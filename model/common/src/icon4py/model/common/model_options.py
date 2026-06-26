# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import logging
import os
from collections.abc import Callable
from typing import Any

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
from gt4py.next import backend as gtx_backend
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations

from icon4py.model.common import model_backends


log = logging.getLogger(__name__)


def dict_values_to_list(d: dict[str, Any]) -> dict[str, list]:
    return {k: [v] for k, v in d.items()}


def get_dace_options(
    program_name: str, **backend_descriptor: Any
) -> model_backends.BackendDescriptor:
    is_rocm_device = backend_descriptor.get("device") == model_backends.DeviceType.ROCM
    optimization_args = backend_descriptor.get("optimization_args", {})
    optimization_hooks = optimization_args.get("optimization_hooks", {})
    if program_name in [
        "vertically_implicit_solver_at_corrector_step",
        "vertically_implicit_solver_at_predictor_step",
    ]:
        if gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowStep not in optimization_hooks:
            # Enable pass that removes access node (next_w) copies for vertically implicit solver programs
            optimization_hooks[gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowStep] = (
                lambda sdfg: sdfg.apply_transformations_repeated(
                    gtx_transformations.RemoveAccessNodeCopies(),
                    validate=False,
                    validate_all=False,
                )
            )
        if "scan_loop_unrolling" not in optimization_args:
            optimization_args["scan_loop_unrolling"] = True
        if "scan_loop_unrolling_factor" not in optimization_args:
            optimization_args["scan_loop_unrolling_factor"] = 0
    # TODO(havogt): Eventually the option `use_zero_origin` should be removed and the default behavior should be `use_zero_origin=False`.
    # We keep it `True` for 'compute_rho_theta_pgrad_and_update_vn' as performance drops,
    # due to it falling into a less optimized code generation (on santis).
    if program_name == "compute_rho_theta_pgrad_and_update_vn":
        backend_descriptor["use_zero_origin"] = True
    if program_name == "graupel_run":
        optimization_args["fuse_tasklets"] = True
        if not is_rocm_device:
            optimization_args["gpu_maxnreg"] = 80
            optimization_args["gpu_block_size_2d"] = (64, 6)
        optimization_args["gpu_memory_pool"] = False
        optimization_args["make_persistent"] = True
    if optimization_hooks:
        optimization_args["optimization_hooks"] = optimization_hooks
    if optimization_args:
        backend_descriptor["optimization_args"] = optimization_args
    return backend_descriptor


def get_gtfn_options(  # noqa: PLR0912  # one branch per tuned program; a flat if-chain is intentional
    program_name: str,
    *,
    horizontal_end: int | None = None,
    num_cells: int | None = None,
    num_levels: int | None = None,
    **backend_descriptor: Any,
) -> model_backends.BackendDescriptor:
    if program_name == "compute_rho_theta_pgrad_and_update_vn":
        # Merge the same-domain Green-Gauss gradient reductions into a single kernel + K-coarsening.
        # loop_v (levels/thread) must scale with the HORIZONTAL grid size: a large loop_v on a small
        # grid leaves too few thread-blocks to fill the GPU (the merged reduction is register-starved,
        # reg128). Tuned by the gt4py per-kernel "compute" metric (the correct signal for kernel
        # tuning): on real-data ch1, loop_v=1 is +24% vs the full-column loop_v=10. Small grids want
        # low loop_v; big global grids want the full column (loop_v = num_levels/block_v). Clamp >=2
        # (loop_v=1 hangs the kband lowering). NB: the dynamics-granule wall-clock (nh_solve) does NOT
        # reflect this per-kernel gain — it is physics/dispatch/node-noise dominated — so tune with
        # gt4py metrics, report with full-component timers. RHO_THETA_LOOP_V overrides.
        backend_descriptor.setdefault("enable_tmp_merge", True)
        backend_descriptor.setdefault("thread_block_sizes", (32, 8))
        block_v = 8
        full_col = max(1, (num_levels if num_levels is not None else 80) // block_v)
        if "RHO_THETA_LOOP_V" in os.environ:
            loop_v = int(os.environ["RHO_THETA_LOOP_V"])
        elif horizontal_end is not None:
            # loop_v ~ full_col * (edges / SATURATION); below SATURATION the grid can't fill the GPU
            # at full coarsening so loop_v shrinks. Clamp [2, full_col]. Calibrate via gt4py metrics.
            loop_v = max(2, min(full_col, round(full_col * int(horizontal_end) / 250_000)))
        else:
            loop_v = full_col
        backend_descriptor.setdefault("loop_block_sizes", (1, loop_v))
        # The merged Green-Gauss reduction kernel is register-starved (reg128 / occ24% @L80).
        # K-band splitting peels work off it into high-occupancy bands (+4% on top of the merge +
        # K-coarsening); concat_where fusion + vertical-shift fusion enable the K-band lowering.
        backend_descriptor.setdefault("enable_concat_where_fusion", True)
        backend_descriptor.setdefault("enable_vertical_shift_fusion", True)
        backend_descriptor.setdefault("enable_kband_split", True)
    if program_name == "compute_horizontal_velocity_quantities_and_fluxes":
        # Bandwidth-bound edge-gather + pointwise program. The dominant win is the
        # vertical-shift fusion: the `tangential_wind` reduction is read at Koff[-1] by the
        # half-level interpolation, so without fusion it is materialized as a separate kernel +
        # DRAM round-trip; inlining (recompute) it drops a kernel. With full-column K-coarsening
        # on top this is +16% (laptop) / +11-12% (GH200) vs dace, vs ~parity / -7% without.
        # loop_block_sizes vertical is full-column = num_levels / thread_block_vertical: (1, 5)
        # is tuned for 40 levels with thread (32, 8); GH200 @ 80 levels wants (1, 10).
        # TODO(havogt): derive loop_block_sizes vertical from num_levels.
        # Env overrides for config sweeps:
        #   HVEL_BLOCK_H (32), HVEL_BLOCK_V (8), HVEL_LOOP_V ("5"; "off"/"0" = no loop-block),
        #   HVEL_FUSE ("1"; "0" disables the fusion).
        block_h = int(os.environ.get("HVEL_BLOCK_H", "32"))
        block_v = int(os.environ.get("HVEL_BLOCK_V", "8"))
        loop_v = os.environ.get("HVEL_LOOP_V", "5")
        backend_descriptor.setdefault("thread_block_sizes", (block_h, block_v))
        if loop_v not in ("off", "0"):
            backend_descriptor.setdefault("loop_block_sizes", (1, int(loop_v)))
        backend_descriptor.setdefault(
            "enable_vertical_shift_fusion", os.environ.get("HVEL_FUSE", "1") == "1"
        )
    if program_name == "compute_advection_in_corrector_vertical_momentum":
        # The CFL-clip concat_where splits the cell chain into kernels that re-read cw/w/ddqz
        # (1.52x dace's DRAM traffic). move_dataflow_into_concat_where (gtfn analog of dace's
        # MoveDataflowIntoIfBody) domain-extends the pointwise CFL branch producers so the chain
        # fuses. That in turn unblocks the vertical-shift fusion on the cw temp (its CFL concat_where
        # no longer trips the shifted-domain lift), inlining the biggest temp by pointwise recompute:
        # together they collapse 8 kernels/7 temps -> 3/2, cutting DRAM traffic to dace-parity
        # (4558 vs 4518MB) for +26% on GH200 (gap to dace 58% -> 23%; -maxrregcount=32 takes it to
        # 17%). No K-coarsening (the fused kernel wants 1 K/thread for occupancy). The transform is
        # validate-or-revert: it falls back to the unfused lowering where the fusion would leave an
        # unlowerable lift (dynamic-domain variant) or extend a field-gather below the vertical start
        # (OOB), so the win holds on compile_time_domain while other variants/programs stay correct.
        backend_descriptor.setdefault("thread_block_sizes", (32, 4))
        backend_descriptor.setdefault("enable_concat_where_fusion", True)
        backend_descriptor.setdefault("enable_vertical_shift_fusion", True)
    if program_name in (
        "vertically_implicit_solver_at_predictor_step",
        "vertically_implicit_solver_at_corrector_step",
    ):
        # GRID-AWARE block shape (tuned by the gt4py per-kernel "compute" metric, validated on real-
        # data ch1). Small limited-area grids (< ~60k cells) want a SMALLER block + NO K-coarsening:
        # (16,4)/off gives more, smaller thread-blocks to fill the SMs — predictor +3.6% / corrector
        # +6.5% vs the big-grid (32,8)/lv5 on ch1 (real-data gt4py-metric A/B). Big global grids keep
        # (32,8)/lv5 ((16,4)/off REGRESSES -13% on R02B06). r2b5 (~82k cells) stays on the big config:
        # its solver is gtfn-favorable there and the small config is untested on it — threshold 60k
        # keeps ch1/mch (~21-44k) small and r2b5/R02B06 big. SOLVER_BLOCK_H/V/LOOP_V env override.
        # (loop_v alone is weak/non-monotonic here; the win is the block shape. Scans don't K-coarsen.)
        small_grid = num_cells is not None and int(num_cells) < 60_000
        block_h = int(os.environ.get("SOLVER_BLOCK_H", "16" if small_grid else "32"))
        block_v = int(os.environ.get("SOLVER_BLOCK_V", "4" if small_grid else "8"))
        loop_v = os.environ.get("SOLVER_LOOP_V", "off" if small_grid else "5")
        backend_descriptor.setdefault("thread_block_sizes", (block_h, block_v))
        if loop_v not in ("off", "0"):
            backend_descriptor.setdefault("loop_block_sizes", (1, int(loop_v)))
        # Merging same-domain temporaries gives +2% on both solver steps (GH200 sweep). gtfn
        # already beats dace here (+13-15%); merge is an additional free win.
        backend_descriptor.setdefault("enable_tmp_merge", True)
    if program_name == "compute_advection_in_horizontal_momentum":
        # Branchless skip-value V2E gather lever: lower the vorticity reduction's skip-value
        # (-1) neighbor accesses to branchless load-then-mask using a once-hoisted neighbor row,
        # collapsing the per-neighbor integer/addressing SASS (V2E INT 158 -> 99). GH200 -13%
        # (gap to dace +25% -> +8.7%), ncu-confirmed, correctness PASS. Opt-in here only: the
        # branchless rewrite can regress a large co-resident kernel on other programs (e.g. vmom).
        backend_descriptor.setdefault("enable_branchless_skip_reduce", True)
    if program_name == "compute_advection_in_predictor_vertical_momentum":
        # Connectivity-inline lever: a single-hop connectivity reduction temp is recomputed at the
        # neighbor location instead of being materialized + re-gathered, dropping a kernel + its DRAM
        # round-trip. With the same-domain temp merge and full-column K-coarsening this makes gtfn
        # beat dace by +7% on GH200 @L80 (0.884 vs dace 0.943; base gtfn 1.729). loop_block vertical
        # is full-column = num_levels / thread_block_vertical: (1, 5) is tuned for 40 levels with
        # thread (32, 8); GH200 @ 80 levels wants (1, 10). TODO(havogt): derive from num_levels.
        # Program-dependent: hero on predictor-vmom, a loss on corrector-vmom -> opt-in here only.
        backend_descriptor.setdefault("enable_tmp_merge", True)
        backend_descriptor.setdefault("enable_connectivity_inline", True)
        backend_descriptor.setdefault("thread_block_sizes", (32, 8))
        backend_descriptor.setdefault("loop_block_sizes", (1, 5))
    if program_name == "apply_divergence_damping_and_update_vn":
        # gtfn beats dace by +45% here at block (32,8) (GH200 sweep @L80: 1.27 vs dace 1.84);
        # loop_block is neutral, no transform needed — the (32,8) thread-block is the win.
        backend_descriptor.setdefault("thread_block_sizes", (32, 8))
    if program_name == "compute_perturbed_quantities_and_interpolation":
        # vsf inlines the cheap producer temp z_rth_pr_2 (read at Koff by the heavy 14-input
        # consumer kernel), dropping a kernel (7 -> 6); closes the dace gap from -9% to parity
        # (GH200 @L80: 1.868 vs dace 1.848). Remaining gap = the producer->consumer chain wall.
        backend_descriptor.setdefault("thread_block_sizes", (32, 8))
        backend_descriptor.setdefault(
            "enable_vertical_shift_fusion", os.environ.get("HVEL_FUSE", "1") == "1"
        )
    if program_name == "compute_hydrostatic_correction_term":
        # gtfn's single fused E2C dual-gather + data-dependent as_offset(Koff, ikoffset) kernel
        # beats dace; the win is the thread-block. GH200 @L80: (32,16) 0.1028 vs dace 0.1132
        # (+10.1%); (32,8) +5.4%, default (32,4) -6%. K-coarsening (loop_block) HURTS (dynamic Koff).
        backend_descriptor.setdefault("thread_block_sizes", (32, 16))
    if program_name == "apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence":
        # Fold the two same-gather C2E2CO gradient reductions (dwdx/dwdy) into one: GH200 @L80
        # +4.6% (0.853 -> 0.815), correctness-safe, stock header. Still a -10% holdout
        # (latency-bound) but the best gtfn config for this program.
        backend_descriptor.setdefault("enable_sibling_reduce_fusion", True)
    if program_name == "apply_diffusion_to_theta_and_exner":
        # Full-column K-coarsening on the single fused C2E2C + data-dependent as_offset(Koff)
        # kernel: GH200 @L80 (32,16) loop(1,5) = 1.324 vs dace 2.010 (+52%); base (32,4) is -20%.
        # K-coarsening cuts launch/addressing overhead (laptop +38% -> GH200 +52%).
        backend_descriptor.setdefault("thread_block_sizes", (32, 16))
        backend_descriptor.setdefault("loop_block_sizes", (1, 5))
    if program_name == "compute_diagnostics_from_normal_wind":
        # concat_where fusion folds the `tangential_wind` E2C2E reduction into the
        # `tangential_wind_on_half_levels` interpolation that reads it at Koff[-1], dropping a kernel
        # (10 -> 9). The half-level interpolation extends the reduction's domain one level below the
        # vertical start to serve the Koff[-1] read; the post-global_tmps start clamp raises it back
        # in-bounds (the below-start level is read only through the concat_where boundary guard, so
        # the clamp is value-preserving) — keeping the fusion while avoiding the OOB gather that
        # would otherwise crash on GPU. Measured +5.7% on GH200 (gtfn 2.552 vs dace 2.698).
        # TODO(havogt): needs a fresh GH200 A/B to confirm the +5.7% with this config.
        backend_descriptor.setdefault("enable_tmp_merge", True)
        backend_descriptor.setdefault("enable_concat_where_fusion", True)
        backend_descriptor.setdefault("thread_block_sizes", (32, 8))
    return backend_descriptor


def get_options(
    program_name: str,
    *,
    horizontal_end: int | None = None,
    num_cells: int | None = None,
    num_levels: int | None = None,
    **backend_descriptor: Any,
) -> model_backends.BackendDescriptor:
    if "backend_factory" not in backend_descriptor:
        # here we could set a backend_factory per program
        backend_descriptor["backend_factory"] = model_backends.make_custom_dace_backend
    if backend_descriptor["backend_factory"] == model_backends.make_custom_dace_backend:
        backend_descriptor = get_dace_options(program_name, **backend_descriptor)
    if backend_descriptor["backend_factory"] == model_backends.make_custom_gtfn_backend:
        backend_descriptor = get_gtfn_options(
            program_name,
            horizontal_end=horizontal_end,
            num_cells=num_cells,
            num_levels=num_levels,
            **backend_descriptor,
        )

    return backend_descriptor


def customize_backend(
    program: gtx_typing.Program | gtx.typing.FieldOperator | None,
    backend: gtx_typing.Backend
    | model_backends.DeviceType
    | model_backends.BackendDescriptor
    | None,
    *,
    horizontal_end: int | None = None,
    num_cells: int | None = None,
    num_levels: int | None = None,
) -> gtx_typing.Backend | None:
    program_name = program.__name__ if program is not None else ""
    if backend is None or isinstance(backend, gtx_backend.Backend):
        backend_name = backend.name if backend is not None else "embedded"
        log.info(f"Using non-custom backend '{backend_name}' for '{program_name}'.")
        return backend  # type: ignore[return-value]

    backend_descriptor = (
        {"device": backend} if isinstance(backend, model_backends.DeviceType) else backend
    )
    backend_descriptor = get_options(
        program_name,
        horizontal_end=horizontal_end,
        num_cells=num_cells,
        num_levels=num_levels,
        **backend_descriptor,
    )
    backend_descriptor["device"] = backend_descriptor.get(
        "device", model_backends.CPU
    )  # set default device
    backend_factory = backend_descriptor.pop(
        "backend_factory", model_backends.make_custom_dace_backend
    )
    custom_backend = backend_factory(**backend_descriptor)
    log.info(
        f"Using custom backend '{custom_backend.name}' for '{program_name}' with options: {backend_descriptor}."
    )
    return custom_backend


def setup_program(
    *,
    program: gtx_typing.Program,
    backend: gtx_typing.Backend
    | model_backends.DeviceType
    | model_backends.BackendDescriptor
    | None,
    constant_args: dict[str, gtx.Field | gtx_typing.Scalar] | None = None,
    variants: dict[str, list[gtx_typing.Scalar]] | None = None,
    horizontal_sizes: dict[str, gtx.int32] | None = None,
    vertical_sizes: dict[str, gtx.int32] | None = None,
    offset_provider: gtx_typing.OffsetProvider | None = None,
) -> Callable[..., None]:
    """
    This function processes arguments to the GT4Py program. It
    - binds arguments that don't change during model run ('constant_args', 'horizontal_sizes', "vertical_sizes');
    - inlines scalar arguments into the GT4Py program at compile-time (via GT4Py's 'compile').
    Args:
        - backend: GT4Py backend,
        - program: GT4Py program,
        - constant_args: constant fields and scalars,
        - variants: list of all scalars potential values from which one is selected at run time,
        - horizontal_sizes: horizontal domain bounds,
        - vertical_sizes: vertical domain bounds,
        - offset_provider: GT4Py offset_provider,
    """
    constant_args = {} if constant_args is None else constant_args
    variants = {} if variants is None else variants
    horizontal_sizes = {} if horizontal_sizes is None else horizontal_sizes
    vertical_sizes = {} if vertical_sizes is None else vertical_sizes
    offset_provider = {} if offset_provider is None else offset_provider

    backend = customize_backend(
        program,
        backend,
        horizontal_end=horizontal_sizes.get("horizontal_end"),
        num_cells=horizontal_sizes.get("end_cell_index_halo_lvl1"),
        num_levels=vertical_sizes.get("vertical_end"),
    )

    bound_static_args = {k: v for k, v in constant_args.items() if gtx.is_scalar_type(v)}
    static_args_program = program.with_backend(backend)
    if backend is not None:
        static_args_program = static_args_program.with_compilation_options(enable_jit=False)
        static_args_program.compile(
            **dict_values_to_list(horizontal_sizes),
            **dict_values_to_list(vertical_sizes),
            **variants,
            **dict_values_to_list(bound_static_args),
            offset_provider=offset_provider,
        )

    return functools.partial(
        static_args_program,
        **constant_args,
        **horizontal_sizes,
        **vertical_sizes,
        offset_provider=offset_provider,
    )
