/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct graupel_run_state_t {
    dace::cuda::Context *gpu_context;
    double * __restrict__ __0_gtir_tmp_1675;
    double * __restrict__ __0_gtir_tmp_2004;
    bool * __restrict__ __0_gtir_tmp_2015;
    double * __restrict__ __0_gtir_tmp_2002;
    bool * __restrict__ __0_gtir_tmp_2012;
    bool * __restrict__ __0_gtir_tmp_2018;
    double * __restrict__ __0_gtir_tmp_2006;
    double * __restrict__ __0_gtir_tmp_2000;
    bool * __restrict__ __0_gtir_tmp_2009;
};

DACE_EXPORTED void __dace_runkernel_map_0_fieldop_0_0_115(graupel_run_state_t *__state, double * __restrict__ gtir_tmp_1675, double * __restrict__ gtir_tmp_2000, double * __restrict__ gtir_tmp_2002, double * __restrict__ gtir_tmp_2004, double * __restrict__ gtir_tmp_2006, bool * __restrict__ gtir_tmp_2009, bool * __restrict__ gtir_tmp_2012, bool * __restrict__ gtir_tmp_2015, bool * __restrict__ gtir_tmp_2018, const double * __restrict__ p, const double * __restrict__ q_in_0, const double * __restrict__ q_in_1, const double * __restrict__ q_in_2, const double * __restrict__ q_in_3, const double * __restrict__ q_in_4, const double * __restrict__ q_in_5, double * __restrict__ q_out_0, double * __restrict__ q_out_1, const double * __restrict__ rho, const double * __restrict__ te, int __p_Cell_range_0, int __p_K_range_0, int __p_K_stride, int __q_in_0_Cell_range_0, int __q_in_0_Cell_stride, int __q_in_0_K_range_0, int __q_in_0_K_stride, int __q_in_1_Cell_range_0, int __q_in_1_Cell_stride, int __q_in_1_K_range_0, int __q_in_1_K_stride, int __q_in_2_Cell_range_0, int __q_in_2_Cell_stride, int __q_in_2_K_range_0, int __q_in_2_K_stride, int __q_in_3_Cell_range_0, int __q_in_3_Cell_stride, int __q_in_3_K_range_0, int __q_in_3_K_stride, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride, int __q_in_4_K_range_0, int __q_in_4_K_stride, int __q_in_5_Cell_range_0, int __q_in_5_Cell_stride, int __q_in_5_K_range_0, int __q_in_5_K_stride, int __q_out_0_Cell_range_0, int __q_out_0_Cell_stride, int __q_out_0_K_range_0, int __q_out_0_K_stride, int __q_out_1_Cell_range_0, int __q_out_1_Cell_stride, int __q_out_1_K_range_0, int __q_out_1_K_stride, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride);
DACE_EXPORTED void __dace_runkernel_map_689_fieldop_0_0_117(graupel_run_state_t *__state, const double * __restrict__ dz, const double * __restrict__ gtir_tmp_1675, const double * __restrict__ gtir_tmp_2000, const double * __restrict__ gtir_tmp_2002, const double * __restrict__ gtir_tmp_2004, const double * __restrict__ gtir_tmp_2006, const bool * __restrict__ gtir_tmp_2009, const bool * __restrict__ gtir_tmp_2012, const bool * __restrict__ gtir_tmp_2015, const bool * __restrict__ gtir_tmp_2018, double * __restrict__ pflx, double * __restrict__ pg, double * __restrict__ pi, double * __restrict__ pr, double * __restrict__ pre, double * __restrict__ ps, const double * __restrict__ q_out_0, const double * __restrict__ q_out_1, double * __restrict__ q_out_2, double * __restrict__ q_out_3, double * __restrict__ q_out_4, double * __restrict__ q_out_5, const double * __restrict__ rho, double * __restrict__ t_out, int __dz_Cell_range_0, int __dz_K_range_0, int __dz_K_stride, int __pflx_Cell_range_0, int __pflx_K_range_0, int __pflx_K_stride, int __pg_Cell_range_0, int __pg_K_range_0, int __pg_K_stride, int __pi_Cell_range_0, int __pi_K_range_0, int __pi_K_stride, int __pr_Cell_range_0, int __pr_K_range_0, int __pr_K_stride, int __pre_Cell_range_0, int __pre_K_range_0, int __pre_K_stride, int __ps_Cell_range_0, int __ps_K_range_0, int __ps_K_stride, int __q_out_0_Cell_range_0, int __q_out_0_Cell_stride, int __q_out_0_K_range_0, int __q_out_0_K_stride, int __q_out_1_Cell_range_0, int __q_out_1_Cell_stride, int __q_out_1_K_range_0, int __q_out_1_K_stride, int __q_out_2_Cell_range_0, int __q_out_2_Cell_stride, int __q_out_2_K_range_0, int __q_out_2_K_stride, int __q_out_3_Cell_range_0, int __q_out_3_Cell_stride, int __q_out_3_K_range_0, int __q_out_3_K_stride, int __q_out_4_Cell_range_0, int __q_out_4_Cell_stride, int __q_out_4_K_range_0, int __q_out_4_K_stride, int __q_out_5_Cell_range_0, int __q_out_5_Cell_stride, int __q_out_5_K_range_0, int __q_out_5_K_stride, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride, int __t_out_Cell_range_0, int __t_out_K_range_0, int __t_out_K_stride);
void __program_graupel_run_internal(graupel_run_state_t*__state, double * __restrict__ dz, double * __restrict__ gt_compute_time, double * __restrict__ p, double * __restrict__ pflx, double * __restrict__ pg, double * __restrict__ pi, double * __restrict__ pr, double * __restrict__ pre, double * __restrict__ ps, double * __restrict__ q_in_0, double * __restrict__ q_in_1, double * __restrict__ q_in_2, double * __restrict__ q_in_3, double * __restrict__ q_in_4, double * __restrict__ q_in_5, double * __restrict__ q_out_0, double * __restrict__ q_out_1, double * __restrict__ q_out_2, double * __restrict__ q_out_3, double * __restrict__ q_out_4, double * __restrict__ q_out_5, double * __restrict__ rho, double * __restrict__ t_out, double * __restrict__ te, int __dz_Cell_range_0, int __dz_K_range_0, int __dz_K_stride, int __p_Cell_range_0, int __p_K_range_0, int __p_K_stride, int __pflx_Cell_range_0, int __pflx_K_range_0, int __pflx_K_stride, int __pg_Cell_range_0, int __pg_K_range_0, int __pg_K_stride, int __pi_Cell_range_0, int __pi_K_range_0, int __pi_K_stride, int __pr_Cell_range_0, int __pr_K_range_0, int __pr_K_stride, int __pre_Cell_range_0, int __pre_K_range_0, int __pre_K_stride, int __ps_Cell_range_0, int __ps_K_range_0, int __ps_K_stride, int __q_in_0_Cell_range_0, int __q_in_0_Cell_stride, int __q_in_0_K_range_0, int __q_in_0_K_stride, int __q_in_1_Cell_range_0, int __q_in_1_Cell_stride, int __q_in_1_K_range_0, int __q_in_1_K_stride, int __q_in_2_Cell_range_0, int __q_in_2_Cell_stride, int __q_in_2_K_range_0, int __q_in_2_K_stride, int __q_in_3_Cell_range_0, int __q_in_3_Cell_stride, int __q_in_3_K_range_0, int __q_in_3_K_stride, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride, int __q_in_4_K_range_0, int __q_in_4_K_stride, int __q_in_5_Cell_range_0, int __q_in_5_Cell_stride, int __q_in_5_K_range_0, int __q_in_5_K_stride, int __q_out_0_Cell_range_0, int __q_out_0_Cell_stride, int __q_out_0_K_range_0, int __q_out_0_K_stride, int __q_out_1_Cell_range_0, int __q_out_1_Cell_stride, int __q_out_1_K_range_0, int __q_out_1_K_stride, int __q_out_2_Cell_range_0, int __q_out_2_Cell_stride, int __q_out_2_K_range_0, int __q_out_2_K_stride, int __q_out_3_Cell_range_0, int __q_out_3_Cell_stride, int __q_out_3_K_range_0, int __q_out_3_K_stride, int __q_out_4_Cell_range_0, int __q_out_4_Cell_stride, int __q_out_4_K_range_0, int __q_out_4_K_stride, int __q_out_5_Cell_range_0, int __q_out_5_Cell_stride, int __q_out_5_K_range_0, int __q_out_5_K_stride, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride, int __t_out_Cell_range_0, int __t_out_K_range_0, int __t_out_K_stride, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride, int gt_metrics_level)
{
    int64_t gt_start_time;

    if ((gt_metrics_level >= 10)) {
        {

            {
                int64_t time;

                ///////////////////
                auto now = std::chrono::high_resolution_clock::now();
                time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch()
                ).count();

                ///////////////////

                gt_start_time = time;
            }

        }
    }
    {

        __dace_runkernel_map_0_fieldop_0_0_115(__state, __state->__0_gtir_tmp_1675, __state->__0_gtir_tmp_2000, __state->__0_gtir_tmp_2002, __state->__0_gtir_tmp_2004, __state->__0_gtir_tmp_2006, __state->__0_gtir_tmp_2009, __state->__0_gtir_tmp_2012, __state->__0_gtir_tmp_2015, __state->__0_gtir_tmp_2018, p, q_in_0, q_in_1, q_in_2, q_in_3, q_in_4, q_in_5, q_out_0, q_out_1, rho, te, __p_Cell_range_0, __p_K_range_0, __p_K_stride, __q_in_0_Cell_range_0, __q_in_0_Cell_stride, __q_in_0_K_range_0, __q_in_0_K_stride, __q_in_1_Cell_range_0, __q_in_1_Cell_stride, __q_in_1_K_range_0, __q_in_1_K_stride, __q_in_2_Cell_range_0, __q_in_2_Cell_stride, __q_in_2_K_range_0, __q_in_2_K_stride, __q_in_3_Cell_range_0, __q_in_3_Cell_stride, __q_in_3_K_range_0, __q_in_3_K_stride, __q_in_4_Cell_range_0, __q_in_4_Cell_stride, __q_in_4_K_range_0, __q_in_4_K_stride, __q_in_5_Cell_range_0, __q_in_5_Cell_stride, __q_in_5_K_range_0, __q_in_5_K_stride, __q_out_0_Cell_range_0, __q_out_0_Cell_stride, __q_out_0_K_range_0, __q_out_0_K_stride, __q_out_1_Cell_range_0, __q_out_1_Cell_stride, __q_out_1_K_range_0, __q_out_1_K_stride, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride, __te_Cell_range_0, __te_K_range_0, __te_K_stride);
        __dace_runkernel_map_689_fieldop_0_0_117(__state, dz, __state->__0_gtir_tmp_1675, __state->__0_gtir_tmp_2000, __state->__0_gtir_tmp_2002, __state->__0_gtir_tmp_2004, __state->__0_gtir_tmp_2006, __state->__0_gtir_tmp_2009, __state->__0_gtir_tmp_2012, __state->__0_gtir_tmp_2015, __state->__0_gtir_tmp_2018, pflx, pg, pi, pr, pre, ps, q_out_0, q_out_1, q_out_2, q_out_3, q_out_4, q_out_5, rho, t_out, __dz_Cell_range_0, __dz_K_range_0, __dz_K_stride, __pflx_Cell_range_0, __pflx_K_range_0, __pflx_K_stride, __pg_Cell_range_0, __pg_K_range_0, __pg_K_stride, __pi_Cell_range_0, __pi_K_range_0, __pi_K_stride, __pr_Cell_range_0, __pr_K_range_0, __pr_K_stride, __pre_Cell_range_0, __pre_K_range_0, __pre_K_stride, __ps_Cell_range_0, __ps_K_range_0, __ps_K_stride, __q_out_0_Cell_range_0, __q_out_0_Cell_stride, __q_out_0_K_range_0, __q_out_0_K_stride, __q_out_1_Cell_range_0, __q_out_1_Cell_stride, __q_out_1_K_range_0, __q_out_1_K_stride, __q_out_2_Cell_range_0, __q_out_2_Cell_stride, __q_out_2_K_range_0, __q_out_2_K_stride, __q_out_3_Cell_range_0, __q_out_3_Cell_stride, __q_out_3_K_range_0, __q_out_3_K_stride, __q_out_4_Cell_range_0, __q_out_4_Cell_stride, __q_out_4_K_range_0, __q_out_4_K_stride, __q_out_5_Cell_range_0, __q_out_5_Cell_stride, __q_out_5_K_range_0, __q_out_5_K_stride, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride, __t_out_Cell_range_0, __t_out_K_range_0, __t_out_K_stride);

    }
    if ((gt_metrics_level >= 10)) {
        {

            {
                int64_t run_cpp_start_time = gt_start_time;
                double duration;

                ///////////////////
                cudaDeviceSynchronize();
                auto now = std::chrono::high_resolution_clock::now();
                auto run_cpp_end_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch()
                ).count();
                duration = static_cast<double>(run_cpp_end_time - run_cpp_start_time) * 1.e-9;

                ///////////////////

                gt_compute_time[0] = duration;
            }

        }
    }
}

DACE_EXPORTED void __program_graupel_run(graupel_run_state_t *__state, double * __restrict__ dz, double * __restrict__ gt_compute_time, double * __restrict__ p, double * __restrict__ pflx, double * __restrict__ pg, double * __restrict__ pi, double * __restrict__ pr, double * __restrict__ pre, double * __restrict__ ps, double * __restrict__ q_in_0, double * __restrict__ q_in_1, double * __restrict__ q_in_2, double * __restrict__ q_in_3, double * __restrict__ q_in_4, double * __restrict__ q_in_5, double * __restrict__ q_out_0, double * __restrict__ q_out_1, double * __restrict__ q_out_2, double * __restrict__ q_out_3, double * __restrict__ q_out_4, double * __restrict__ q_out_5, double * __restrict__ rho, double * __restrict__ t_out, double * __restrict__ te, int __dz_Cell_range_0, int __dz_K_range_0, int __dz_K_stride, int __p_Cell_range_0, int __p_K_range_0, int __p_K_stride, int __pflx_Cell_range_0, int __pflx_K_range_0, int __pflx_K_stride, int __pg_Cell_range_0, int __pg_K_range_0, int __pg_K_stride, int __pi_Cell_range_0, int __pi_K_range_0, int __pi_K_stride, int __pr_Cell_range_0, int __pr_K_range_0, int __pr_K_stride, int __pre_Cell_range_0, int __pre_K_range_0, int __pre_K_stride, int __ps_Cell_range_0, int __ps_K_range_0, int __ps_K_stride, int __q_in_0_Cell_range_0, int __q_in_0_Cell_stride, int __q_in_0_K_range_0, int __q_in_0_K_stride, int __q_in_1_Cell_range_0, int __q_in_1_Cell_stride, int __q_in_1_K_range_0, int __q_in_1_K_stride, int __q_in_2_Cell_range_0, int __q_in_2_Cell_stride, int __q_in_2_K_range_0, int __q_in_2_K_stride, int __q_in_3_Cell_range_0, int __q_in_3_Cell_stride, int __q_in_3_K_range_0, int __q_in_3_K_stride, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride, int __q_in_4_K_range_0, int __q_in_4_K_stride, int __q_in_5_Cell_range_0, int __q_in_5_Cell_stride, int __q_in_5_K_range_0, int __q_in_5_K_stride, int __q_out_0_Cell_range_0, int __q_out_0_Cell_stride, int __q_out_0_K_range_0, int __q_out_0_K_stride, int __q_out_1_Cell_range_0, int __q_out_1_Cell_stride, int __q_out_1_K_range_0, int __q_out_1_K_stride, int __q_out_2_Cell_range_0, int __q_out_2_Cell_stride, int __q_out_2_K_range_0, int __q_out_2_K_stride, int __q_out_3_Cell_range_0, int __q_out_3_Cell_stride, int __q_out_3_K_range_0, int __q_out_3_K_stride, int __q_out_4_Cell_range_0, int __q_out_4_Cell_stride, int __q_out_4_K_range_0, int __q_out_4_K_stride, int __q_out_5_Cell_range_0, int __q_out_5_Cell_stride, int __q_out_5_K_range_0, int __q_out_5_K_stride, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride, int __t_out_Cell_range_0, int __t_out_K_range_0, int __t_out_K_stride, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride, int gt_metrics_level)
{
    __program_graupel_run_internal(__state, dz, gt_compute_time, p, pflx, pg, pi, pr, pre, ps, q_in_0, q_in_1, q_in_2, q_in_3, q_in_4, q_in_5, q_out_0, q_out_1, q_out_2, q_out_3, q_out_4, q_out_5, rho, t_out, te, __dz_Cell_range_0, __dz_K_range_0, __dz_K_stride, __p_Cell_range_0, __p_K_range_0, __p_K_stride, __pflx_Cell_range_0, __pflx_K_range_0, __pflx_K_stride, __pg_Cell_range_0, __pg_K_range_0, __pg_K_stride, __pi_Cell_range_0, __pi_K_range_0, __pi_K_stride, __pr_Cell_range_0, __pr_K_range_0, __pr_K_stride, __pre_Cell_range_0, __pre_K_range_0, __pre_K_stride, __ps_Cell_range_0, __ps_K_range_0, __ps_K_stride, __q_in_0_Cell_range_0, __q_in_0_Cell_stride, __q_in_0_K_range_0, __q_in_0_K_stride, __q_in_1_Cell_range_0, __q_in_1_Cell_stride, __q_in_1_K_range_0, __q_in_1_K_stride, __q_in_2_Cell_range_0, __q_in_2_Cell_stride, __q_in_2_K_range_0, __q_in_2_K_stride, __q_in_3_Cell_range_0, __q_in_3_Cell_stride, __q_in_3_K_range_0, __q_in_3_K_stride, __q_in_4_Cell_range_0, __q_in_4_Cell_stride, __q_in_4_K_range_0, __q_in_4_K_stride, __q_in_5_Cell_range_0, __q_in_5_Cell_stride, __q_in_5_K_range_0, __q_in_5_K_stride, __q_out_0_Cell_range_0, __q_out_0_Cell_stride, __q_out_0_K_range_0, __q_out_0_K_stride, __q_out_1_Cell_range_0, __q_out_1_Cell_stride, __q_out_1_K_range_0, __q_out_1_K_stride, __q_out_2_Cell_range_0, __q_out_2_Cell_stride, __q_out_2_K_range_0, __q_out_2_K_stride, __q_out_3_Cell_range_0, __q_out_3_Cell_stride, __q_out_3_K_range_0, __q_out_3_K_stride, __q_out_4_Cell_range_0, __q_out_4_Cell_stride, __q_out_4_K_range_0, __q_out_4_K_stride, __q_out_5_Cell_range_0, __q_out_5_Cell_stride, __q_out_5_K_range_0, __q_out_5_K_stride, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride, __t_out_Cell_range_0, __t_out_K_range_0, __t_out_K_stride, __te_Cell_range_0, __te_K_range_0, __te_K_stride, gt_metrics_level);
}
DACE_EXPORTED int __dace_init_cuda(graupel_run_state_t *__state, int __dz_Cell_range_0, int __dz_K_range_0, int __dz_K_stride, int __p_Cell_range_0, int __p_K_range_0, int __p_K_stride, int __pflx_Cell_range_0, int __pflx_K_range_0, int __pflx_K_stride, int __pg_Cell_range_0, int __pg_K_range_0, int __pg_K_stride, int __pi_Cell_range_0, int __pi_K_range_0, int __pi_K_stride, int __pr_Cell_range_0, int __pr_K_range_0, int __pr_K_stride, int __pre_Cell_range_0, int __pre_K_range_0, int __pre_K_stride, int __ps_Cell_range_0, int __ps_K_range_0, int __ps_K_stride, int __q_in_0_Cell_range_0, int __q_in_0_Cell_stride, int __q_in_0_K_range_0, int __q_in_0_K_stride, int __q_in_1_Cell_range_0, int __q_in_1_Cell_stride, int __q_in_1_K_range_0, int __q_in_1_K_stride, int __q_in_2_Cell_range_0, int __q_in_2_Cell_stride, int __q_in_2_K_range_0, int __q_in_2_K_stride, int __q_in_3_Cell_range_0, int __q_in_3_Cell_stride, int __q_in_3_K_range_0, int __q_in_3_K_stride, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride, int __q_in_4_K_range_0, int __q_in_4_K_stride, int __q_in_5_Cell_range_0, int __q_in_5_Cell_stride, int __q_in_5_K_range_0, int __q_in_5_K_stride, int __q_out_0_Cell_range_0, int __q_out_0_Cell_stride, int __q_out_0_K_range_0, int __q_out_0_K_stride, int __q_out_1_Cell_range_0, int __q_out_1_Cell_stride, int __q_out_1_K_range_0, int __q_out_1_K_stride, int __q_out_2_Cell_range_0, int __q_out_2_Cell_stride, int __q_out_2_K_range_0, int __q_out_2_K_stride, int __q_out_3_Cell_range_0, int __q_out_3_Cell_stride, int __q_out_3_K_range_0, int __q_out_3_K_stride, int __q_out_4_Cell_range_0, int __q_out_4_Cell_stride, int __q_out_4_K_range_0, int __q_out_4_K_stride, int __q_out_5_Cell_range_0, int __q_out_5_Cell_stride, int __q_out_5_K_range_0, int __q_out_5_K_stride, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride, int __t_out_Cell_range_0, int __t_out_K_range_0, int __t_out_K_stride, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride, int gt_metrics_level);
DACE_EXPORTED int __dace_exit_cuda(graupel_run_state_t *__state);

DACE_EXPORTED graupel_run_state_t *__dace_init_graupel_run(int __dz_Cell_range_0, int __dz_K_range_0, int __dz_K_stride, int __p_Cell_range_0, int __p_K_range_0, int __p_K_stride, int __pflx_Cell_range_0, int __pflx_K_range_0, int __pflx_K_stride, int __pg_Cell_range_0, int __pg_K_range_0, int __pg_K_stride, int __pi_Cell_range_0, int __pi_K_range_0, int __pi_K_stride, int __pr_Cell_range_0, int __pr_K_range_0, int __pr_K_stride, int __pre_Cell_range_0, int __pre_K_range_0, int __pre_K_stride, int __ps_Cell_range_0, int __ps_K_range_0, int __ps_K_stride, int __q_in_0_Cell_range_0, int __q_in_0_Cell_stride, int __q_in_0_K_range_0, int __q_in_0_K_stride, int __q_in_1_Cell_range_0, int __q_in_1_Cell_stride, int __q_in_1_K_range_0, int __q_in_1_K_stride, int __q_in_2_Cell_range_0, int __q_in_2_Cell_stride, int __q_in_2_K_range_0, int __q_in_2_K_stride, int __q_in_3_Cell_range_0, int __q_in_3_Cell_stride, int __q_in_3_K_range_0, int __q_in_3_K_stride, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride, int __q_in_4_K_range_0, int __q_in_4_K_stride, int __q_in_5_Cell_range_0, int __q_in_5_Cell_stride, int __q_in_5_K_range_0, int __q_in_5_K_stride, int __q_out_0_Cell_range_0, int __q_out_0_Cell_stride, int __q_out_0_K_range_0, int __q_out_0_K_stride, int __q_out_1_Cell_range_0, int __q_out_1_Cell_stride, int __q_out_1_K_range_0, int __q_out_1_K_stride, int __q_out_2_Cell_range_0, int __q_out_2_Cell_stride, int __q_out_2_K_range_0, int __q_out_2_K_stride, int __q_out_3_Cell_range_0, int __q_out_3_Cell_stride, int __q_out_3_K_range_0, int __q_out_3_K_stride, int __q_out_4_Cell_range_0, int __q_out_4_Cell_stride, int __q_out_4_K_range_0, int __q_out_4_K_stride, int __q_out_5_Cell_range_0, int __q_out_5_Cell_stride, int __q_out_5_K_range_0, int __q_out_5_K_stride, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride, int __t_out_Cell_range_0, int __t_out_K_range_0, int __t_out_K_stride, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride, int gt_metrics_level)
{

    int __result = 0;
    graupel_run_state_t *__state = new graupel_run_state_t;
    __result |= __dace_init_cuda(__state, __dz_Cell_range_0, __dz_K_range_0, __dz_K_stride, __p_Cell_range_0, __p_K_range_0, __p_K_stride, __pflx_Cell_range_0, __pflx_K_range_0, __pflx_K_stride, __pg_Cell_range_0, __pg_K_range_0, __pg_K_stride, __pi_Cell_range_0, __pi_K_range_0, __pi_K_stride, __pr_Cell_range_0, __pr_K_range_0, __pr_K_stride, __pre_Cell_range_0, __pre_K_range_0, __pre_K_stride, __ps_Cell_range_0, __ps_K_range_0, __ps_K_stride, __q_in_0_Cell_range_0, __q_in_0_Cell_stride, __q_in_0_K_range_0, __q_in_0_K_stride, __q_in_1_Cell_range_0, __q_in_1_Cell_stride, __q_in_1_K_range_0, __q_in_1_K_stride, __q_in_2_Cell_range_0, __q_in_2_Cell_stride, __q_in_2_K_range_0, __q_in_2_K_stride, __q_in_3_Cell_range_0, __q_in_3_Cell_stride, __q_in_3_K_range_0, __q_in_3_K_stride, __q_in_4_Cell_range_0, __q_in_4_Cell_stride, __q_in_4_K_range_0, __q_in_4_K_stride, __q_in_5_Cell_range_0, __q_in_5_Cell_stride, __q_in_5_K_range_0, __q_in_5_K_stride, __q_out_0_Cell_range_0, __q_out_0_Cell_stride, __q_out_0_K_range_0, __q_out_0_K_stride, __q_out_1_Cell_range_0, __q_out_1_Cell_stride, __q_out_1_K_range_0, __q_out_1_K_stride, __q_out_2_Cell_range_0, __q_out_2_Cell_stride, __q_out_2_K_range_0, __q_out_2_K_stride, __q_out_3_Cell_range_0, __q_out_3_Cell_stride, __q_out_3_K_range_0, __q_out_3_K_stride, __q_out_4_Cell_range_0, __q_out_4_Cell_stride, __q_out_4_K_range_0, __q_out_4_K_stride, __q_out_5_Cell_range_0, __q_out_5_Cell_stride, __q_out_5_K_range_0, __q_out_5_K_stride, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride, __t_out_Cell_range_0, __t_out_K_range_0, __t_out_K_stride, __te_Cell_range_0, __te_K_range_0, __te_K_stride, gt_metrics_level);
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gtir_tmp_1675, 11200 * sizeof(double)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gtir_tmp_2004, 11200 * sizeof(double)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gtir_tmp_2015, 11200 * sizeof(bool)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gtir_tmp_2002, 11200 * sizeof(double)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gtir_tmp_2012, 11200 * sizeof(bool)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gtir_tmp_2018, 11200 * sizeof(bool)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gtir_tmp_2006, 11200 * sizeof(double)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gtir_tmp_2000, 11200 * sizeof(double)));
    DACE_GPU_CHECK(cudaMalloc((void**)&__state->__0_gtir_tmp_2009, 11200 * sizeof(bool)));

    if (__result) {
        delete __state;
        return nullptr;
    }

    return __state;
}

DACE_EXPORTED int __dace_exit_graupel_run(graupel_run_state_t *__state)
{

    int __err = 0;
    DACE_GPU_CHECK(cudaFree(__state->__0_gtir_tmp_1675));
    DACE_GPU_CHECK(cudaFree(__state->__0_gtir_tmp_2004));
    DACE_GPU_CHECK(cudaFree(__state->__0_gtir_tmp_2015));
    DACE_GPU_CHECK(cudaFree(__state->__0_gtir_tmp_2002));
    DACE_GPU_CHECK(cudaFree(__state->__0_gtir_tmp_2012));
    DACE_GPU_CHECK(cudaFree(__state->__0_gtir_tmp_2018));
    DACE_GPU_CHECK(cudaFree(__state->__0_gtir_tmp_2006));
    DACE_GPU_CHECK(cudaFree(__state->__0_gtir_tmp_2000));
    DACE_GPU_CHECK(cudaFree(__state->__0_gtir_tmp_2009));

    int __err_cuda = __dace_exit_cuda(__state);
    if (__err_cuda) {
        __err = __err_cuda;
    }
    delete __state;
    return __err;
}
