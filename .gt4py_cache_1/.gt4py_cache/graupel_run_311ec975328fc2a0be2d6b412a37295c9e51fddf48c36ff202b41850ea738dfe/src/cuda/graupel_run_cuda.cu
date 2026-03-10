
#include <cuda_runtime.h>
#include <dace/dace.h>


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



DACE_EXPORTED int __dace_init_cuda(graupel_run_state_t *__state, int __dz_Cell_range_0, int __dz_K_range_0, int __dz_K_stride, int __p_Cell_range_0, int __p_K_range_0, int __p_K_stride, int __pflx_Cell_range_0, int __pflx_K_range_0, int __pflx_K_stride, int __pg_Cell_range_0, int __pg_K_range_0, int __pg_K_stride, int __pi_Cell_range_0, int __pi_K_range_0, int __pi_K_stride, int __pr_Cell_range_0, int __pr_K_range_0, int __pr_K_stride, int __pre_Cell_range_0, int __pre_K_range_0, int __pre_K_stride, int __ps_Cell_range_0, int __ps_K_range_0, int __ps_K_stride, int __q_in_0_Cell_range_0, int __q_in_0_Cell_stride, int __q_in_0_K_range_0, int __q_in_0_K_stride, int __q_in_1_Cell_range_0, int __q_in_1_Cell_stride, int __q_in_1_K_range_0, int __q_in_1_K_stride, int __q_in_2_Cell_range_0, int __q_in_2_Cell_stride, int __q_in_2_K_range_0, int __q_in_2_K_stride, int __q_in_3_Cell_range_0, int __q_in_3_Cell_stride, int __q_in_3_K_range_0, int __q_in_3_K_stride, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride, int __q_in_4_K_range_0, int __q_in_4_K_stride, int __q_in_5_Cell_range_0, int __q_in_5_Cell_stride, int __q_in_5_K_range_0, int __q_in_5_K_stride, int __q_out_0_Cell_range_0, int __q_out_0_Cell_stride, int __q_out_0_K_range_0, int __q_out_0_K_stride, int __q_out_1_Cell_range_0, int __q_out_1_Cell_stride, int __q_out_1_K_range_0, int __q_out_1_K_stride, int __q_out_2_Cell_range_0, int __q_out_2_Cell_stride, int __q_out_2_K_range_0, int __q_out_2_K_stride, int __q_out_3_Cell_range_0, int __q_out_3_Cell_stride, int __q_out_3_K_range_0, int __q_out_3_K_stride, int __q_out_4_Cell_range_0, int __q_out_4_Cell_stride, int __q_out_4_K_range_0, int __q_out_4_K_stride, int __q_out_5_Cell_range_0, int __q_out_5_Cell_stride, int __q_out_5_K_range_0, int __q_out_5_K_stride, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride, int __t_out_Cell_range_0, int __t_out_K_range_0, int __t_out_K_stride, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride, int gt_metrics_level);
DACE_EXPORTED int __dace_exit_cuda(graupel_run_state_t *__state);
DACE_EXPORTED bool __dace_gpu_set_stream(graupel_run_state_t *__state, int streamid, gpuStream_t stream);
DACE_EXPORTED void __dace_gpu_set_all_streams(graupel_run_state_t *__state, gpuStream_t stream);

DACE_DFI void if_stmt_7_3_0_18(const bool&  __cond, const double&  __map_fusion_gtir_tmp_100, const double*  te, double&  __output, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1____________;
            double gtir_tmp_422_0;
            double gtir_tmp_429_0;
            double gtir_tmp_428_0;
            double gtir_tmp_421_0;
            double __map_fusion_gtir_tmp_446;
            double gtir_tmp_427_0;
            double gtir_tmp_430_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_252_get_value__clone_5b4076a6_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1.4706483095996819e-07;
                ///////////////////

                gtir_tmp_421_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_256_get_value__clone_5b45741c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1.0;
                ///////////////////

                gtir_tmp_427_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_253_get_value__clone_5b2baf50_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.94;
                ///////////////////

                gtir_tmp_422_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_257_get_value__clone_5aef0730_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 725371628819.8883;
                ///////////////////

                gtir_tmp_428_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_259_get_value__clone_5b3b56ee_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.94;
                ///////////////////

                gtir_tmp_430_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_258_get_value__clone_5b314460_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1.4706483095996819e-07;
                ///////////////////

                gtir_tmp_429_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_421_0;
                double __tlet_arg0_0 = gtir_tmp_427_0;
                double __tlet_arg0_0_0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg0_0_1 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg0_1 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg0_1_0 = gtir_tmp_429_0;
                double __tlet_arg0_2 = gtir_tmp_428_0;
                double __tlet_arg1 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_0 = gtir_tmp_422_0;
                double __tlet_arg1_0_0 = gtir_tmp_430_0;
                double __tlet_arg1_1 = __map_fusion_gtir_tmp_100;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_260_power_fused_tlet_261_multiplies_fused_tlet_262_multiplies_fused_tlet_263_multiplies_fused_tlet_264_multiplies_fused_tlet_265_divides_fused_tlet_254_power_fused_tlet_266_plus_fused_tlet_255_multiplies_fused_tlet_267_divides)
                __tlet_result = ((__tlet_arg0 * dace::math::pow(__tlet_arg0_1, __tlet_arg1_0)) / (__tlet_arg0_0 + (((__tlet_arg0_2 * (__tlet_arg0_1_0 * dace::math::pow(__tlet_arg0_0_1, __tlet_arg1_0_0))) * __tlet_arg1_1) / (__tlet_arg0_0_0 * __tlet_arg1))));
                ///////////////////

                __map_fusion_gtir_tmp_446 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_446, &__arg1____________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1____________, &__output, 1);

        }
    } else {
        {
            double __arg2_________;
            double gtir_tmp_447_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_268_get_value__clone_5b3674bc_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_447_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_447_0, &__arg2_________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_________, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_8_10_0_12(const bool&  __cond, const double&  __map_fusion_gtir_tmp_102, const double&  __map_fusion_gtir_tmp_463, const double*  q_in_4, double&  __output, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride_0_0_0, int __q_in_4_K_range_0, int __q_in_4_K_stride_0_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1____________;
            double gtir_tmp_472_0;
            double __map_fusion_gtir_tmp_476;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_282_get_value__clone_5b66b2e4_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_472_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = __map_fusion_gtir_tmp_463;
                double __tlet_arg0_0 = __map_fusion_gtir_tmp_102;
                double __tlet_arg1 = gtir_tmp_472_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_283_divides_fused_tlet_284_minimum)
                __tlet_result = min(__tlet_arg0, (__tlet_arg0_0 / __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_476 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_476, &__arg1____________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1____________, &__output, 1);

        }
    } else {
        {
            double __arg2_________;
            double __map_fusion_gtir_tmp_488;
            double gtir_tmp_477_0;
            double gtir_tmp_484_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_285_get_value__clone_5b720ea0_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_477_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_289_get_value__clone_5b564d64_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_484_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = __map_fusion_gtir_tmp_463;
                double __tlet_arg0_0 = __map_fusion_gtir_tmp_102;
                double __tlet_arg0_1 = q_in_4[((__q_in_4_Cell_stride_0_0_0 * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride_0_0_0 * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_484_0;
                double __tlet_arg1_0 = gtir_tmp_477_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_288_neg_fused_tlet_290_divides_fused_tlet_286_divides_fused_tlet_287_maximum_fused_tlet_291_maximum)
                __tlet_result = max(max(__tlet_arg0, (__tlet_arg0_0 / __tlet_arg1_0)), ((- __tlet_arg0_1) / __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_488 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_488, &__arg2_________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_________, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_9_3_0_43(const bool&  __cond, const double&  __map_fusion_gtir_tmp_102, const double&  __map_fusion_gtir_tmp_398, const double&  gtir_tmp_448, const double*  q_in_4, const double* __restrict__ rho, double&  __output, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride_0_0, int __q_in_4_K_range_0, int __q_in_4_K_stride_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1_______________;
            double gtir_tmp_489;
            bool __map_fusion_gtir_tmp_471;
            double gtir_tmp_457_0;
            double gtir_tmp_450_0;
            double __map_fusion_gtir_tmp_463;
            double gtir_tmp_469_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_273_get_value__clone_5b502b28_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = -0.67;
                ///////////////////

                gtir_tmp_457_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_269_get_value__clone_5b4ac00c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.7896092134994286;
                ///////////////////

                gtir_tmp_450_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_280_get_value__clone_5b5bf7be_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_469_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_450_0;
                double __tlet_arg0_0 = __map_fusion_gtir_tmp_398;
                double __tlet_arg1 = __map_fusion_gtir_tmp_102;
                double __tlet_arg1_0 = gtir_tmp_457_0;
                double __tlet_arg1_0_0 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_1 = gtir_tmp_448;
                double __tlet_arg1_2 = q_in_4[((__q_in_4_Cell_stride_0_0 * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride_0_0 * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_270_multiplies_fused_tlet_271_multiplies_fused_tlet_272_multiplies_fused_tlet_274_power_fused_tlet_275_multiplies_fused_tlet_276_multiplies)
                __tlet_result = (((((__tlet_arg0 * __tlet_arg1_1) * __tlet_arg1_0_0) * __tlet_arg1_2) * dace::math::pow(__tlet_arg0_0, __tlet_arg1_0)) * __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_463 = __tlet_result;
            }
            {
                double __tlet_arg0 = __map_fusion_gtir_tmp_463;
                double __tlet_arg1 = gtir_tmp_469_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_281_greater)
                __tlet_result = (__tlet_arg0 > __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_471 = __tlet_result;
            }
            if_stmt_8_10_0_12(__map_fusion_gtir_tmp_471, __map_fusion_gtir_tmp_102, __map_fusion_gtir_tmp_463, &q_in_4[0], gtir_tmp_489, __q_in_4_Cell_range_0, __q_in_4_Cell_stride_0_0, __q_in_4_K_range_0, __q_in_4_K_stride_0_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_489, &__arg1_______________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_______________, &__output, 1);

        }
    } else {
        {
            double __arg2_;
            double gtir_tmp_491_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_292_get_value__clone_5b61214e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_491_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_491_0, &__arg2_, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_13_3_2_8(const bool&  __cond, const double&  __map_fusion_gtir_tmp_102, const double&  __map_fusion_gtir_tmp_390, const double&  gtir_tmp_506, const double*  q_in_1, const double*  q_in_4, const double*  te, double&  __output, int __q_in_1_Cell_range_0, int __q_in_1_Cell_stride_0_0, int __q_in_1_K_range_0, int __q_in_1_K_stride_0_0, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride_0_0, int __q_in_4_K_range_0, int __q_in_4_K_stride_0_0, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {
    double gtir_tmp_543;
    bool __map_fusion_gtir_tmp_530;

    if (__cond) {
        {
            double gtir_tmp_522_0;
            double gtir_tmp_519_0;
            double gtir_tmp_511_0;
            double gtir_tmp_514_0;
            double gtir_tmp_508_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_307_get_value__clone_5b8e42aa_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 267.15;
                ///////////////////

                gtir_tmp_519_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_300_get_value__clone_5be099a6_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_508_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_309_get_value__clone_5bf28a6c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_522_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_302_get_value__clone_5be694fa_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 248.15;
                ///////////////////

                gtir_tmp_511_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_304_get_value__clone_5b99ef7e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_514_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_4[((__q_in_4_Cell_stride_0_0 * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride_0_0 * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg0_0_0 = __map_fusion_gtir_tmp_102;
                double __tlet_arg0_1 = q_in_1[((__q_in_1_Cell_stride_0_0 * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride_0_0 * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_2 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1 = gtir_tmp_522_0;
                double __tlet_arg1_0 = gtir_tmp_508_0;
                double __tlet_arg1_0_0 = gtir_tmp_511_0;
                double __tlet_arg1_0_1 = gtir_tmp_519_0;
                double __tlet_arg1_1 = gtir_tmp_514_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_303_less_fused_tlet_305_greater_fused_tlet_306_and__fused_tlet_310_greater_fused_tlet_308_less_equal_fused_tlet_311_and__fused_tlet_312_or__fused_tlet_301_less_equal_fused_tlet_313_and_)
                __tlet_result = ((__tlet_arg0 <= __tlet_arg1_0) && (((__tlet_arg0_2 < __tlet_arg1_0_0) && (__tlet_arg0_0_0 > __tlet_arg1_1)) || ((__tlet_arg0_0 <= __tlet_arg1_0_1) && (__tlet_arg0_1 > __tlet_arg1))));
                ///////////////////

                __map_fusion_gtir_tmp_530 = __tlet_result;
            }

        }
        if (__map_fusion_gtir_tmp_530) {
            {
                double if_stmt_12___arg1____________________________;
                double __map_fusion_gtir_tmp_541;
                double gtir_tmp_534_0;
                double gtir_tmp_539_0;
                double gtir_tmp_531_0;

                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_314_get_value__clone_5b946252_1c88_11f1_a73a_1f4e8615a1c5)
                    __tlet_out = 1e-12;
                    ///////////////////

                    gtir_tmp_531_0 = __tlet_out;
                }
                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_319_get_value__clone_5b89034e_1c88_11f1_a73a_1f4e8615a1c5)
                    __tlet_out = 30.0;
                    ///////////////////

                    gtir_tmp_539_0 = __tlet_out;
                }
                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_316_get_value__clone_5bf877e2_1c88_11f1_a73a_1f4e8615a1c5)
                    __tlet_out = 0.0;
                    ///////////////////

                    gtir_tmp_534_0 = __tlet_out;
                }
                {
                    double __tlet_arg0 = gtir_tmp_531_0;
                    double __tlet_arg0_0 = gtir_tmp_534_0;
                    double __tlet_arg1 = gtir_tmp_539_0;
                    double __tlet_arg1_0 = __map_fusion_gtir_tmp_390;
                    double __tlet_arg1_1 = __map_fusion_gtir_tmp_102;
                    double __tlet_result;

                    ///////////////////
                    // Tasklet code (tlet_315_multiplies_fused_tlet_317_maximum_fused_tlet_318_minimum_fused_tlet_320_divides)
                    __tlet_result = (min((__tlet_arg0 * __tlet_arg1_0), max(__tlet_arg0_0, __tlet_arg1_1)) / __tlet_arg1);
                    ///////////////////

                    __map_fusion_gtir_tmp_541 = __tlet_result;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__map_fusion_gtir_tmp_541, &if_stmt_12___arg1____________________________, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &if_stmt_12___arg1____________________________, &gtir_tmp_543, 1);

            }
        } else {
            {
                double if_stmt_12___arg2__________________;
                double gtir_tmp_542_0;

                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_321_get_value__clone_5b9fdbbe_1c88_11f1_a73a_1f4e8615a1c5)
                    __tlet_out = 0.0;
                    ///////////////////

                    gtir_tmp_542_0 = __tlet_out;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_542_0, &if_stmt_12___arg2__________________, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &if_stmt_12___arg2__________________, &gtir_tmp_543, 1);

            }
        }
        {
            double __arg1____________________________;
            double __map_fusion_gtir_tmp_546;

            {
                double __tlet_arg0 = gtir_tmp_506;
                double __tlet_arg1 = gtir_tmp_543;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_322_plus)
                __tlet_result = (__tlet_arg0 + __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_546 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_546, &__arg1____________________________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1____________________________, &__output, 1);

        }
    } else {
        {
            double __arg2__________________;
            double gtir_tmp_547_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_323_get_value__clone_5b83b4fc_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_547_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_547_0, &__arg2__________________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2__________________, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_2_3_4_11(const bool&  __cond, const double*  q_in_1, const double*  q_in_2, double&  __output, int __q_in_1_Cell_range_0, int __q_in_1_Cell_stride_0_0, int __q_in_1_K_range_0, int __q_in_1_K_stride_0_0, int __q_in_2_Cell_range_0, int __q_in_2_Cell_stride_0_0, int __q_in_2_K_range_0, int __q_in_2_K_stride_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1_;
            double gtir_tmp_282_0;
            double gtir_tmp_243_0;
            double gtir_tmp_255_0;
            double gtir_tmp_237_0;
            double gtir_tmp_272_0;
            double gtir_tmp_248_0;
            double gtir_tmp_213_0;
            double __map_fusion_gtir_tmp_228;
            double gtir_tmp_263_0;
            double gtir_tmp_214_0;
            double gtir_tmp_260_0;
            double __map_fusion_gtir_tmp_288;
            double gtir_tmp_277_0;
            double gtir_tmp_221_0;
            double __map_fusion_gtir_tmp_225;
            double gtir_tmp_226_0;
            double gtir_tmp_240_0;
            double gtir_tmp_249_0;
            double gtir_tmp_252_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_155_get_value__clone_5a07221c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 3.0;
                ///////////////////

                gtir_tmp_255_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_158_get_value__clone_5a0b88fc_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1.0;
                ///////////////////

                gtir_tmp_260_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_171_get_value__clone_59e0284c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 4.0;
                ///////////////////

                gtir_tmp_282_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_145_get_value__clone_5a10dde8_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 100.0;
                ///////////////////

                gtir_tmp_240_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_143_get_value__clone_59f1f2ca_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 4.841025641025641e+18;
                ///////////////////

                gtir_tmp_237_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_133_get_value__clone_599881c2_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.9;
                ///////////////////

                gtir_tmp_221_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_151_get_value__clone_59e52676_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 600.0;
                ///////////////////

                gtir_tmp_249_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_129_get_value__clone_599346c6_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1.0;
                ///////////////////

                gtir_tmp_214_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_168_get_value__clone_59ed8d98_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 5e-05;
                ///////////////////

                gtir_tmp_277_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_128_get_value__clone_598e4ca2_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-30;
                ///////////////////

                gtir_tmp_213_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_147_get_value__clone_59fa89da_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 2.0;
                ///////////////////

                gtir_tmp_243_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_160_get_value__clone_5a152eb6_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 2.0;
                ///////////////////

                gtir_tmp_263_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_153_get_value__clone_59fec072_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1.0;
                ///////////////////

                gtir_tmp_252_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_136_get_value__clone_599d7d58_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.68;
                ///////////////////

                gtir_tmp_226_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_150_get_value__clone_59e93770_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1.0;
                ///////////////////

                gtir_tmp_248_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_165_get_value__clone_59d69e94_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 5.25;
                ///////////////////

                gtir_tmp_272_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_213_0;
                double __tlet_arg0_0 = q_in_1[((__q_in_1_Cell_stride_0_0 * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride_0_0 * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_1 = q_in_1[((__q_in_1_Cell_stride_0_0 * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride_0_0 * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_2 = gtir_tmp_214_0;
                double __tlet_arg1 = gtir_tmp_221_0;
                double __tlet_arg1_0 = q_in_2[((__q_in_2_Cell_stride_0_0 * ((- __q_in_2_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_2_K_stride_0_0 * ((- __q_in_2_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_130_plus_fused_tlet_131_divides_fused_tlet_132_minus_fused_tlet_134_minimum_fused_tlet_135_maximum)
                __tlet_result = max(__tlet_arg0, min((__tlet_arg0_2 - (__tlet_arg0_0 / (__tlet_arg0_1 + __tlet_arg1_0))), __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_225 = __tlet_result;
            }
            {
                double __tlet_arg0 = __map_fusion_gtir_tmp_225;
                double __tlet_arg1 = gtir_tmp_226_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_137_power)
                __tlet_result = dace::math::pow(__tlet_arg0, __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_228 = __tlet_result;
            }
            {
                double __tlet_arg0 = gtir_tmp_237_0;
                double __tlet_arg0_0 = gtir_tmp_272_0;
                double __tlet_arg0_0_0 = q_in_1[((__q_in_1_Cell_stride_0_0 * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride_0_0 * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0_1 = __map_fusion_gtir_tmp_225;
                double __tlet_arg0_0_2 = gtir_tmp_252_0;
                double __tlet_arg0_1 = __map_fusion_gtir_tmp_225;
                double __tlet_arg0_1_0 = gtir_tmp_260_0;
                double __tlet_arg0_2 = gtir_tmp_249_0;
                double __tlet_arg0_3 = gtir_tmp_248_0;
                double __tlet_arg1 = gtir_tmp_282_0;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_225;
                double __tlet_arg1_0_0 = q_in_1[((__q_in_1_Cell_stride_0_0 * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride_0_0 * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1_0_0_0 = __map_fusion_gtir_tmp_228;
                double __tlet_arg1_0_1 = gtir_tmp_240_0;
                double __tlet_arg1_0_2 = gtir_tmp_277_0;
                double __tlet_arg1_1 = q_in_2[((__q_in_2_Cell_stride_0_0 * ((- __q_in_2_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_2_K_stride_0_0 * ((- __q_in_2_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1_1_0 = q_in_1[((__q_in_1_Cell_stride_0_0 * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride_0_0 * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1_1_1 = __map_fusion_gtir_tmp_228;
                double __tlet_arg1_2 = gtir_tmp_243_0;
                double __tlet_arg1_2_0 = gtir_tmp_255_0;
                double __tlet_arg1_3 = gtir_tmp_263_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_152_multiplies_fused_tlet_154_minus_fused_tlet_156_power_fused_tlet_157_multiplies_fused_tlet_159_minus_fused_tlet_161_power_fused_tlet_162_divides_fused_tlet_163_plus_fused_tlet_169_plus_fused_tlet_170_divides_fused_tlet_172_power_fused_tlet_144_multiplies_fused_tlet_146_divides_fused_tlet_148_power_fused_tlet_149_multiplies_fused_tlet_166_multiplies_fused_tlet_167_multiplies_fused_tlet_164_multiplies_fused_tlet_173_multiplies_fused_tlet_174_plus)
                __tlet_result = (((__tlet_arg0 * dace::math::pow(((__tlet_arg0_0_0 * __tlet_arg1_1_0) / __tlet_arg1_0_1), __tlet_arg1_2)) * (__tlet_arg0_3 + (((__tlet_arg0_2 * __tlet_arg1_1_1) * dace::math::pow((__tlet_arg0_0_2 - __tlet_arg1_0_0_0), __tlet_arg1_2_0)) / dace::math::pow((__tlet_arg0_1_0 - __tlet_arg1_0), __tlet_arg1_3)))) + (((__tlet_arg0_0 * __tlet_arg1_0_0) * __tlet_arg1_1) * dace::math::pow((__tlet_arg0_1 / (__tlet_arg0_0_1 + __tlet_arg1_0_2)), __tlet_arg1)));
                ///////////////////

                __map_fusion_gtir_tmp_288 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_288, &__arg1_, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_, &__output, 1);

        }
    } else {
        {
            double __arg2;
            double gtir_tmp_289_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_175_get_value__clone_5a02f502_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_289_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_289_0, &__arg2, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_6_3_4_25(const bool&  __cond, const double*  q_in_1, const double*  q_in_5, const double* __restrict__ rho, double&  __output, int __q_in_1_Cell_range_0, int __q_in_1_Cell_stride_0_0, int __q_in_1_K_range_0, int __q_in_1_K_stride_0_0, int __q_in_5_Cell_range_0, int __q_in_5_Cell_stride_0_0, int __q_in_5_K_range_0, int __q_in_5_K_stride_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1________________;
            double __map_fusion_gtir_tmp_371;
            double gtir_tmp_362_0;
            double gtir_tmp_367_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_219_get_value__clone_5a99d60c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.94878;
                ///////////////////

                gtir_tmp_367_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_216_get_value__clone_5aa88008_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 4.43;
                ///////////////////

                gtir_tmp_362_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_362_0;
                double __tlet_arg0_0 = q_in_5[((__q_in_5_Cell_stride_0_0 * ((- __q_in_5_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_5_K_stride_0_0 * ((- __q_in_5_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_367_0;
                double __tlet_arg1_0 = q_in_1[((__q_in_1_Cell_stride_0_0 * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride_0_0 * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1_0_0 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_218_multiplies_fused_tlet_220_power_fused_tlet_217_multiplies_fused_tlet_221_multiplies)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) * dace::math::pow((__tlet_arg0_0 * __tlet_arg1_0_0), __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_371 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_371, &__arg1________________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1________________, &__output, 1);

        }
    } else {
        {
            double __arg2____________;
            double gtir_tmp_372_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_222_get_value__clone_5a9ecae0_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_372_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_372_0, &__arg2____________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2____________, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_0_3_4_41(const bool&  __cond, const double*  q_in_3, const double* __restrict__ rho, const double*  te, double&  __output, int __q_in_3_Cell_range_0, int __q_in_3_Cell_stride_0_0, int __q_in_3_K_range_0, int __q_in_3_K_stride_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1_;
            double gtir_tmp_156_0;
            double gtir_tmp_157_0;
            double gtir_tmp_126_0;
            double gtir_tmp_103_0;
            double gtir_tmp_152_0;
            double gtir_tmp_109_0;
            double __map_fusion_gtir_tmp_130;
            double gtir_tmp_113_0;
            double gtir_tmp_115_0;
            double gtir_tmp_159_0;
            double gtir_tmp_137_0;
            double gtir_tmp_155_0;
            double gtir_tmp_143_0;
            double gtir_tmp_146_0;
            double __map_fusion_gtir_tmp_111;
            double __map_fusion_gtir_tmp_185;
            double gtir_tmp_147_0;
            double gtir_tmp_112_0;
            double gtir_tmp_106_0;
            double __map_fusion_gtir_tmp_125;
            double gtir_tmp_140_0;
            double gtir_tmp_134_0;
            double gtir_tmp_158_0;
            double gtir_tmp_114_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_63_get_value__clone_58e0859a_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 233.14999999999998;
                ///////////////////

                gtir_tmp_106_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_97_get_value__clone_5942b90e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1.42;
                ///////////////////

                gtir_tmp_157_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_70_get_value__clone_58e927ea_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.000327;
                ///////////////////

                gtir_tmp_115_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_93_get_value__clone_5939cb96_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.069;
                ///////////////////

                gtir_tmp_152_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_67_get_value__clone_58f26fb2_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 10.0;
                ///////////////////

                gtir_tmp_112_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_87_get_value__clone_5930b8e4_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1000000.0;
                ///////////////////

                gtir_tmp_143_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_61_get_value__clone_58d77aa4_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_103_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_81_get_value__clone_5964e222_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 762750000.0;
                ///////////////////

                gtir_tmp_134_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_83_get_value__clone_5956ba12_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1000000000.0;
                ///////////////////

                gtir_tmp_137_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_95_get_value__clone_593e25c4_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 4.0;
                ///////////////////

                gtir_tmp_155_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_98_get_value__clone_59604334_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0119;
                ///////////////////

                gtir_tmp_158_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_68_get_value__clone_58ed730e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = -1.65;
                ///////////////////

                gtir_tmp_113_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_69_get_value__clone_58e4cde4_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0545;
                ///////////////////

                gtir_tmp_114_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_96_get_value__clone_596e443e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 3.0;
                ///////////////////

                gtir_tmp_156_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_99_get_value__clone_5969e128_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 9.6e-05;
                ///////////////////

                gtir_tmp_159_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_85_get_value__clone_59480a1c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 3813750.0;
                ///////////////////

                gtir_tmp_140_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_90_get_value__clone_5952047c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 2e-06;
                ///////////////////

                gtir_tmp_147_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_89_get_value__clone_595b59be_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 13.5;
                ///////////////////

                gtir_tmp_146_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_65_get_value__clone_58dbe152_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_109_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_76_get_value__clone_58f7728c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = -0.107;
                ///////////////////

                gtir_tmp_126_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1 = gtir_tmp_109_0;
                double __tlet_arg1_0 = gtir_tmp_106_0;
                double __tlet_arg1_1 = gtir_tmp_103_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_62_minimum_fused_tlet_64_maximum_fused_tlet_66_minus)
                __tlet_result = (max(min(__tlet_arg0, __tlet_arg1_1), __tlet_arg1_0) - __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_111 = __tlet_result;
            }
            {
                double __tlet_arg0 = gtir_tmp_112_0;
                double __tlet_arg0_0 = gtir_tmp_113_0;
                double __tlet_arg0_0_0 = gtir_tmp_114_0;
                double __tlet_arg0_1 = __map_fusion_gtir_tmp_111;
                double __tlet_arg1 = gtir_tmp_115_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_71_multiplies_fused_tlet_72_plus_fused_tlet_73_multiplies_fused_tlet_74_plus_fused_tlet_75_power)
                __tlet_result = dace::math::pow(__tlet_arg0, (__tlet_arg0_0 + (__tlet_arg0_1 * (__tlet_arg0_0_0 + (__tlet_arg0_1 * __tlet_arg1)))));
                ///////////////////

                __map_fusion_gtir_tmp_125 = __tlet_result;
            }
            {
                double __tlet_arg0 = gtir_tmp_126_0;
                double __tlet_arg1 = __map_fusion_gtir_tmp_111;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_77_multiplies_fused_tlet_78_exp)
                __tlet_result = dace::math::exp((__tlet_arg0 * __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_130 = __tlet_result;
            }
            {
                double __tlet_arg0 = gtir_tmp_134_0;
                double __tlet_arg0_0 = gtir_tmp_146_0;
                double __tlet_arg0_0_0 = __map_fusion_gtir_tmp_111;
                double __tlet_arg0_0_1 = gtir_tmp_155_0;
                double __tlet_arg0_1 = __map_fusion_gtir_tmp_125;
                double __tlet_arg0_1_0 = gtir_tmp_158_0;
                double __tlet_arg0_1_1 = gtir_tmp_156_0;
                double __tlet_arg0_2 = gtir_tmp_140_0;
                double __tlet_arg0_2_0 = __map_fusion_gtir_tmp_111;
                double __tlet_arg0_3 = gtir_tmp_157_0;
                double __tlet_arg0_4 = q_in_3[((__q_in_3_Cell_stride_0_0 * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride_0_0 * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = __map_fusion_gtir_tmp_125;
                double __tlet_arg1_0 = gtir_tmp_137_0;
                double __tlet_arg1_0_0 = __map_fusion_gtir_tmp_125;
                double __tlet_arg1_0_0_0 = gtir_tmp_147_0;
                double __tlet_arg1_0_1 = gtir_tmp_143_0;
                double __tlet_arg1_0_2 = gtir_tmp_152_0;
                double __tlet_arg1_1 = __map_fusion_gtir_tmp_130;
                double __tlet_arg1_1_0 = __map_fusion_gtir_tmp_130;
                double __tlet_arg1_1_1 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_2 = gtir_tmp_159_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_100_multiplies_fused_tlet_101_plus_fused_tlet_102_multiplies_fused_tlet_103_plus_fused_tlet_91_plus_fused_tlet_92_multiplies_fused_tlet_104_multiplies_fused_tlet_94_divides_fused_tlet_105_minus_fused_tlet_106_power_fused_tlet_108_multiplies_fused_tlet_109_multiplies_fused_tlet_107_multiplies_fused_tlet_86_multiplies_fused_tlet_110_divides_fused_tlet_88_maximum_fused_tlet_111_maximum_fused_tlet_82_multiplies_fused_tlet_84_minimum_fused_tlet_112_minimum)
                __tlet_result = min(min((__tlet_arg0 * __tlet_arg1_1), __tlet_arg1_0), max(max((__tlet_arg0_2 * __tlet_arg1_1_0), __tlet_arg1_0_1), ((__tlet_arg0_0 * dace::math::pow((((__tlet_arg0_4 + __tlet_arg1_0_0_0) * __tlet_arg1_1_1) / __tlet_arg1_0_2), (__tlet_arg0_0_1 - (__tlet_arg0_1_1 * (__tlet_arg0_3 + (__tlet_arg0_0_0 * (__tlet_arg0_1_0 + (__tlet_arg0_2_0 * __tlet_arg1_2)))))))) / ((__tlet_arg0_1 * __tlet_arg1_0_0) * __tlet_arg1))));
                ///////////////////

                __map_fusion_gtir_tmp_185 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_185, &__arg1_, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_, &__output, 1);

        }
    } else {
        {
            double __arg2;
            double gtir_tmp_186_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_113_get_value__clone_5935698e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 800000.0;
                ///////////////////

                gtir_tmp_186_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_186_0, &__arg2, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_1_3_4_9(const bool&  __cond, const double&  gtir_tmp_187, const double*  q_in_3, const double* __restrict__ rho, double&  __output, int __q_in_3_Cell_range_0, int __q_in_3_Cell_stride_0_0, int __q_in_3_K_range_0, int __q_in_3_K_stride_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1_________________;
            double gtir_tmp_202_0;
            double gtir_tmp_195_0;
            double gtir_tmp_192_0;
            double __map_fusion_gtir_tmp_204;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_122_get_value__clone_597716cc_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.3333333333333333;
                ///////////////////

                gtir_tmp_202_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_118_get_value__clone_5984fabc_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_195_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_116_get_value__clone_5972c28e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.138;
                ///////////////////

                gtir_tmp_192_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_192_0;
                double __tlet_arg0_0 = q_in_3[((__q_in_3_Cell_stride_0_0 * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride_0_0 * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_202_0;
                double __tlet_arg1_0 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_1 = gtir_tmp_187;
                double __tlet_arg1_2 = gtir_tmp_195_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_119_plus_fused_tlet_120_multiplies_fused_tlet_117_multiplies_fused_tlet_121_divides_fused_tlet_123_power)
                __tlet_result = dace::math::pow(((__tlet_arg0 * __tlet_arg1_1) / ((__tlet_arg0_0 + __tlet_arg1_2) * __tlet_arg1_0)), __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_204 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_204, &__arg1_________________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_________________, &__output, 1);

        }
    } else {
        {
            double __arg2;
            double gtir_tmp_205_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_124_get_value__clone_597c236a_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 10000000000.0;
                ///////////////////

                gtir_tmp_205_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_205_0, &__arg2, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_5_3_4_27(const bool&  __cond, const double&  gtir_tmp_187, const double&  gtir_tmp_206, const double*  q_in_1, double&  __output, int __q_in_1_Cell_range_0, int __q_in_1_Cell_stride_0_0, int __q_in_1_K_range_0, int __q_in_1_K_stride_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1________________;
            double gtir_tmp_344_0;
            double __map_fusion_gtir_tmp_348;
            double gtir_tmp_339_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_203_get_value__clone_5a7b82e2_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 58.724999999999994;
                ///////////////////

                gtir_tmp_339_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_206_get_value__clone_5a85174e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = -3.5;
                ///////////////////

                gtir_tmp_344_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_339_0;
                double __tlet_arg0_0 = gtir_tmp_206;
                double __tlet_arg1 = gtir_tmp_344_0;
                double __tlet_arg1_0 = q_in_1[((__q_in_1_Cell_stride_0_0 * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride_0_0 * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1_1 = gtir_tmp_187;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_204_multiplies_fused_tlet_207_power_fused_tlet_205_multiplies_fused_tlet_208_multiplies)
                __tlet_result = (((__tlet_arg0 * __tlet_arg1_1) * __tlet_arg1_0) * dace::math::pow(__tlet_arg0_0, __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_348 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_348, &__arg1________________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1________________, &__output, 1);

        }
    } else {
        {
            double __arg2____________;
            double gtir_tmp_349_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_209_get_value__clone_5a805c90_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_349_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_349_0, &__arg2____________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2____________, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_40_3_5_103(const bool&  __cond, const double&  __map_fusion_gtir_tmp_77, const double*  q_in_2, const double* __restrict__ rho, const double*  te, double&  __output, int __q_in_2_Cell_range_0, int __q_in_2_Cell_stride_0_0, int __q_in_2_K_range_0, int __q_in_2_K_stride_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1_;
            double gtir_tmp_973_0;
            double gtir_tmp_953_0;
            double gtir_tmp_958_0;
            double gtir_tmp_935_0;
            double gtir_tmp_933_0;
            double gtir_tmp_959_0;
            double gtir_tmp_938_0;
            double __map_fusion_gtir_tmp_977;
            double gtir_tmp_920_0;
            double gtir_tmp_934_0;
            double __map_fusion_gtir_tmp_922;
            double gtir_tmp_960_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_550_get_value__clone_5f2acb5e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_973_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_527_get_value__clone_5f18146e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1.0;
                ///////////////////

                gtir_tmp_934_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_542_get_value__clone_5eb429fe_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = -0.0163;
                ///////////////////

                gtir_tmp_959_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_538_get_value__clone_5f11a052_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.55555;
                ///////////////////

                gtir_tmp_953_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_541_get_value__clone_5f237520_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.61;
                ///////////////////

                gtir_tmp_958_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_543_get_value__clone_5eba0fc2_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0001111;
                ///////////////////

                gtir_tmp_960_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_518_get_value__clone_5e42fe28_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_920_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_530_get_value__clone_5ebf8ac4_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.16667;
                ///////////////////

                gtir_tmp_938_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_528_get_value__clone_5ec59ea0_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 19.0621;
                ///////////////////

                gtir_tmp_935_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_526_get_value__clone_5f318e08_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.001536;
                ///////////////////

                gtir_tmp_933_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1 = gtir_tmp_920_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_519_minus)
                __tlet_result = (__tlet_arg0 - __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_922 = __tlet_result;
            }
            {
                double __tlet_arg0 = gtir_tmp_933_0;
                double __tlet_arg0_0 = __map_fusion_gtir_tmp_77;
                double __tlet_arg0_0_0 = gtir_tmp_934_0;
                double __tlet_arg0_0_0_0 = gtir_tmp_959_0;
                double __tlet_arg0_0_1 = __map_fusion_gtir_tmp_77;
                double __tlet_arg0_1 = q_in_2[((__q_in_2_Cell_stride_0_0 * ((- __q_in_2_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_2_K_stride_0_0 * ((- __q_in_2_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_1_0 = gtir_tmp_935_0;
                double __tlet_arg0_1_1 = gtir_tmp_960_0;
                double __tlet_arg0_2 = q_in_2[((__q_in_2_Cell_stride_0_0 * ((- __q_in_2_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_2_K_stride_0_0 * ((- __q_in_2_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_2_0 = __map_fusion_gtir_tmp_922;
                double __tlet_arg0_3 = gtir_tmp_958_0;
                double __tlet_arg1 = gtir_tmp_973_0;
                double __tlet_arg1_0 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_0_0 = __map_fusion_gtir_tmp_922;
                double __tlet_arg1_1 = gtir_tmp_953_0;
                double __tlet_arg1_2 = gtir_tmp_938_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_544_multiplies_fused_tlet_545_plus_fused_tlet_546_multiplies_fused_tlet_547_plus_fused_tlet_548_neg_fused_tlet_549_multiplies_fused_tlet_551_divides_fused_tlet_529_multiplies_fused_tlet_531_power_fused_tlet_532_multiplies_fused_tlet_533_plus_fused_tlet_534_multiplies_fused_tlet_537_multiplies_fused_tlet_539_power_fused_tlet_535_neg_fused_tlet_536_multiplies_fused_tlet_540_multiplies_fused_tlet_552_minimum)
                __tlet_result = min((((__tlet_arg0 * (__tlet_arg0_0_0 + (__tlet_arg0_1_0 * dace::math::pow((__tlet_arg0_2 * __tlet_arg1_0), __tlet_arg1_2)))) * (- __tlet_arg0_0)) * dace::math::pow((__tlet_arg0_1 * __tlet_arg1_0), __tlet_arg1_1)), (((__tlet_arg0_3 + (__tlet_arg0_2_0 * (__tlet_arg0_0_0_0 + (__tlet_arg0_1_1 * __tlet_arg1_0_0)))) * (- __tlet_arg0_0_1)) / __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_977 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_977, &__arg1_, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_, &__output, 1);

        }
    } else {
        {
            double __arg2;
            double gtir_tmp_978_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_553_get_value__clone_5f1da0fa_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_978_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_978_0, &__arg2, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_3_79_0_6(const bool&  __cond, const double*  q_in_1, double&  __output, int __q_in_1_Cell_range_0, int __q_in_1_Cell_stride_0_0_0, int __q_in_1_K_range_0, int __q_in_1_K_stride_0_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1________________;
            double gtir_tmp_313_0;
            double __map_fusion_gtir_tmp_315;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_189_get_value__clone_5a2d3cfe_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_313_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_1[((__q_in_1_Cell_stride_0_0_0 * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride_0_0_0 * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_313_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_190_divides)
                __tlet_result = (__tlet_arg0 / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_315 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_315, &__arg1________________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1________________, &__output, 1);

        }
    } else {
        {
            double __arg2____________;
            double gtir_tmp_316_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_191_get_value__clone_5a286fee_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_316_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_316_0, &__arg2____________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2____________, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_4_3_5_71(const bool&  __cond, const double*  q_in_1, const double*  q_in_4, const double*  te, double&  __output, int __q_in_1_Cell_range_0, int __q_in_1_Cell_stride_0_0, int __q_in_1_K_range_0, int __q_in_1_K_stride_0_0, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride_0_0, int __q_in_4_K_range_0, int __q_in_4_K_stride_0_0, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1_______________;
            double __map_fusion_gtir_tmp_304;
            double gtir_tmp_302_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_182_get_value__clone_5a3226b0_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_302_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_4[((__q_in_4_Cell_stride_0_0 * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride_0_0 * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_302_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_181_neg_fused_tlet_183_divides)
                __tlet_result = ((- __tlet_arg0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_304 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_304, &__arg1_______________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_______________, &__output, 1);

        }
    } else {
        {
            double __arg2_;
            double gtir_tmp_308_0;
            bool __map_fusion_gtir_tmp_312;
            double gtir_tmp_305_0;
            double gtir_tmp_317;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_184_get_value__clone_5a1e88b2_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_305_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_186_get_value__clone_5a377fca_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 236.15;
                ///////////////////

                gtir_tmp_308_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_1[((__q_in_1_Cell_stride_0_0 * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride_0_0 * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1 = gtir_tmp_308_0;
                double __tlet_arg1_0 = gtir_tmp_305_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_187_less_fused_tlet_185_greater_fused_tlet_188_and_)
                __tlet_result = ((__tlet_arg0 > __tlet_arg1_0) && (__tlet_arg0_0 < __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_312 = __tlet_result;
            }
            if_stmt_3_79_0_6(__map_fusion_gtir_tmp_312, &q_in_1[0], gtir_tmp_317, __q_in_1_Cell_range_0, __q_in_1_Cell_stride_0_0, __q_in_1_K_range_0, __q_in_1_K_stride_0_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_317, &__arg2_, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_59_184_0_4(const bool&  __cond, const double&  __map_fusion_gtir_tmp_420, const double*  q_in_4, const double*  q_in_5, const double* __restrict__ rho, double&  __output, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride_0_0_0, int __q_in_4_K_range_0, int __q_in_4_K_stride_0_0_0, int __q_in_5_Cell_range_0, int __q_in_5_Cell_stride_0_0_0, int __q_in_5_K_range_0, int __q_in_5_K_stride_0_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1___________;
            double __map_fusion_gtir_tmp_1296;
            double gtir_tmp_1287_0;
            double gtir_tmp_1292_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_720_get_value__clone_601f2370_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.94878;
                ///////////////////

                gtir_tmp_1292_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_717_get_value__clone_601836b4_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 2.46;
                ///////////////////

                gtir_tmp_1287_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = __map_fusion_gtir_tmp_420;
                double __tlet_arg0_0 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0_0_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1 = gtir_tmp_1292_0;
                double __tlet_arg1_0 = q_in_5[((__q_in_5_Cell_stride_0_0_0 * ((- __q_in_5_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_5_K_stride_0_0_0 * ((- __q_in_5_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1_0_0 = q_in_4[((__q_in_4_Cell_stride_0_0_0 * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride_0_0_0 * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1_1 = gtir_tmp_1287_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_716_multiplies_fused_tlet_718_multiplies_fused_tlet_719_multiplies_fused_tlet_721_power_fused_tlet_722_multiplies)
                __tlet_result = (((__tlet_arg0 * __tlet_arg1_0_0) * __tlet_arg1_1) * dace::math::pow((__tlet_arg0_0 * __tlet_arg1_0), __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_1296 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1296, &__arg1___________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1___________, &__output, 1);

        }
    } else {
        {
            double __arg2________;
            double gtir_tmp_1297_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_723_get_value__clone_600d4876_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1297_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1297_0, &__arg2________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2________, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_60_184_0_14(const double&  __arg2________, const bool&  __cond, const double&  gtir_tmp_1298, const double*  q_in_2, const double*  q_in_4, const double* __restrict__ rho, double&  __output, int __q_in_2_Cell_range_0, int __q_in_2_Cell_stride_0_0_0, int __q_in_2_K_range_0, int __q_in_2_K_stride_0_0_0, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride_0_0_0, int __q_in_4_K_range_0, int __q_in_4_K_stride_0_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1___________;
            double __map_fusion_gtir_tmp_1319;
            double gtir_tmp_1313_0;
            double gtir_tmp_1308_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_729_get_value__clone_603693de_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1.72;
                ///////////////////

                gtir_tmp_1308_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_732_get_value__clone_603c05bc_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.875;
                ///////////////////

                gtir_tmp_1313_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_1298;
                double __tlet_arg0_0 = gtir_tmp_1308_0;
                double __tlet_arg0_1 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0_0_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1 = gtir_tmp_1313_0;
                double __tlet_arg1_0 = q_in_4[((__q_in_4_Cell_stride_0_0_0 * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride_0_0_0 * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1_0_0 = q_in_2[((__q_in_2_Cell_stride_0_0_0 * ((- __q_in_2_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_2_K_stride_0_0_0 * ((- __q_in_2_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_731_multiplies_fused_tlet_733_power_fused_tlet_730_multiplies_fused_tlet_734_multiplies_fused_tlet_735_plus)
                __tlet_result = (__tlet_arg0 + ((__tlet_arg0_0 * __tlet_arg1_0) * dace::math::pow((__tlet_arg0_1 * __tlet_arg1_0_0), __tlet_arg1)));
                ///////////////////

                __map_fusion_gtir_tmp_1319 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1319, &__arg1___________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1___________, &__output, 1);

        }
    } else {
        {


            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2________, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_61_3_5_210(const bool&  __cond, const double&  __map_fusion_gtir_tmp_420, const double*  q_in_2, const double*  q_in_4, const double*  q_in_5, const double* __restrict__ rho, double&  __output, int __q_in_2_Cell_range_0, int __q_in_2_Cell_stride_0_0, int __q_in_2_K_range_0, int __q_in_2_K_stride_0_0, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride_0_0, int __q_in_4_K_range_0, int __q_in_4_K_stride_0_0, int __q_in_5_Cell_range_0, int __q_in_5_Cell_stride_0_0, int __q_in_5_K_range_0, int __q_in_5_K_stride_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1___________;
            double gtir_tmp_1280_0;
            double gtir_tmp_1320;
            double gtir_tmp_1277_0;
            bool __map_fusion_gtir_tmp_1284;
            double gtir_tmp_1298;
            double gtir_tmp_1303_0;
            double gtir_tmp_1300_0;
            bool __map_fusion_gtir_tmp_1307;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_724_get_value__clone_602b4de4_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_1300_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_713_get_value__clone_6012aa28_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_1280_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_726_get_value__clone_6031329a_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_1303_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_711_get_value__clone_60256168_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_1277_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_4[((__q_in_4_Cell_stride_0_0 * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride_0_0 * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0 = q_in_2[((__q_in_2_Cell_stride_0_0 * ((- __q_in_2_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_2_K_stride_0_0 * ((- __q_in_2_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_1303_0;
                double __tlet_arg1_0 = gtir_tmp_1300_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_727_greater_fused_tlet_725_greater_fused_tlet_728_and_)
                __tlet_result = ((__tlet_arg0 > __tlet_arg1_0) && (__tlet_arg0_0 > __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_1307 = __tlet_result;
            }
            {
                double __tlet_arg0 = q_in_4[((__q_in_4_Cell_stride_0_0 * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride_0_0 * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0 = q_in_5[((__q_in_5_Cell_stride_0_0 * ((- __q_in_5_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_5_K_stride_0_0 * ((- __q_in_5_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_1280_0;
                double __tlet_arg1_0 = gtir_tmp_1277_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_714_greater_fused_tlet_712_greater_fused_tlet_715_and_)
                __tlet_result = ((__tlet_arg0 > __tlet_arg1_0) && (__tlet_arg0_0 > __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_1284 = __tlet_result;
            }
            if_stmt_59_184_0_4(__map_fusion_gtir_tmp_1284, __map_fusion_gtir_tmp_420, &q_in_4[0], &q_in_5[0], &rho[0], gtir_tmp_1298, __q_in_4_Cell_range_0, __q_in_4_Cell_stride_0_0, __q_in_4_K_range_0, __q_in_4_K_stride_0_0, __q_in_5_Cell_range_0, __q_in_5_Cell_stride_0_0, __q_in_5_K_range_0, __q_in_5_K_stride_0_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);
            if_stmt_60_184_0_14(gtir_tmp_1298, __map_fusion_gtir_tmp_1307, gtir_tmp_1298, &q_in_2[0], &q_in_4[0], &rho[0], gtir_tmp_1320, __q_in_2_Cell_range_0, __q_in_2_Cell_stride_0_0, __q_in_2_K_range_0, __q_in_2_K_stride_0_0, __q_in_4_Cell_range_0, __q_in_4_Cell_stride_0_0, __q_in_4_K_range_0, __q_in_4_K_stride_0_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1320, &__arg1___________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1___________, &__output, 1);

        }
    } else {
        {
            double __arg2________;
            double gtir_tmp_1323_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_736_get_value__clone_60aaf4d6_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1323_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1323_0, &__arg2________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2________, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_50_128_0_5(const bool&  __cond, const double*  q_in_1, const double*  q_in_3, const double* __restrict__ rho, double&  __output, int __q_in_1_Cell_range_0, int __q_in_1_Cell_stride_0_0_0, int __q_in_1_K_range_0, int __q_in_1_K_stride_0_0_0, int __q_in_3_Cell_range_0, int __q_in_3_Cell_stride_0_0_0, int __q_in_3_K_range_0, int __q_in_3_K_stride_0_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1__________;
            double __map_fusion_gtir_tmp_1172;
            double gtir_tmp_1168_0;
            double gtir_tmp_1163_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_655_get_value__clone_5f5ea992_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.75;
                ///////////////////

                gtir_tmp_1168_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_652_get_value__clone_5f7cf5e6_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.5;
                ///////////////////

                gtir_tmp_1163_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_1163_0;
                double __tlet_arg0_0 = q_in_3[((__q_in_3_Cell_stride_0_0_0 * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride_0_0_0 * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_1168_0;
                double __tlet_arg1_0 = q_in_1[((__q_in_1_Cell_stride_0_0_0 * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride_0_0_0 * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1_0_0 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0_0_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_654_multiplies_fused_tlet_656_power_fused_tlet_653_multiplies_fused_tlet_657_multiplies)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) * dace::math::pow((__tlet_arg0_0 * __tlet_arg1_0_0), __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_1172 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1172, &__arg1__________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1__________, &__output, 1);

        }
    } else {
        {
            double __arg2_______;
            double gtir_tmp_1173_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_658_get_value__clone_5f6cae70_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1173_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1173_0, &__arg2_______, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_______, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_51_3_5_126(const bool&  __cond, const double*  q_in_1, const double*  q_in_3, const double* __restrict__ rho, const double*  te, double&  __output, int __q_in_1_Cell_range_0, int __q_in_1_Cell_stride_0_0, int __q_in_1_K_range_0, int __q_in_1_K_stride_0_0, int __q_in_3_Cell_range_0, int __q_in_3_Cell_stride_0_0, int __q_in_3_K_range_0, int __q_in_3_K_stride_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1__________;
            double gtir_tmp_1155_0;
            bool __map_fusion_gtir_tmp_1162;
            double gtir_tmp_1174;
            double gtir_tmp_1158_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_649_get_value__clone_5f9095b0_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 236.15;
                ///////////////////

                gtir_tmp_1158_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_647_get_value__clone_5ff63686_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_1155_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_1[((__q_in_1_Cell_stride_0_0 * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride_0_0 * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1 = gtir_tmp_1158_0;
                double __tlet_arg1_0 = gtir_tmp_1155_0;
                double __tlet_arg1_1 = q_in_3[((__q_in_3_Cell_stride_0_0 * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride_0_0 * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))];
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_646_minimum_fused_tlet_650_greater_fused_tlet_648_greater_fused_tlet_651_and_)
                __tlet_result = ((min(__tlet_arg0, __tlet_arg1_1) > __tlet_arg1_0) && (__tlet_arg0_0 > __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_1162 = __tlet_result;
            }
            if_stmt_50_128_0_5(__map_fusion_gtir_tmp_1162, &q_in_1[0], &q_in_3[0], &rho[0], gtir_tmp_1174, __q_in_1_Cell_range_0, __q_in_1_Cell_stride_0_0, __q_in_1_K_range_0, __q_in_1_K_stride_0_0, __q_in_3_Cell_range_0, __q_in_3_Cell_stride_0_0, __q_in_3_K_range_0, __q_in_3_K_stride_0_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1174, &__arg1__________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1__________, &__output, 1);

        }
    } else {
        {
            double __arg2_______;
            double gtir_tmp_1176_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_659_get_value__clone_5f760f2e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1176_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1176_0, &__arg2_______, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_______, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_15_94_0_12(const bool&  __cond, const double&  __map_fusion_gtir_tmp_102, const double&  gtir_tmp_187, const double&  gtir_tmp_206, const double&  gtir_tmp_448, const double* __restrict__ rho, double&  __output, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0_0_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1________;
            double gtir_tmp_594_0;
            double gtir_tmp_580_0;
            double __map_fusion_gtir_tmp_598;
            double gtir_tmp_572_0;
            double gtir_tmp_579_0;
            double gtir_tmp_581_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_350_get_value__clone_5c370ad4_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_594_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_343_get_value__clone_5c2b5522_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = -0.75;
                ///////////////////

                gtir_tmp_581_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_342_get_value__clone_5c25828c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 499.8446044236435;
                ///////////////////

                gtir_tmp_580_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_337_get_value__clone_5c3113cc_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 4.0;
                ///////////////////

                gtir_tmp_572_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_341_get_value__clone_5c43346c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1.0;
                ///////////////////

                gtir_tmp_579_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_572_0;
                double __tlet_arg0_0 = gtir_tmp_206;
                double __tlet_arg0_0_0 = gtir_tmp_580_0;
                double __tlet_arg0_1 = gtir_tmp_206;
                double __tlet_arg0_2 = gtir_tmp_579_0;
                double __tlet_arg1 = gtir_tmp_594_0;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_102;
                double __tlet_arg1_0_0 = gtir_tmp_187;
                double __tlet_arg1_1 = gtir_tmp_581_0;
                double __tlet_arg1_2 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0_0_0_0_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_3 = gtir_tmp_206;
                double __tlet_arg1_4 = gtir_tmp_448;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_338_multiplies_fused_tlet_339_multiplies_fused_tlet_344_power_fused_tlet_345_multiplies_fused_tlet_346_plus_fused_tlet_349_multiplies_fused_tlet_340_divides_fused_tlet_347_multiplies_fused_tlet_348_multiplies_fused_tlet_351_plus_fused_tlet_352_divides)
                __tlet_result = ((((((__tlet_arg0 * __tlet_arg1_0_0) * __tlet_arg1_4) / __tlet_arg1_2) * (__tlet_arg0_2 + (__tlet_arg0_0_0 * dace::math::pow(__tlet_arg0_1, __tlet_arg1_1)))) * __tlet_arg1_0) / ((__tlet_arg0_0 * __tlet_arg1_3) + __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_598 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_598, &__arg1________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1________, &__output, 1);

        }
    } else {
        {
            double __arg2____;
            double gtir_tmp_599_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_353_get_value__clone_5c1fe82c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_599_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_599_0, &__arg2____, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2____, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_18_90_0_5(const bool&  __cond, const double&  __map_fusion_gtir_tmp_102, const double&  __map_fusion_gtir_tmp_568, const double&  gtir_tmp_187, const double&  gtir_tmp_206, const double&  gtir_tmp_448, const double&  gtir_tmp_556, const double* __restrict__ p, const double*  q_in_3, const double* __restrict__ rho, const double*  te, double&  __output, int __p_Cell_range_0, int __p_K_range_0, int __p_K_stride_0_0_0_0, int __q_in_3_Cell_range_0, int __q_in_3_Cell_stride_0_0_0_0, int __q_in_3_K_range_0, int __q_in_3_K_stride_0_0_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0_0_0, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride_0_0_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {
    double gtir_tmp_692;
    double gtir_tmp_600;
    double gtir_tmp_617;
    bool __map_fusion_gtir_tmp_688;
    bool __map_fusion_gtir_tmp_609;

    if (__cond) {
        {
            double __arg1___;
            double gtir_tmp_663_0;
            double gtir_tmp_666_0;
            double gtir_tmp_676_0;
            double gtir_tmp_669_0;
            double __map_fusion_gtir_tmp_680;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_398_get_value__clone_5ce75b1e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.8;
                ///////////////////

                gtir_tmp_676_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_392_get_value__clone_5c7a48bc_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.241897;
                ///////////////////

                gtir_tmp_666_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_394_get_value__clone_5cfa4990_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_669_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_390_get_value__clone_5c64663c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 31282.3;
                ///////////////////

                gtir_tmp_663_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_663_0;
                double __tlet_arg0_0 = q_in_3[((__q_in_3_Cell_stride_0_0_0_0 * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride_0_0_0_0 * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_1 = gtir_tmp_669_0;
                double __tlet_arg1 = gtir_tmp_676_0;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_568;
                double __tlet_arg1_1 = gtir_tmp_666_0;
                double __tlet_arg1_2 = p[(((- __p_Cell_range_0) + (__p_K_stride_0_0_0_0 * ((- __p_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_3 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0_0_0_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_395_minimum_fused_tlet_397_multiplies_fused_tlet_391_divides_fused_tlet_399_power_fused_tlet_393_plus_fused_tlet_396_multiplies_fused_tlet_400_multiplies)
                __tlet_result = ((((__tlet_arg0 / __tlet_arg1_2) + __tlet_arg1_1) * min(__tlet_arg0_1, __tlet_arg1_0)) * dace::math::pow((__tlet_arg0_0 * __tlet_arg1_3), __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_680 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_680, &__arg1___, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1___, &__output, 1);

        }
    } else {
        {
            double gtir_tmp_605_0;
            bool __map_fusion_gtir_tmp_571;
            double gtir_tmp_569_0;
            double gtir_tmp_602_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_356_get_value__clone_5c48e7c2_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_605_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_335_get_value__clone_5c3cea26_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_569_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_354_get_value__clone_5c5423ee_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_602_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1 = gtir_tmp_569_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_336_less)
                __tlet_result = (__tlet_arg0 < __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_571 = __tlet_result;
            }
            if_stmt_15_94_0_12(__map_fusion_gtir_tmp_571, __map_fusion_gtir_tmp_102, gtir_tmp_187, gtir_tmp_206, gtir_tmp_448, &rho[0], gtir_tmp_600, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0_0_0_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);
            {
                double __tlet_arg0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg0_0 = gtir_tmp_600;
                double __tlet_arg1 = gtir_tmp_605_0;
                double __tlet_arg1_0 = gtir_tmp_602_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_357_greater_fused_tlet_355_less_fused_tlet_358_and_)
                __tlet_result = ((__tlet_arg0 < __tlet_arg1_0) && (__tlet_arg0_0 > __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_609 = __tlet_result;
            }

        }
        if (__map_fusion_gtir_tmp_609) {
            {
                double __arg1________;
                double __map_fusion_gtir_tmp_616;
                double gtir_tmp_610_0;

                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_359_get_value__clone_5c4e3efc_1c88_11f1_a73a_1f4e8615a1c5)
                    __tlet_out = 30.0;
                    ///////////////////

                    gtir_tmp_610_0 = __tlet_out;
                }
                {
                    double __tlet_arg0 = gtir_tmp_600;
                    double __tlet_arg0_0 = __map_fusion_gtir_tmp_102;
                    double __tlet_arg1 = gtir_tmp_556;
                    double __tlet_arg1_0 = gtir_tmp_610_0;
                    double __tlet_result;

                    ///////////////////
                    // Tasklet code (tlet_360_divides_fused_tlet_361_minus_fused_tlet_362_minimum)
                    __tlet_result = min(__tlet_arg0, ((__tlet_arg0_0 / __tlet_arg1_0) - __tlet_arg1));
                    ///////////////////

                    __map_fusion_gtir_tmp_616 = __tlet_result;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__map_fusion_gtir_tmp_616, &__arg1________, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__arg1________, &gtir_tmp_617, 1);

            }
        } else {
            {


                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_600, &gtir_tmp_617, 1);

            }
        }
        {
            double gtir_tmp_681_0;
            double gtir_tmp_684_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_403_get_value__clone_5cecbbe0_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-07;
                ///////////////////

                gtir_tmp_684_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_401_get_value__clone_5c595a44_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_681_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg0_0 = q_in_3[((__q_in_3_Cell_stride_0_0_0_0 * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride_0_0_0_0 * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_684_0;
                double __tlet_arg1_0 = gtir_tmp_681_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_404_less_equal_fused_tlet_402_less_fused_tlet_405_and_)
                __tlet_result = ((__tlet_arg0 < __tlet_arg1_0) && (__tlet_arg0_0 <= __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_688 = __tlet_result;
            }

        }
        if (__map_fusion_gtir_tmp_688) {
            {
                double if_stmt_17___arg1________;
                double gtir_tmp_689_0;
                double __map_fusion_gtir_tmp_691;

                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_406_get_value__clone_5cc8e36e_1c88_11f1_a73a_1f4e8615a1c5)
                    __tlet_out = 0.0;
                    ///////////////////

                    gtir_tmp_689_0 = __tlet_out;
                }
                {
                    double __tlet_arg0 = gtir_tmp_617;
                    double __tlet_arg1 = gtir_tmp_689_0;
                    double __tlet_result;

                    ///////////////////
                    // Tasklet code (tlet_407_minimum)
                    __tlet_result = min(__tlet_arg0, __tlet_arg1);
                    ///////////////////

                    __map_fusion_gtir_tmp_691 = __tlet_result;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__map_fusion_gtir_tmp_691, &if_stmt_17___arg1________, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &if_stmt_17___arg1________, &gtir_tmp_692, 1);

            }
        } else {
            {


                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_617, &gtir_tmp_692, 1);

            }
        }
        {
            double __arg2___;


            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_692, &__arg2___, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2___, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_19_86_0_13(const bool&  __cond, const double&  __map_fusion_gtir_tmp_102, const double&  __map_fusion_gtir_tmp_568, const double&  __map_fusion_gtir_tmp_77, const double&  gtir_tmp_187, const double&  gtir_tmp_206, const double&  gtir_tmp_448, const double&  gtir_tmp_556, const double* __restrict__ p, const double*  q_in_3, const double* __restrict__ rho, const double*  te, double&  __output, int __p_Cell_range_0, int __p_K_range_0, int __p_K_stride_0_0_0, int __q_in_3_Cell_range_0, int __q_in_3_Cell_stride_0_0_0, int __q_in_3_K_range_0, int __q_in_3_K_stride_0_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0_0, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride_0_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1______;
            double gtir_tmp_635_0;
            double gtir_tmp_636_0;
            double __map_fusion_gtir_tmp_649;
            double gtir_tmp_645_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_373_get_value__clone_5cf2fb04_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.28003;
                ///////////////////

                gtir_tmp_635_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_379_get_value__clone_5c8008ce_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.8;
                ///////////////////

                gtir_tmp_645_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_374_get_value__clone_5cd58600_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = -1.46293e-07;
                ///////////////////

                gtir_tmp_636_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_635_0;
                double __tlet_arg0_0 = q_in_3[((__q_in_3_Cell_stride_0_0_0 * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride_0_0_0 * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0_0 = gtir_tmp_636_0;
                double __tlet_arg1 = gtir_tmp_645_0;
                double __tlet_arg1_0 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0_0_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_0_0 = p[(((- __p_Cell_range_0) + (__p_K_stride_0_0_0 * ((- __p_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_1 = __map_fusion_gtir_tmp_77;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_375_multiplies_fused_tlet_376_plus_fused_tlet_377_multiplies_fused_tlet_378_multiplies_fused_tlet_380_power_fused_tlet_381_multiplies)
                __tlet_result = (((__tlet_arg0 + (__tlet_arg0_0_0 * __tlet_arg1_0_0)) * __tlet_arg1_1) * dace::math::pow((__tlet_arg0_0 * __tlet_arg1_0), __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_649 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_649, &__arg1______, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1______, &__output, 1);

        }
    } else {
        {
            double __arg2__;
            bool __map_fusion_gtir_tmp_662;
            double gtir_tmp_654_0;
            double gtir_tmp_653_0;
            double gtir_tmp_650_0;
            double gtir_tmp_694;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_382_get_value__clone_5c74e8cc_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_650_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_385_get_value__clone_5c69cb54_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 3339.5;
                ///////////////////

                gtir_tmp_654_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_384_get_value__clone_5c6f6e42_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_653_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg0_0 = gtir_tmp_653_0;
                double __tlet_arg0_1 = gtir_tmp_654_0;
                double __tlet_arg1 = __map_fusion_gtir_tmp_568;
                double __tlet_arg1_0 = gtir_tmp_650_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_386_multiplies_fused_tlet_387_minus_fused_tlet_388_greater_fused_tlet_383_greater_equal_fused_tlet_389_and_)
                __tlet_result = ((__tlet_arg0 >= __tlet_arg1_0) && (__tlet_arg0 > (__tlet_arg0_0 - (__tlet_arg0_1 * __tlet_arg1))));
                ///////////////////

                __map_fusion_gtir_tmp_662 = __tlet_result;
            }
            if_stmt_18_90_0_5(__map_fusion_gtir_tmp_662, __map_fusion_gtir_tmp_102, __map_fusion_gtir_tmp_568, gtir_tmp_187, gtir_tmp_206, gtir_tmp_448, gtir_tmp_556, &p[0], &q_in_3[0], &rho[0], &te[0], gtir_tmp_694, __p_Cell_range_0, __p_K_range_0, __p_K_stride_0_0_0, __q_in_3_Cell_range_0, __q_in_3_Cell_stride_0_0_0, __q_in_3_K_range_0, __q_in_3_K_stride_0_0_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0_0_0, __te_Cell_range_0, __te_K_range_0, __te_K_stride_0_0_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_694, &__arg2__, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2__, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_20_3_5_72(const bool&  __cond, const double&  __map_fusion_gtir_tmp_102, const double&  __map_fusion_gtir_tmp_568, const double&  __map_fusion_gtir_tmp_77, const double&  gtir_tmp_187, const double&  gtir_tmp_206, const double&  gtir_tmp_448, const double&  gtir_tmp_556, const double* __restrict__ p, const double*  q_in_3, const double* __restrict__ rho, const double*  te, double&  __output, int __p_Cell_range_0, int __p_K_range_0, int __p_K_stride_0_0, int __q_in_3_Cell_range_0, int __q_in_3_Cell_stride_0_0, int __q_in_3_K_range_0, int __q_in_3_K_stride_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1_______;
            double gtir_tmp_625_0;
            double gtir_tmp_696;
            double gtir_tmp_622_0;
            bool __map_fusion_gtir_tmp_634;
            double __map_fusion_gtir_tmp_704;
            double gtir_tmp_700_0;
            double gtir_tmp_626_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_409_get_value__clone_5c5f0fac_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_700_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_368_get_value__clone_5cdb2df8_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 3339.5;
                ///////////////////

                gtir_tmp_626_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_365_get_value__clone_5cbc4dd4_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_622_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_367_get_value__clone_5ce13400_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_625_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg0_0 = gtir_tmp_626_0;
                double __tlet_arg0_1 = gtir_tmp_625_0;
                double __tlet_arg1 = __map_fusion_gtir_tmp_568;
                double __tlet_arg1_0 = gtir_tmp_622_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_369_multiplies_fused_tlet_370_minus_fused_tlet_371_less_equal_fused_tlet_366_greater_equal_fused_tlet_372_and_)
                __tlet_result = ((__tlet_arg0 >= __tlet_arg1_0) && (__tlet_arg0 <= (__tlet_arg0_1 - (__tlet_arg0_0 * __tlet_arg1))));
                ///////////////////

                __map_fusion_gtir_tmp_634 = __tlet_result;
            }
            if_stmt_19_86_0_13(__map_fusion_gtir_tmp_634, __map_fusion_gtir_tmp_102, __map_fusion_gtir_tmp_568, __map_fusion_gtir_tmp_77, gtir_tmp_187, gtir_tmp_206, gtir_tmp_448, gtir_tmp_556, &p[0], &q_in_3[0], &rho[0], &te[0], gtir_tmp_696, __p_Cell_range_0, __p_K_range_0, __p_K_stride_0_0, __q_in_3_Cell_range_0, __q_in_3_Cell_stride_0_0, __q_in_3_K_range_0, __q_in_3_K_stride_0_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0_0, __te_Cell_range_0, __te_K_range_0, __te_K_stride_0_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);
            {
                double __tlet_arg0 = gtir_tmp_696;
                double __tlet_arg0_0 = q_in_3[((__q_in_3_Cell_stride_0_0 * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride_0_0 * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_700_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_408_neg_fused_tlet_410_divides_fused_tlet_411_maximum)
                __tlet_result = max(__tlet_arg0, ((- __tlet_arg0_0) / __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_704 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_704, &__arg1_______, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_______, &__output, 1);

        }
    } else {
        {
            double __arg2_____;
            double gtir_tmp_705_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_412_get_value__clone_5cc28e74_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_705_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_705_0, &__arg2_____, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_____, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_23_55_0_5(const bool&  __cond, const double&  __map_fusion_gtir_tmp_568, const double&  __map_fusion_gtir_tmp_77, const double* __restrict__ p, const double*  q_in_5, const double* __restrict__ rho, double&  __output, int __p_Cell_range_0, int __p_K_range_0, int __p_K_stride_0_0_0_0, int __q_in_5_Cell_range_0, int __q_in_5_Cell_stride_0_0_0_0, int __q_in_5_K_range_0, int __q_in_5_K_stride_0_0_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1___________________________;
            double __map_fusion_gtir_tmp_778;
            double gtir_tmp_761_0;
            double gtir_tmp_767_0;
            double gtir_tmp_762_0;
            double gtir_tmp_774_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_444_get_value__clone_5da1d6f6_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.153907;
                ///////////////////

                gtir_tmp_761_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_445_get_value__clone_5d1c86fe_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = -7.86703e-07;
                ///////////////////

                gtir_tmp_762_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_448_get_value__clone_5d3ae6d0_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_767_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_452_get_value__clone_5d35b5de_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.6;
                ///////////////////

                gtir_tmp_774_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_761_0;
                double __tlet_arg0_0 = q_in_5[((__q_in_5_Cell_stride_0_0_0_0 * ((- __q_in_5_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_5_K_stride_0_0_0_0 * ((- __q_in_5_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0_0 = gtir_tmp_767_0;
                double __tlet_arg0_1 = gtir_tmp_762_0;
                double __tlet_arg1 = gtir_tmp_774_0;
                double __tlet_arg1_0 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0_0_0_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_0_0 = p[(((- __p_Cell_range_0) + (__p_K_stride_0_0_0_0 * ((- __p_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_1 = __map_fusion_gtir_tmp_568;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_446_multiplies_fused_tlet_447_plus_fused_tlet_449_minimum_fused_tlet_450_multiplies_fused_tlet_451_multiplies_fused_tlet_453_power_fused_tlet_454_multiplies)
                __tlet_result = (((__tlet_arg0 + (__tlet_arg0_1 * __tlet_arg1_0_0)) * min(__tlet_arg0_0_0, __tlet_arg1_1)) * dace::math::pow((__tlet_arg0_0 * __tlet_arg1_0), __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_778 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_778, &__arg1___________________________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1___________________________, &__output, 1);

        }
    } else {
        {
            double __arg2____;
            double gtir_tmp_780_0;
            double gtir_tmp_789_0;
            double gtir_tmp_779_0;
            double __map_fusion_gtir_tmp_793;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_455_get_value__clone_5d89af36_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0418521;
                ///////////////////

                gtir_tmp_779_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_456_get_value__clone_5d9bd792_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = -4.7524e-08;
                ///////////////////

                gtir_tmp_780_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_461_get_value__clone_5d2982c8_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.6;
                ///////////////////

                gtir_tmp_789_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_779_0;
                double __tlet_arg0_0 = q_in_5[((__q_in_5_Cell_stride_0_0_0_0 * ((- __q_in_5_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_5_K_stride_0_0_0_0 * ((- __q_in_5_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0_0 = gtir_tmp_780_0;
                double __tlet_arg1 = gtir_tmp_789_0;
                double __tlet_arg1_0 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0_0_0_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_1 = __map_fusion_gtir_tmp_77;
                double __tlet_arg1_2 = p[(((- __p_Cell_range_0) + (__p_K_stride_0_0_0_0 * ((- __p_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_457_multiplies_fused_tlet_458_plus_fused_tlet_459_multiplies_fused_tlet_460_multiplies_fused_tlet_462_power_fused_tlet_463_multiplies)
                __tlet_result = (((__tlet_arg0 + (__tlet_arg0_0_0 * __tlet_arg1_2)) * __tlet_arg1_1) * dace::math::pow((__tlet_arg0_0 * __tlet_arg1_0), __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_793 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_793, &__arg2____, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2____, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_24_51_0_8(const bool&  __cond, const double&  __map_fusion_gtir_tmp_102, const double&  __map_fusion_gtir_tmp_568, const double&  __map_fusion_gtir_tmp_77, const double* __restrict__ p, const double*  q_in_5, const double* __restrict__ rho, const double*  te, double&  __output, int __p_Cell_range_0, int __p_K_range_0, int __p_K_stride_0_0_0, int __q_in_5_Cell_range_0, int __q_in_5_Cell_stride_0_0_0, int __q_in_5_K_range_0, int __q_in_5_K_stride_0_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0_0, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride_0_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1___________________________;
            double gtir_tmp_729_0;
            double __map_fusion_gtir_tmp_752;
            double gtir_tmp_748_0;
            double gtir_tmp_739_0;
            double gtir_tmp_728_0;
            double gtir_tmp_734_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_431_get_value__clone_5da7a6e4_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 2.6531e-07;
                ///////////////////

                gtir_tmp_739_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_436_get_value__clone_5dae115a_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.6;
                ///////////////////

                gtir_tmp_748_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_428_get_value__clone_5d407898_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 2554.99;
                ///////////////////

                gtir_tmp_734_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_425_get_value__clone_5d8f8b9a_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = -0.00152398;
                ///////////////////

                gtir_tmp_729_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_424_get_value__clone_5d835988_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.398561;
                ///////////////////

                gtir_tmp_728_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_728_0;
                double __tlet_arg0_0 = q_in_5[((__q_in_5_Cell_stride_0_0_0 * ((- __q_in_5_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_5_K_stride_0_0_0 * ((- __q_in_5_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0_0 = gtir_tmp_729_0;
                double __tlet_arg0_1 = gtir_tmp_739_0;
                double __tlet_arg0_2 = gtir_tmp_734_0;
                double __tlet_arg1 = gtir_tmp_748_0;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_102;
                double __tlet_arg1_0_0 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0_0_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_1 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_2 = p[(((- __p_Cell_range_0) + (__p_K_stride_0_0_0 * ((- __p_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_429_divides_fused_tlet_432_multiplies_fused_tlet_426_multiplies_fused_tlet_427_plus_fused_tlet_430_plus_fused_tlet_433_plus_fused_tlet_435_multiplies_fused_tlet_437_power_fused_tlet_434_multiplies_fused_tlet_438_multiplies)
                __tlet_result = (((((__tlet_arg0 + (__tlet_arg0_0_0 * __tlet_arg1_1)) + (__tlet_arg0_2 / __tlet_arg1_2)) + (__tlet_arg0_1 * __tlet_arg1_2)) * __tlet_arg1_0) * dace::math::pow((__tlet_arg0_0 * __tlet_arg1_0_0), __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_752 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_752, &__arg1___________________________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1___________________________, &__output, 1);

        }
    } else {
        {
            double __arg2____;
            double gtir_tmp_754_0;
            double gtir_tmp_794;
            double gtir_tmp_753_0;
            bool __map_fusion_gtir_tmp_760;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_439_get_value__clone_5d22c8f2_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_753_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_440_get_value__clone_5d7ca610_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 3339.5;
                ///////////////////

                gtir_tmp_754_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg0_0 = gtir_tmp_753_0;
                double __tlet_arg0_1 = gtir_tmp_754_0;
                double __tlet_arg1 = __map_fusion_gtir_tmp_568;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_441_multiplies_fused_tlet_442_minus_fused_tlet_443_greater)
                __tlet_result = (__tlet_arg0 > (__tlet_arg0_0 - (__tlet_arg0_1 * __tlet_arg1)));
                ///////////////////

                __map_fusion_gtir_tmp_760 = __tlet_result;
            }
            if_stmt_23_55_0_5(__map_fusion_gtir_tmp_760, __map_fusion_gtir_tmp_568, __map_fusion_gtir_tmp_77, &p[0], &q_in_5[0], &rho[0], gtir_tmp_794, __p_Cell_range_0, __p_K_range_0, __p_K_stride_0_0_0, __q_in_5_Cell_range_0, __q_in_5_Cell_stride_0_0_0, __q_in_5_K_range_0, __q_in_5_K_stride_0_0_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0_0_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_794, &__arg2____, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2____, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_25_3_5_15(const bool&  __cond, const double&  __map_fusion_gtir_tmp_102, const double&  __map_fusion_gtir_tmp_568, const double&  __map_fusion_gtir_tmp_77, const double* __restrict__ p, const double*  q_in_5, const double* __restrict__ rho, const double*  te, double&  __output, int __p_Cell_range_0, int __p_K_range_0, int __p_K_stride_0_0, int __q_in_5_Cell_range_0, int __q_in_5_Cell_stride_0_0, int __q_in_5_K_range_0, int __q_in_5_K_stride_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1_______;
            double gtir_tmp_800_0;
            double __map_fusion_gtir_tmp_804;
            double gtir_tmp_725_0;
            bool __map_fusion_gtir_tmp_727;
            double gtir_tmp_796;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_465_get_value__clone_5db44840_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_800_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_422_get_value__clone_5d95f458_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_725_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1 = gtir_tmp_725_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_423_less)
                __tlet_result = (__tlet_arg0 < __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_727 = __tlet_result;
            }
            if_stmt_24_51_0_8(__map_fusion_gtir_tmp_727, __map_fusion_gtir_tmp_102, __map_fusion_gtir_tmp_568, __map_fusion_gtir_tmp_77, &p[0], &q_in_5[0], &rho[0], &te[0], gtir_tmp_796, __p_Cell_range_0, __p_K_range_0, __p_K_stride_0_0, __q_in_5_Cell_range_0, __q_in_5_Cell_stride_0_0, __q_in_5_K_range_0, __q_in_5_K_stride_0_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0_0, __te_Cell_range_0, __te_K_range_0, __te_K_stride_0_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);
            {
                double __tlet_arg0 = gtir_tmp_796;
                double __tlet_arg0_0 = q_in_5[((__q_in_5_Cell_stride_0_0 * ((- __q_in_5_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_5_K_stride_0_0 * ((- __q_in_5_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_800_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_464_neg_fused_tlet_466_divides_fused_tlet_467_maximum)
                __tlet_result = max(__tlet_arg0, ((- __tlet_arg0_0) / __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_804 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_804, &__arg1_______, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_______, &__output, 1);

        }
    } else {
        {
            double __arg2_____;
            double gtir_tmp_805_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_468_get_value__clone_5d1672d2_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_805_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_805_0, &__arg2_____, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_____, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_41_156_0_4(const bool&  __cond, const double*  q_in_2, const double* __restrict__ rho, const double*  te, double&  __output, int __q_in_2_Cell_range_0, int __q_in_2_Cell_stride_0_0_0_0, int __q_in_2_K_range_0, int __q_in_2_K_stride_0_0_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0_0_0, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride_0_0_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1__;
            double gtir_tmp_1018_0;
            double __map_fusion_gtir_tmp_1038;
            double gtir_tmp_1026_0;
            double gtir_tmp_1019_0;
            double gtir_tmp_1032_0;
            double gtir_tmp_1029_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_577_get_value__clone_5e87452e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 271.15;
                ///////////////////

                gtir_tmp_1019_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_585_get_value__clone_5e7b48dc_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1.75;
                ///////////////////

                gtir_tmp_1032_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_583_get_value__clone_5e6deaf2_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 9.95e-05;
                ///////////////////

                gtir_tmp_1029_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_581_get_value__clone_5e8d4aaa_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1.0;
                ///////////////////

                gtir_tmp_1026_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_576_get_value__clone_5e550ee2_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.66;
                ///////////////////

                gtir_tmp_1018_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_1018_0;
                double __tlet_arg0_0 = gtir_tmp_1029_0;
                double __tlet_arg0_0_0 = gtir_tmp_1019_0;
                double __tlet_arg0_1 = q_in_2[((__q_in_2_Cell_stride_0_0_0_0 * ((- __q_in_2_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_2_K_stride_0_0_0_0 * ((- __q_in_2_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_1032_0;
                double __tlet_arg1_0 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0_0_0_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_1 = gtir_tmp_1026_0;
                double __tlet_arg1_2 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_578_minus_fused_tlet_579_multiplies_fused_tlet_580_exp_fused_tlet_582_minus_fused_tlet_584_multiplies_fused_tlet_586_power_fused_tlet_587_multiplies_fused_tlet_588_multiplies)
                __tlet_result = ((dace::math::exp((__tlet_arg0 * (__tlet_arg0_0_0 - __tlet_arg1_2))) - __tlet_arg1_1) * (__tlet_arg0_0 * dace::math::pow((__tlet_arg0_1 * __tlet_arg1_0), __tlet_arg1)));
                ///////////////////

                __map_fusion_gtir_tmp_1038 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1038, &__arg1__, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1__, &__output, 1);

        }
    } else {
        {
            double __arg2______;
            double gtir_tmp_1039_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_589_get_value__clone_5e615aa8_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1039_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1039_0, &__arg2______, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2______, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_42_148_0_20(const bool&  __cond, const double&  __map_fusion_gtir_tmp_77, const bool&  __map_fusion_gtir_tmp_990, const double*  q_in_1, const double*  q_in_2, const double* __restrict__ rho, const double*  te, double&  __output, int __q_in_1_Cell_range_0, int __q_in_1_Cell_stride_0_0_0, int __q_in_1_K_range_0, int __q_in_1_K_stride_0_0_0, int __q_in_2_Cell_range_0, int __q_in_2_Cell_stride_0_0_0, int __q_in_2_K_range_0, int __q_in_2_K_stride_0_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0_0, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride_0_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1_______________;
            double __map_fusion_gtir_tmp_998;
            double gtir_tmp_996_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_563_get_value__clone_5e816c08_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_996_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_2[((__q_in_2_Cell_stride_0_0_0 * ((- __q_in_2_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_2_K_stride_0_0_0 * ((- __q_in_2_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_996_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_564_divides)
                __tlet_result = (__tlet_arg0 / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_998 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_998, &__arg1_______________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_______________, &__output, 1);

        }
    } else {
        {
            double __arg2_;
            double gtir_tmp_1040;
            double gtir_tmp_1006_0;
            bool __map_fusion_gtir_tmp_1017;
            double gtir_tmp_1009_0;
            double gtir_tmp_999_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_569_get_value__clone_5e74c3ae_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1006_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_565_get_value__clone_5e5b101c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 236.15;
                ///////////////////

                gtir_tmp_999_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_571_get_value__clone_5e67b682_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.1;
                ///////////////////

                gtir_tmp_1009_0 = __tlet_out;
            }
            {
                bool __tlet_arg0 = __map_fusion_gtir_tmp_990;
                double __tlet_arg0_0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg0_0_0 = gtir_tmp_1009_0;
                double __tlet_arg0_1 = q_in_2[((__q_in_2_Cell_stride_0_0_0 * ((- __q_in_2_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_2_K_stride_0_0_0 * ((- __q_in_2_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_2 = __map_fusion_gtir_tmp_77;
                double __tlet_arg1 = q_in_1[((__q_in_1_Cell_stride_0_0_0 * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride_0_0_0 * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1_0 = gtir_tmp_999_0;
                double __tlet_arg1_0_0 = gtir_tmp_1006_0;
                double __tlet_arg1_1 = q_in_1[((__q_in_1_Cell_stride_0_0_0 * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride_0_0_0 * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_572_multiplies_fused_tlet_573_greater_fused_tlet_568_plus_fused_tlet_570_less_equal_fused_tlet_574_or__fused_tlet_566_greater_fused_tlet_567_and__fused_tlet_575_and_)
                __tlet_result = ((__tlet_arg0 && (__tlet_arg0_0 > __tlet_arg1_0)) && (((__tlet_arg0_2 + __tlet_arg1_1) <= __tlet_arg1_0_0) || (__tlet_arg0_1 > (__tlet_arg0_0_0 * __tlet_arg1))));
                ///////////////////

                __map_fusion_gtir_tmp_1017 = __tlet_result;
            }
            if_stmt_41_156_0_4(__map_fusion_gtir_tmp_1017, &q_in_2[0], &rho[0], &te[0], gtir_tmp_1040, __q_in_2_Cell_range_0, __q_in_2_Cell_stride_0_0_0, __q_in_2_K_range_0, __q_in_2_K_stride_0_0_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0_0_0, __te_Cell_range_0, __te_K_range_0, __te_K_stride_0_0_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1040, &__arg2_, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_43_148_0_17(const double&  __arg2_, const bool&  __cond, const double&  __map_fusion_gtir_tmp_398, const double&  gtir_tmp_1042, const double*  q_in_2, const double*  q_in_4, const double* __restrict__ rho, double&  __output, int __q_in_2_Cell_range_0, int __q_in_2_Cell_stride_0_0_0, int __q_in_2_K_range_0, int __q_in_2_K_stride_0_0_0, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride_0_0_0, int __q_in_4_K_range_0, int __q_in_4_K_stride_0_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1_______________;
            double __map_fusion_gtir_tmp_1067;
            double gtir_tmp_1054_0;
            double gtir_tmp_1061_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_596_get_value__clone_5ea6bf58_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.00124;
                ///////////////////

                gtir_tmp_1054_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_600_get_value__clone_5ead5476_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1.625;
                ///////////////////

                gtir_tmp_1061_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_1042;
                double __tlet_arg0_0 = q_in_4[((__q_in_4_Cell_stride_0_0_0 * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride_0_0_0 * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_1 = gtir_tmp_1054_0;
                double __tlet_arg0_2 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0_0_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1 = gtir_tmp_1061_0;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_398;
                double __tlet_arg1_0_0 = q_in_2[((__q_in_2_Cell_stride_0_0_0 * ((- __q_in_2_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_2_K_stride_0_0_0 * ((- __q_in_2_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_599_multiplies_fused_tlet_601_power_fused_tlet_597_divides_fused_tlet_598_multiplies_fused_tlet_602_multiplies_fused_tlet_603_plus)
                __tlet_result = (__tlet_arg0 + ((__tlet_arg0_1 * (__tlet_arg0_0 / __tlet_arg1_0)) * dace::math::pow((__tlet_arg0_2 * __tlet_arg1_0_0), __tlet_arg1)));
                ///////////////////

                __map_fusion_gtir_tmp_1067 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1067, &__arg1_______________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_______________, &__output, 1);

        }
    } else {
        {


            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_44_3_5_168(const bool&  __cond, const double&  __map_fusion_gtir_tmp_398, const double&  __map_fusion_gtir_tmp_77, const double*  q_in_1, const double*  q_in_2, const double*  q_in_3, const double*  q_in_4, const double* __restrict__ rho, const double*  te, double&  __output, int __q_in_1_Cell_range_0, int __q_in_1_Cell_stride_0_0, int __q_in_1_K_range_0, int __q_in_1_K_stride_0_0, int __q_in_2_Cell_range_0, int __q_in_2_Cell_stride_0_0, int __q_in_2_K_range_0, int __q_in_2_K_stride_0_0, int __q_in_3_Cell_range_0, int __q_in_3_Cell_stride_0_0, int __q_in_3_K_range_0, int __q_in_3_K_stride_0_0, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride_0_0, int __q_in_4_K_range_0, int __q_in_4_K_stride_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1_________;
            double gtir_tmp_1049_0;
            bool __map_fusion_gtir_tmp_990;
            double gtir_tmp_983_0;
            double gtir_tmp_1068;
            double gtir_tmp_986_0;
            double gtir_tmp_1042;
            double gtir_tmp_991_0;
            bool __map_fusion_gtir_tmp_1053;
            bool __map_fusion_gtir_tmp_995;
            double gtir_tmp_1046_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_591_get_value__clone_5e9a2e6e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_1046_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_560_get_value__clone_5e940d90_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 236.15;
                ///////////////////

                gtir_tmp_991_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_557_get_value__clone_5e48fc88_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 271.15;
                ///////////////////

                gtir_tmp_986_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_555_get_value__clone_5e4efb24_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_983_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_593_get_value__clone_5ea06496_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-07;
                ///////////////////

                gtir_tmp_1049_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_4[((__q_in_4_Cell_stride_0_0 * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride_0_0 * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0 = q_in_3[((__q_in_3_Cell_stride_0_0 * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride_0_0 * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_1049_0;
                double __tlet_arg1_0 = q_in_2[((__q_in_2_Cell_stride_0_0 * ((- __q_in_2_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_2_K_stride_0_0 * ((- __q_in_2_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1_1 = gtir_tmp_1046_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_590_minimum_fused_tlet_592_greater_fused_tlet_594_greater_fused_tlet_595_and_)
                __tlet_result = ((min(__tlet_arg0, __tlet_arg1_0) > __tlet_arg1_1) && (__tlet_arg0_0 > __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_1053 = __tlet_result;
            }
            {
                double __tlet_arg0 = q_in_2[((__q_in_2_Cell_stride_0_0 * ((- __q_in_2_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_2_K_stride_0_0 * ((- __q_in_2_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1 = gtir_tmp_986_0;
                double __tlet_arg1_0 = gtir_tmp_983_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_556_greater_fused_tlet_558_less_fused_tlet_559_and_)
                __tlet_result = ((__tlet_arg0 > __tlet_arg1_0) && (__tlet_arg0_0 < __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_990 = __tlet_result;
            }
            {
                bool __tlet_arg0 = __map_fusion_gtir_tmp_990;
                double __tlet_arg0_0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1 = gtir_tmp_991_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_561_less_equal_fused_tlet_562_and_)
                __tlet_result = (__tlet_arg0 && (__tlet_arg0_0 <= __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_995 = __tlet_result;
            }
            if_stmt_42_148_0_20(__map_fusion_gtir_tmp_995, __map_fusion_gtir_tmp_77, __map_fusion_gtir_tmp_990, &q_in_1[0], &q_in_2[0], &rho[0], &te[0], gtir_tmp_1042, __q_in_1_Cell_range_0, __q_in_1_Cell_stride_0_0, __q_in_1_K_range_0, __q_in_1_K_stride_0_0, __q_in_2_Cell_range_0, __q_in_2_Cell_stride_0_0, __q_in_2_K_range_0, __q_in_2_K_stride_0_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0_0, __te_Cell_range_0, __te_K_range_0, __te_K_stride_0_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);
            if_stmt_43_148_0_17(gtir_tmp_1042, __map_fusion_gtir_tmp_1053, __map_fusion_gtir_tmp_398, gtir_tmp_1042, &q_in_2[0], &q_in_4[0], &rho[0], gtir_tmp_1068, __q_in_2_Cell_range_0, __q_in_2_Cell_stride_0_0, __q_in_2_K_range_0, __q_in_2_K_stride_0_0, __q_in_4_Cell_range_0, __q_in_4_Cell_stride_0_0, __q_in_4_K_range_0, __q_in_4_K_stride_0_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1068, &__arg1_________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_________, &__output, 1);

        }
    } else {
        {
            double __arg2______;
            double gtir_tmp_1072_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_604_get_value__clone_5f0b3aa0_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1072_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1072_0, &__arg2______, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2______, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_57_67_1_8(const bool&  __cond, const double&  __map_fusion_gtir_tmp_420, const double&  gtir_tmp_187, const double&  gtir_tmp_206, const double*  q_in_4, double&  __output, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride_0_0_0, int __q_in_4_K_range_0, int __q_in_4_K_stride_0_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1____________________________;
            double __map_fusion_gtir_tmp_1266;
            double gtir_tmp_1246_0;
            double gtir_tmp_1258_0;
            double gtir_tmp_1245_0;
            double gtir_tmp_1244_0;
            double gtir_tmp_1253_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_694_get_value__clone_60926f88_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1245_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_695_get_value__clone_60bd6dc8_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1246_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_693_get_value__clone_609f54be_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.001;
                ///////////////////

                gtir_tmp_1244_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_699_get_value__clone_60416a52_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 65.25;
                ///////////////////

                gtir_tmp_1253_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_702_get_value__clone_60cad062_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = -3.5;
                ///////////////////

                gtir_tmp_1258_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = __map_fusion_gtir_tmp_420;
                double __tlet_arg0_0 = gtir_tmp_1253_0;
                double __tlet_arg0_0_0 = q_in_4[((__q_in_4_Cell_stride_0_0_0 * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride_0_0_0 * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_1 = gtir_tmp_206;
                double __tlet_arg0_2 = gtir_tmp_1245_0;
                double __tlet_arg0_3 = q_in_4[((__q_in_4_Cell_stride_0_0_0 * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride_0_0_0 * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_4 = gtir_tmp_1244_0;
                double __tlet_arg1 = gtir_tmp_1258_0;
                double __tlet_arg1_0 = gtir_tmp_1246_0;
                double __tlet_arg1_1 = gtir_tmp_187;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_700_multiplies_fused_tlet_701_multiplies_fused_tlet_696_minus_fused_tlet_697_maximum_fused_tlet_703_power_fused_tlet_698_multiplies_fused_tlet_704_multiplies_fused_tlet_705_plus_fused_tlet_706_multiplies)
                __tlet_result = (__tlet_arg0 * ((__tlet_arg0_4 * max(__tlet_arg0_2, (__tlet_arg0_0_0 - __tlet_arg1_0))) + ((__tlet_arg0_3 * (__tlet_arg0_0 * __tlet_arg1_1)) * dace::math::pow(__tlet_arg0_1, __tlet_arg1))));
                ///////////////////

                __map_fusion_gtir_tmp_1266 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1266, &__arg1____________________________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1____________________________, &__output, 1);

        }
    } else {
        {
            double __arg2__________________;
            double gtir_tmp_1267_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_707_get_value__clone_6098335a_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1267_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1267_0, &__arg2__________________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2__________________, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_58_3_5_66(const bool&  __cond, const double&  __map_fusion_gtir_tmp_398, const double&  __map_fusion_gtir_tmp_420, const double&  gtir_tmp_187, const double&  gtir_tmp_206, const double&  gtir_tmp_556, const double*  q_in_4, double&  __output, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride_0_0, int __q_in_4_K_range_0, int __q_in_4_K_stride_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {
    bool __map_fusion_gtir_tmp_1220;
    double gtir_tmp_1239;

    if (__cond) {
        {
            double gtir_tmp_1218_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_677_get_value__clone_6046d154_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_1218_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_4[((__q_in_4_Cell_stride_0_0 * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride_0_0 * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_1218_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_678_greater)
                __tlet_result = (__tlet_arg0 > __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1220 = __tlet_result;
            }

        }
        if (__map_fusion_gtir_tmp_1220) {
            {
                double if_stmt_56___arg1____________________________;
                double gtir_tmp_1230_0;
                double gtir_tmp_1227_0;
                double gtir_tmp_1233_0;
                double gtir_tmp_1224_0;
                double __map_fusion_gtir_tmp_1237;
                double gtir_tmp_1221_0;

                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_685_get_value__clone_604c6a4c_1c88_11f1_a73a_1f4e8615a1c5)
                    __tlet_out = 0.6666666666666666;
                    ///////////////////

                    gtir_tmp_1230_0 = __tlet_out;
                }
                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_687_get_value__clone_60a52e5c_1c88_11f1_a73a_1f4e8615a1c5)
                    __tlet_out = 1.0;
                    ///////////////////

                    gtir_tmp_1233_0 = __tlet_out;
                }
                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_681_get_value__clone_60d1077a_1c88_11f1_a73a_1f4e8615a1c5)
                    __tlet_out = 0.6666666666666666;
                    ///////////////////

                    gtir_tmp_1224_0 = __tlet_out;
                }
                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_683_get_value__clone_60c43892_1c88_11f1_a73a_1f4e8615a1c5)
                    __tlet_out = 3e-09;
                    ///////////////////

                    gtir_tmp_1227_0 = __tlet_out;
                }
                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_679_get_value__clone_60b0ec92_1c88_11f1_a73a_1f4e8615a1c5)
                    __tlet_out = 0.0;
                    ///////////////////

                    gtir_tmp_1221_0 = __tlet_out;
                }
                {
                    double __tlet_arg0 = gtir_tmp_1221_0;
                    double __tlet_arg0_0 = gtir_tmp_1227_0;
                    double __tlet_arg1 = gtir_tmp_1233_0;
                    double __tlet_arg1_0 = gtir_tmp_1224_0;
                    double __tlet_arg1_0_0 = __map_fusion_gtir_tmp_398;
                    double __tlet_arg1_1 = gtir_tmp_556;
                    double __tlet_arg1_2 = gtir_tmp_1230_0;
                    double __tlet_result;

                    ///////////////////
                    // Tasklet code (tlet_684_divides_fused_tlet_686_power_fused_tlet_680_maximum_fused_tlet_688_minus_fused_tlet_682_multiplies_fused_tlet_689_divides)
                    __tlet_result = ((max(__tlet_arg0, __tlet_arg1_1) * __tlet_arg1_0) / (dace::math::pow((__tlet_arg0_0 / __tlet_arg1_0_0), __tlet_arg1_2) - __tlet_arg1));
                    ///////////////////

                    __map_fusion_gtir_tmp_1237 = __tlet_result;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__map_fusion_gtir_tmp_1237, &if_stmt_56___arg1____________________________, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &if_stmt_56___arg1____________________________, &gtir_tmp_1239, 1);

            }
        } else {
            {
                double if_stmt_56___arg2__________________;
                double gtir_tmp_1238_0;

                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_690_get_value__clone_60b72972_1c88_11f1_a73a_1f4e8615a1c5)
                    __tlet_out = 0.0;
                    ///////////////////

                    gtir_tmp_1238_0 = __tlet_out;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1238_0, &if_stmt_56___arg2__________________, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &if_stmt_56___arg2__________________, &gtir_tmp_1239, 1);

            }
        }
        {
            double __arg1____________________________;
            double __map_fusion_gtir_tmp_1271;
            double gtir_tmp_1268;
            bool __map_fusion_gtir_tmp_1243;
            double gtir_tmp_1241_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_691_get_value__clone_6051a00c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_1241_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_4[((__q_in_4_Cell_stride_0_0 * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride_0_0 * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_1241_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_692_greater)
                __tlet_result = (__tlet_arg0 > __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1243 = __tlet_result;
            }
            if_stmt_57_67_1_8(__map_fusion_gtir_tmp_1243, __map_fusion_gtir_tmp_420, gtir_tmp_187, gtir_tmp_206, &q_in_4[0], gtir_tmp_1268, __q_in_4_Cell_range_0, __q_in_4_Cell_stride_0_0, __q_in_4_K_range_0, __q_in_4_K_stride_0_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);
            {
                double __tlet_arg0 = gtir_tmp_1239;
                double __tlet_arg1 = gtir_tmp_1268;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_708_plus)
                __tlet_result = (__tlet_arg0 + __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1271 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1271, &__arg1____________________________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1____________________________, &__output, 1);

        }
    } else {
        {
            double __arg2__________________;
            double gtir_tmp_1272_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_709_get_value__clone_608ca742_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1272_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1272_0, &__arg2__________________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2__________________, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_48_168_0_28(const bool&  __cond, const double&  __map_fusion_gtir_tmp_568, const double* __restrict__ p, const double*  q_in_3, const double* __restrict__ rho, const double*  te, double&  __output, int __p_Cell_range_0, int __p_K_range_0, int __p_K_stride_0_0_0, int __q_in_3_Cell_range_0, int __q_in_3_Cell_stride_0_0_0, int __q_in_3_K_range_0, int __q_in_3_K_stride_0_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0_0, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride_0_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1__________________________;
            double gtir_tmp_1131_0;
            double __map_fusion_gtir_tmp_1144;
            double gtir_tmp_1122_0;
            double gtir_tmp_1125_0;
            double gtir_tmp_1140_0;
            double gtir_tmp_1128_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_633_get_value__clone_5f58a646_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_1128_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_631_get_value__clone_5fcdea3c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.000612654;
                ///////////////////

                gtir_tmp_1125_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_629_get_value__clone_5f83966c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 79.6863;
                ///////////////////

                gtir_tmp_1122_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_635_get_value__clone_5f6525e2_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 2950.0;
                ///////////////////

                gtir_tmp_1131_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_640_get_value__clone_5fdad6fc_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.8;
                ///////////////////

                gtir_tmp_1140_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_1122_0;
                double __tlet_arg0_0 = q_in_3[((__q_in_3_Cell_stride_0_0_0 * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride_0_0_0 * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0_0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg0_1 = gtir_tmp_1131_0;
                double __tlet_arg1 = gtir_tmp_1140_0;
                double __tlet_arg1_0 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0_0_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_0_0 = p[(((- __p_Cell_range_0) + (__p_K_stride_0_0_0 * ((- __p_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_0_1 = gtir_tmp_1128_0;
                double __tlet_arg1_1 = gtir_tmp_1125_0;
                double __tlet_arg1_2 = __map_fusion_gtir_tmp_568;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_630_divides_fused_tlet_632_plus_fused_tlet_634_minus_fused_tlet_636_multiplies_fused_tlet_637_plus_fused_tlet_638_multiplies_fused_tlet_639_multiplies_fused_tlet_641_power_fused_tlet_642_multiplies)
                __tlet_result = ((((__tlet_arg0 / __tlet_arg1_0_0) + __tlet_arg1_1) * ((__tlet_arg0_0_0 - __tlet_arg1_0_1) + (__tlet_arg0_1 * __tlet_arg1_2))) * dace::math::pow((__tlet_arg0_0 * __tlet_arg1_0), __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_1144 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1144, &__arg1__________________________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1__________________________, &__output, 1);

        }
    } else {
        {
            double __arg2____;
            double gtir_tmp_1145_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_643_get_value__clone_5fe943ea_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1145_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1145_0, &__arg2____, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2____, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_67_168_0_52(const bool&  __cond, const double&  __map_fusion_gtir_tmp_568, const double* __restrict__ p, const double*  q_in_5, const double* __restrict__ rho, const double*  te, double&  __output, int __p_Cell_range_0, int __p_K_range_0, int __p_K_stride_0_0_0, int __q_in_5_Cell_range_0, int __q_in_5_Cell_stride_0_0_0, int __q_in_5_K_range_0, int __q_in_5_K_stride_0_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0_0, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride_0_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1_____;
            double gtir_tmp_1388_0;
            double gtir_tmp_1391_0;
            double gtir_tmp_1394_0;
            double gtir_tmp_1403_0;
            double __map_fusion_gtir_tmp_1407;
            double gtir_tmp_1385_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_771_get_value__clone_610a7b54_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 2950.0;
                ///////////////////

                gtir_tmp_1394_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_776_get_value__clone_61597c5e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.6;
                ///////////////////

                gtir_tmp_1403_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_769_get_value__clone_6153b6de_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_1391_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_765_get_value__clone_60e859d4_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 12.31698;
                ///////////////////

                gtir_tmp_1385_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_767_get_value__clone_6101d850_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 7.39441e-05;
                ///////////////////

                gtir_tmp_1388_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_1385_0;
                double __tlet_arg0_0 = q_in_5[((__q_in_5_Cell_stride_0_0_0 * ((- __q_in_5_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_5_K_stride_0_0_0 * ((- __q_in_5_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0_0 = gtir_tmp_1394_0;
                double __tlet_arg0_1 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1 = gtir_tmp_1403_0;
                double __tlet_arg1_0 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0_0_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_0_0 = gtir_tmp_1391_0;
                double __tlet_arg1_0_1 = gtir_tmp_1388_0;
                double __tlet_arg1_1 = p[(((- __p_Cell_range_0) + (__p_K_stride_0_0_0 * ((- __p_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_2 = __map_fusion_gtir_tmp_568;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_770_minus_fused_tlet_772_multiplies_fused_tlet_773_plus_fused_tlet_766_divides_fused_tlet_768_plus_fused_tlet_774_multiplies_fused_tlet_775_multiplies_fused_tlet_777_power_fused_tlet_778_multiplies)
                __tlet_result = ((((__tlet_arg0 / __tlet_arg1_1) + __tlet_arg1_0_1) * ((__tlet_arg0_1 - __tlet_arg1_0_0) + (__tlet_arg0_0_0 * __tlet_arg1_2))) * dace::math::pow((__tlet_arg0_0 * __tlet_arg1_0), __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_1407 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1407, &__arg1_____, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_____, &__output, 1);

        }
    } else {
        {
            double __arg2__;
            double gtir_tmp_1408_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_779_get_value__clone_6110f682_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1408_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1408_0, &__arg2__, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2__, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_21_3_5_184(const bool&  __cond, const double&  __map_fusion_gtir_tmp_328, const double&  __map_fusion_gtir_tmp_568, const double&  gtir_tmp_1273, const double&  gtir_tmp_1324, const double&  gtir_tmp_500, const double&  gtir_tmp_706, const double&  gtir_tmp_806, const double* __restrict__ p, const double*  q_in_3, const double*  q_in_5, const double* __restrict__ rho, const double*  te, double&  __output, double&  __output_699295a4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_69e0b0d6_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_6a2de87e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_6a79d720_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_6ac690a6_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_6b140480_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, int __p_Cell_range_0, int __p_K_range_0, int __p_K_stride_0_0, int __q_in_3_Cell_range_0, int __q_in_3_Cell_stride_0_0, int __q_in_3_K_range_0, int __q_in_3_K_stride_0_0, int __q_in_5_Cell_range_0, int __q_in_5_Cell_stride_0_0, int __q_in_5_K_range_0, int __q_in_5_K_stride_0_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0_0, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride_0_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {

    if (__cond) {
        {
            double __arg1___________________________;
            double __arg1___________________________69929158_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1______________69e0ac80_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1____________________________6a2de450_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1___________________________6a79d2f2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1______6ac68c6e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1___________________________6b13ff4e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double gtir_tmp_708_0;
            double __map_fusion_gtir_tmp_710;
            double __map_fusion_gtir_tmp_1331;
            double gtir_tmp_1108_0;
            bool __map_fusion_gtir_tmp_1121;
            double gtir_tmp_1146;
            double gtir_tmp_1117_0;
            double gtir_tmp_1107_0;
            double gtir_tmp_1106_0;
            bool __map_fusion_gtir_tmp_1384;
            double gtir_tmp_1369_0;
            double gtir_tmp_1409;
            double gtir_tmp_1380_0;
            double gtir_tmp_1370_0;
            double gtir_tmp_1371_0;
            double __map_fusion_gtir_tmp_718;
            double gtir_tmp_714_0;
            double gtir_tmp_808_0;
            double __map_fusion_gtir_tmp_810;
            double __map_fusion_gtir_tmp_818;
            double gtir_tmp_814_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_413_get_value__clone_5d0b53de_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_708_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_706;
                double __tlet_arg1 = gtir_tmp_708_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_414_maximum)
                __tlet_result = max(__tlet_arg0, __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_710 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_710, &__arg1___________________________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1___________________________, &__output, 1);
            {
                double __tlet_arg0 = gtir_tmp_500;
                double __tlet_arg1 = gtir_tmp_1324;
                double __tlet_arg1_0 = gtir_tmp_1273;
                double __tlet_arg1_1 = __map_fusion_gtir_tmp_328;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_737_plus_fused_tlet_738_plus_fused_tlet_739_plus)
                __tlet_result = (((__tlet_arg0 + __tlet_arg1_1) + __tlet_arg1_0) + __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1331 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1331, &__arg1______________69e0ac80_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1______________69e0ac80_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_69e0b0d6_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_620_get_value__clone_5f8a2d38_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_1107_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_621_get_value__clone_5fe26da4_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 3339.5;
                ///////////////////

                gtir_tmp_1108_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_626_get_value__clone_5fd4424c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_1117_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_619_get_value__clone_5f51eaea_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_1106_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg0_0 = q_in_3[((__q_in_3_Cell_stride_0_0 * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride_0_0 * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0_0 = gtir_tmp_1106_0;
                double __tlet_arg0_0_1 = gtir_tmp_1108_0;
                double __tlet_arg0_1 = gtir_tmp_1107_0;
                double __tlet_arg1 = gtir_tmp_1117_0;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_568;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_622_multiplies_fused_tlet_623_minus_fused_tlet_624_maximum_fused_tlet_625_greater_fused_tlet_627_greater_fused_tlet_628_and_)
                __tlet_result = ((__tlet_arg0 > max(__tlet_arg0_0_0, (__tlet_arg0_1 - (__tlet_arg0_0_1 * __tlet_arg1_0)))) && (__tlet_arg0_0 > __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_1121 = __tlet_result;
            }
            if_stmt_48_168_0_28(__map_fusion_gtir_tmp_1121, __map_fusion_gtir_tmp_568, &p[0], &q_in_3[0], &rho[0], &te[0], gtir_tmp_1146, __p_Cell_range_0, __p_K_range_0, __p_K_stride_0_0, __q_in_3_Cell_range_0, __q_in_3_Cell_stride_0_0, __q_in_3_K_range_0, __q_in_3_K_stride_0_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0_0, __te_Cell_range_0, __te_K_range_0, __te_K_stride_0_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1146, &__arg1___________________________69929158_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1___________________________69929158_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_699295a4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_755_get_value__clone_615f1dd0_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_1369_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_756_get_value__clone_60fb5994_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_1370_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_762_get_value__clone_614dea06_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_1380_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_757_get_value__clone_60ee1856_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 3339.5;
                ///////////////////

                gtir_tmp_1371_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg0_0 = q_in_5[((__q_in_5_Cell_stride_0_0 * ((- __q_in_5_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_5_K_stride_0_0 * ((- __q_in_5_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0_0 = gtir_tmp_1371_0;
                double __tlet_arg0_1 = gtir_tmp_1369_0;
                double __tlet_arg0_2 = gtir_tmp_1370_0;
                double __tlet_arg1 = gtir_tmp_1380_0;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_568;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_758_multiplies_fused_tlet_759_minus_fused_tlet_760_maximum_fused_tlet_761_greater_fused_tlet_763_greater_fused_tlet_764_and_)
                __tlet_result = ((__tlet_arg0 > max(__tlet_arg0_1, (__tlet_arg0_2 - (__tlet_arg0_0_0 * __tlet_arg1_0)))) && (__tlet_arg0_0 > __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_1384 = __tlet_result;
            }
            if_stmt_67_168_0_52(__map_fusion_gtir_tmp_1384, __map_fusion_gtir_tmp_568, &p[0], &q_in_5[0], &rho[0], &te[0], gtir_tmp_1409, __p_Cell_range_0, __p_K_range_0, __p_K_stride_0_0, __q_in_5_Cell_range_0, __q_in_5_Cell_stride_0_0, __q_in_5_K_range_0, __q_in_5_K_stride_0_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0_0, __te_Cell_range_0, __te_K_range_0, __te_K_stride_0_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1409, &__arg1______6ac68c6e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1______6ac68c6e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6ac690a6_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_416_get_value__clone_5cfffffc_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_714_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_706;
                double __tlet_arg1 = gtir_tmp_714_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_417_minimum_fused_tlet_418_neg)
                __tlet_result = (- min(__tlet_arg0, __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_718 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_718, &__arg1____________________________6a2de450_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1____________________________6a2de450_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6a2de87e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_469_get_value__clone_5dc75764_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_808_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_806;
                double __tlet_arg1 = gtir_tmp_808_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_470_maximum)
                __tlet_result = max(__tlet_arg0, __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_810 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_810, &__arg1___________________________6a79d2f2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1___________________________6a79d2f2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6a79d720_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_472_get_value__clone_5dc0e7da_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_814_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_806;
                double __tlet_arg1 = gtir_tmp_814_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_473_minimum_fused_tlet_474_neg)
                __tlet_result = (- min(__tlet_arg0, __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_818 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_818, &__arg1___________________________6b13ff4e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1___________________________6b13ff4e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6b140480_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

        }
    } else {
        {
            double __arg2____;
            double __arg2_____699297c0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg2___________69e0b2c0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg2_____6a2dea7c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg2_____6a79d932_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg2___6ac692d6_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg2_____6b1406ce_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double gtir_tmp_1411_0;
            double gtir_tmp_1332_0;
            double gtir_tmp_811_0;
            double gtir_tmp_819_0;
            double gtir_tmp_1148_0;
            double gtir_tmp_719_0;
            double gtir_tmp_711_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_780_get_value__clone_60f4e488_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1411_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1411_0, &__arg2___6ac692d6_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2___6ac692d6_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6ac690a6_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_740_get_value__clone_60d732bc_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1332_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1332_0, &__arg2___________69e0b2c0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2___________69e0b2c0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_69e0b0d6_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_471_get_value__clone_5dba794a_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_811_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_811_0, &__arg2_____6a79d932_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_____6a79d932_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6a79d720_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_475_get_value__clone_5dcd10f0_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_819_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_819_0, &__arg2_____6b1406ce_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_____6b1406ce_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6b140480_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_644_get_value__clone_5fefd1f6_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1148_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1148_0, &__arg2_____699297c0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_____699297c0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_699295a4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_419_get_value__clone_5d05d58a_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_719_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_719_0, &__arg2_____6a2dea7c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_____6a2dea7c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6a2de87e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_415_get_value__clone_5d10b6b2_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_711_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_711_0, &__arg2____, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2____, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_63_3_5_167(const double&  __arg2__________, const double&  __arg2___________706b03a2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const double&  __arg2___________70b8380c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const double&  __arg2___________7104a764_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const bool&  __cond, const double&  __map_fusion_gtir_tmp_1337, const double&  __map_fusion_gtir_tmp_328, const double&  gtir_tmp_1273, const double&  gtir_tmp_1324, const double&  gtir_tmp_1333, const double&  gtir_tmp_500, double&  __output, double&  __output_706b01a4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_70b835f0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_7104a566_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion) {

    if (__cond) {
        {
            double __arg1_____________;
            double __arg1______________706afd80_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1______________70b83078_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1______________7104a14c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __map_fusion_gtir_tmp_1366;
            double __map_fusion_gtir_tmp_1348;
            double __map_fusion_gtir_tmp_1360;
            double __map_fusion_gtir_tmp_1354;

            {
                double __tlet_arg0 = gtir_tmp_1324;
                double __tlet_arg1 = gtir_tmp_1333;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_1337;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_753_multiplies_fused_tlet_754_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1366 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1366, &__arg1______________706afd80_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1______________706afd80_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_706b01a4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_arg0 = gtir_tmp_500;
                double __tlet_arg1 = gtir_tmp_1333;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_1337;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_747_multiplies_fused_tlet_748_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1348 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1348, &__arg1_____________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_____________, &__output, 1);
            {
                double __tlet_arg0 = gtir_tmp_1273;
                double __tlet_arg1 = gtir_tmp_1333;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_1337;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_751_multiplies_fused_tlet_752_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1360 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1360, &__arg1______________7104a14c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1______________7104a14c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_7104a566_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_arg0 = __map_fusion_gtir_tmp_328;
                double __tlet_arg1 = gtir_tmp_1333;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_1337;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_749_multiplies_fused_tlet_750_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1354 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1354, &__arg1______________70b83078_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1______________70b83078_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_70b835f0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

        }
    } else {
        {


            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2__________, &__output, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2___________706b03a2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_706b01a4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2___________70b8380c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_70b835f0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2___________7104a764_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_7104a566_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

        }
    }
}

DACE_DFI void if_stmt_80_3_5_104(const double&  __arg2__________, const bool&  __cond, const double&  gtir_tmp_1349, const double&  gtir_tmp_1355, const double&  gtir_tmp_1361, const double&  gtir_tmp_1367, double&  __output) {

    if (__cond) {
        {
            double __arg1_____________;
            double __map_fusion_gtir_tmp_1562;

            {
                double __tlet_arg0 = gtir_tmp_1349;
                double __tlet_arg1 = gtir_tmp_1367;
                double __tlet_arg1_0 = gtir_tmp_1355;
                double __tlet_arg1_1 = gtir_tmp_1361;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_847_plus_fused_tlet_848_plus_fused_tlet_849_plus)
                __tlet_result = (((__tlet_arg0 + __tlet_arg1_0) + __tlet_arg1_1) + __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1562 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1562, &__arg1_____________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_____________, &__output, 1);

        }
    } else {
        {


            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2__________, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_52_3_5_191(const bool&  __cond, const double&  gtir_tmp_1149, const double&  gtir_tmp_1177, const double&  gtir_tmp_1412, const double&  gtir_tmp_720, const double&  gtir_tmp_820, const double&  gtir_tmp_822_0, double&  __output, double&  __output_6b619768_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion) {

    if (__cond) {
        {
            double __arg1_______________;
            double __arg1___________6b6192cc_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __map_fusion_gtir_tmp_1184;
            double __map_fusion_gtir_tmp_1419;

            {
                double __tlet_arg0 = gtir_tmp_720;
                double __tlet_arg1 = gtir_tmp_822_0;
                double __tlet_arg1_0 = gtir_tmp_1177;
                double __tlet_arg1_1 = gtir_tmp_1149;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_660_plus_fused_tlet_661_plus_fused_tlet_662_plus)
                __tlet_result = (((__tlet_arg0 + __tlet_arg1_1) + __tlet_arg1_0) + __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1184 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1184, &__arg1_______________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_______________, &__output, 1);
            {
                double __tlet_arg0 = gtir_tmp_820;
                double __tlet_arg1 = gtir_tmp_822_0;
                double __tlet_arg1_0 = gtir_tmp_1412;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_781_plus_fused_tlet_782_plus_fused_tlet_783_plus)
                __tlet_result = (((__tlet_arg0 + __tlet_arg1_0) + __tlet_arg1) + __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1419 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1419, &__arg1___________6b6192cc_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1___________6b6192cc_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6b619768_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

        }
    } else {
        {
            double __arg2____________;
            double __arg2_________6b619970_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double gtir_tmp_1420_0;
            double gtir_tmp_1185_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_784_get_value__clone_6164c7e4_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1420_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1420_0, &__arg2_________6b619970_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_________6b619970_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6b619768_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_663_get_value__clone_5ffc3162_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1185_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1185_0, &__arg2____________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2____________, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_53_3_5_149(const double&  __arg2____________, const double&  __arg2_____________6f837910_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const double&  __arg2_____________6fd1726e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const double&  __arg2_____________701e92e2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const bool&  __cond, const double&  __map_fusion_gtir_tmp_1190, const double&  gtir_tmp_1149, const double&  gtir_tmp_1177, const double&  gtir_tmp_1186, const double&  gtir_tmp_720, const double&  gtir_tmp_822_0, double&  __output, double&  __output_6f8376b8_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_6fd17034_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_701e90d0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion) {

    if (__cond) {
        {
            double __arg1_______________;
            double __arg1________________6f837172_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1________________6fd16b7a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1________________701e8ca2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __map_fusion_gtir_tmp_1213;
            double __map_fusion_gtir_tmp_1201;
            double __map_fusion_gtir_tmp_1538;
            double __map_fusion_gtir_tmp_1207;

            {
                double __tlet_arg0 = gtir_tmp_1177;
                double __tlet_arg1 = gtir_tmp_1186;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_1190;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_674_multiplies_fused_tlet_675_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1213 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1213, &__arg1________________701e8ca2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1________________701e8ca2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_701e90d0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_arg0 = gtir_tmp_720;
                double __tlet_arg1 = gtir_tmp_1186;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_1190;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_670_multiplies_fused_tlet_671_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1201 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1201, &__arg1_______________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_______________, &__output, 1);
            {
                double __tlet_arg0 = gtir_tmp_822_0;
                double __tlet_arg1 = gtir_tmp_1186;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_1190;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_837_multiplies_fused_tlet_838_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1538 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1538, &__arg1________________6fd16b7a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1________________6fd16b7a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6fd17034_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_arg0 = gtir_tmp_1149;
                double __tlet_arg1 = gtir_tmp_1186;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_1190;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_672_multiplies_fused_tlet_673_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1207 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1207, &__arg1________________6f837172_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1________________6f837172_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6f8376b8_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

        }
    } else {
        {


            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2____________, &__output, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_____________6f837910_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6f8376b8_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_____________6fd1726e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6fd17034_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_____________701e92e2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_701e90d0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

        }
    }
}

DACE_DFI void if_stmt_79_3_5_262(const double&  __arg2___________, const bool&  __cond, const double&  gtir_tmp_1202, const double&  gtir_tmp_1208, const double&  gtir_tmp_1214, const double&  gtir_tmp_1539, double&  __output) {

    if (__cond) {
        {
            double __arg1______________;
            double __map_fusion_gtir_tmp_1542;

            {
                double __tlet_arg0 = gtir_tmp_1202;
                double __tlet_arg1 = gtir_tmp_1539;
                double __tlet_arg1_0 = gtir_tmp_1208;
                double __tlet_arg1_1 = gtir_tmp_1214;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_835_plus_fused_tlet_836_plus_fused_tlet_839_plus)
                __tlet_result = (((__tlet_arg0 + __tlet_arg1_0) + __tlet_arg1_1) + __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1542 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1542, &__arg1______________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1______________, &__output, 1);

        }
    } else {
        {


            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2___________, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_70_3_5_96(const double&  __arg2________, const double&  __arg2________719e02ba_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_71ea41ac_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const double&  __arg2_________7151735a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const double&  __arg2_________71ea3c5c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const bool&  __cond, const double&  __map_fusion_gtir_tmp_1425, const double&  gtir_tmp_1412, const double&  gtir_tmp_1421, const double&  gtir_tmp_820, const double&  gtir_tmp_822_0, double&  __output, double&  __output_71517166_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_719e00a8_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_71ea3ff4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_71ea3a2c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion) {

    if (__cond) {
        {
            double __arg1__________;
            double __arg1___________71516d38_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1___________71ea359a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1__________719dfc5c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_71ea3e1e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __map_fusion_gtir_tmp_1436;
            double __map_fusion_gtir_tmp_1442;
            double __map_fusion_gtir_tmp_1588;
            double __map_fusion_gtir_tmp_1596;

            {
                double __tlet_arg0 = gtir_tmp_820;
                double __tlet_arg1 = gtir_tmp_1421;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_1425;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_791_multiplies_fused_tlet_792_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1436 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1436, &__arg1__________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1__________, &__output, 1);
            {
                double __tlet_arg0 = gtir_tmp_1412;
                double __tlet_arg1 = gtir_tmp_1421;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_1425;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_793_multiplies_fused_tlet_794_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1442 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1442, &__arg1___________71516d38_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1___________71516d38_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_71517166_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_arg0 = gtir_tmp_822_0;
                double __tlet_arg1 = gtir_tmp_1421;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_1425;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_861_multiplies_fused_tlet_862_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1588 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1588, &__arg1___________71ea359a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1___________71ea359a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_71ea3a2c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_arg0 = gtir_tmp_822_0;
                double __tlet_arg1 = gtir_tmp_1421;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_1425;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_864_multiplies_fused_tlet_865_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1596 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1596, &__arg1__________719dfc5c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_71ea3e1e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1__________719dfc5c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_71ea3e1e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_719e00a8_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_71ea3ff4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

        }
    } else {
        {


            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2________, &__output, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_________7151735a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_71517166_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_________71ea3c5c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_71ea3a2c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2________719e02ba_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_71ea41ac_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_719e00a8_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_71ea3ff4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

        }
    }
}

DACE_DFI void if_stmt_83_3_5_54(const double&  __arg2______, const bool&  __cond, const double&  gtir_tmp_1437, const double&  gtir_tmp_1443, const double&  gtir_tmp_1589, const double&  gtir_tmp_1597, double&  __output) {

    if (__cond) {
        {
            double __arg1__________________________;
            double __map_fusion_gtir_tmp_1600;

            {
                double __tlet_arg0 = gtir_tmp_1437;
                double __tlet_arg1 = gtir_tmp_1597;
                double __tlet_arg1_0 = gtir_tmp_1589;
                double __tlet_arg1_1 = gtir_tmp_1443;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_860_plus_fused_tlet_863_plus_fused_tlet_866_plus)
                __tlet_result = (((__tlet_arg0 + __tlet_arg1_1) + __tlet_arg1_0) + __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1600 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1600, &__arg1__________________________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1__________________________, &__output, 1);

        }
    } else {
        {


            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2______, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_28_3_5_163(const bool&  __cond, const double&  __map_fusion_gtir_tmp_323, const double&  gtir_tmp_1073, const double&  gtir_tmp_548, const double&  gtir_tmp_712, const double&  gtir_tmp_812, const double&  gtir_tmp_822_0, const double&  gtir_tmp_869, const double&  gtir_tmp_872, const double&  gtir_tmp_875, const double&  gtir_tmp_979, double&  __output, double&  __output_7237c15c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_728633d2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion) {

    if (__cond) {
        {
            double __arg1____;
            double __arg1________________________7237bca2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1_________________________72862fae_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __map_fusion_gtir_tmp_833;
            double __map_fusion_gtir_tmp_1080;
            double __map_fusion_gtir_tmp_882;

            {
                double __tlet_arg0 = gtir_tmp_712;
                double __tlet_arg1 = gtir_tmp_822_0;
                double __tlet_arg1_0 = gtir_tmp_812;
                double __tlet_arg1_1 = gtir_tmp_548;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_480_plus_fused_tlet_481_plus_fused_tlet_482_plus)
                __tlet_result = (((__tlet_arg0 + __tlet_arg1_1) + __tlet_arg1_0) + __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_833 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_833, &__arg1____, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1____, &__output, 1);
            {
                double __tlet_arg0 = gtir_tmp_979;
                double __tlet_arg1 = gtir_tmp_822_0;
                double __tlet_arg1_0 = gtir_tmp_1073;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_605_plus_fused_tlet_606_plus_fused_tlet_607_plus)
                __tlet_result = (((__tlet_arg0 + __tlet_arg1_0) + __tlet_arg1) + __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1080 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1080, &__arg1________________________7237bca2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1________________________7237bca2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_7237c15c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_arg0 = gtir_tmp_869;
                double __tlet_arg1 = gtir_tmp_875;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_323;
                double __tlet_arg1_1 = gtir_tmp_872;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_500_plus_fused_tlet_501_plus_fused_tlet_502_plus)
                __tlet_result = (((__tlet_arg0 + __tlet_arg1_1) + __tlet_arg1_0) + __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_882 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_882, &__arg1_________________________72862fae_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_________________________72862fae_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_728633d2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

        }
    } else {
        {
            double __arg2_____;
            double __arg2________________7237c35a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg2_________________728635da_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double gtir_tmp_1081_0;
            double gtir_tmp_834_0;
            double gtir_tmp_883_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_608_get_value__clone_5f3e5ffc_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1081_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1081_0, &__arg2________________7237c35a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2________________7237c35a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_7237c15c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_483_get_value__clone_5ddd24a4_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_834_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_834_0, &__arg2_____, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_____, &__output, 1);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_503_get_value__clone_5dfa868e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_883_0 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_883_0, &__arg2_________________728635da_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_________________728635da_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_728633d2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

        }
    }
}

DACE_DFI void if_stmt_29_3_5_180(const double&  __arg2_____, const double&  __arg2______6c98246c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const double&  __arg2______6ce4fda0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const double&  __arg2______6d31d6ac_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const bool&  __cond, const double&  __map_fusion_gtir_tmp_839, const double&  gtir_tmp_548, const double&  gtir_tmp_712, const double&  gtir_tmp_812, const double&  gtir_tmp_822_0, const double&  gtir_tmp_835, double&  __output, double&  __output_6c982264_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_6ce4fb7a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_6d31d4a4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion) {

    if (__cond) {
        {
            double __arg1____;
            double __arg1_____6c981e40_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1_____6ce4f6de_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1_____6d31d09e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __map_fusion_gtir_tmp_856;
            double __map_fusion_gtir_tmp_850;
            double __map_fusion_gtir_tmp_1459;
            double __map_fusion_gtir_tmp_862;

            {
                double __tlet_arg0 = gtir_tmp_548;
                double __tlet_arg1 = gtir_tmp_835;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_839;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_492_multiplies_fused_tlet_493_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_856 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_856, &__arg1_____6d31d09e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_____6d31d09e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6d31d4a4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_arg0 = gtir_tmp_712;
                double __tlet_arg1 = gtir_tmp_835;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_839;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_490_multiplies_fused_tlet_491_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_850 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_850, &__arg1____, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1____, &__output, 1);
            {
                double __tlet_arg0 = gtir_tmp_822_0;
                double __tlet_arg1 = gtir_tmp_835;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_839;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_801_multiplies_fused_tlet_802_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1459 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1459, &__arg1_____6c981e40_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_____6c981e40_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6c982264_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_arg0 = gtir_tmp_812;
                double __tlet_arg1 = gtir_tmp_835;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_839;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_494_multiplies_fused_tlet_495_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_862 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_862, &__arg1_____6ce4f6de_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_____6ce4f6de_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6ce4fb7a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

        }
    } else {
        {


            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_____, &__output, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2______6c98246c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6c982264_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2______6ce4fda0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6ce4fb7a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2______6d31d6ac_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6d31d4a4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

        }
    }
}

DACE_DFI void if_stmt_73_3_5_297(const double&  __arg2_________________, const bool&  __cond, const double&  gtir_tmp_1460, const double&  gtir_tmp_851, const double&  gtir_tmp_857, const double&  gtir_tmp_863, double&  __output) {

    if (__cond) {
        {
            double __arg1_________________________;
            double __map_fusion_gtir_tmp_1463;

            {
                double __tlet_arg0 = gtir_tmp_851;
                double __tlet_arg1 = gtir_tmp_1460;
                double __tlet_arg1_0 = gtir_tmp_863;
                double __tlet_arg1_1 = gtir_tmp_857;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_799_plus_fused_tlet_800_plus_fused_tlet_803_plus)
                __tlet_result = (((__tlet_arg0 + __tlet_arg1_1) + __tlet_arg1_0) + __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1463 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1463, &__arg1_________________________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_________________________, &__output, 1);

        }
    } else {
        {


            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_________________, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_46_3_5_211(const double&  __arg2_______________, const double&  __arg2_______________6ee729b6_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const double&  __arg2________________6e98aa52_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const double&  __arg2________________6f34a966_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const bool&  __cond, const double&  __map_fusion_gtir_tmp_1086, const double&  gtir_tmp_1073, const double&  gtir_tmp_1082, const double&  gtir_tmp_822_0, const double&  gtir_tmp_979, double&  __output, double&  __output_6e98a872_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_6ee7279a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_6f34a768_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion) {

    if (__cond) {
        {
            double __arg1_______________________;
            double __arg1________________________6e98a408_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1__________________6ee7236c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1________________________6f34a308_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __map_fusion_gtir_tmp_1097;
            double __map_fusion_gtir_tmp_1502;
            double __map_fusion_gtir_tmp_1103;
            double __map_fusion_gtir_tmp_1510;

            {
                double __tlet_arg0 = gtir_tmp_979;
                double __tlet_arg1 = gtir_tmp_1082;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_1086;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_615_multiplies_fused_tlet_616_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1097 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1097, &__arg1_______________________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_______________________, &__output, 1);
            {
                double __tlet_arg0 = gtir_tmp_822_0;
                double __tlet_arg1 = gtir_tmp_1082;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_1086;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_821_multiplies_fused_tlet_822_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1502 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1502, &__arg1________________________6e98a408_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1________________________6e98a408_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6e98a872_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_arg0 = gtir_tmp_1073;
                double __tlet_arg1 = gtir_tmp_1082;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_1086;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_617_multiplies_fused_tlet_618_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1103 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1103, &__arg1________________________6f34a308_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1________________________6f34a308_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6f34a768_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_arg0 = gtir_tmp_822_0;
                double __tlet_arg1 = gtir_tmp_1082;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_1086;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_824_multiplies_fused_tlet_825_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1510 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1510, &__arg1__________________6ee7236c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1__________________6ee7236c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6ee7279a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

        }
    } else {
        {


            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_______________, &__output, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2________________6e98aa52_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6e98a872_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_______________6ee729b6_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6ee7279a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2________________6f34a966_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6f34a768_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

        }
    }
}

DACE_DFI void if_stmt_77_3_5_108(const double&  __arg2_____________, const bool&  __cond, const double&  gtir_tmp_1098, const double&  gtir_tmp_1104, const double&  gtir_tmp_1503, const double&  gtir_tmp_1511, double&  __output) {

    if (__cond) {
        {
            double __arg1________________;
            double __map_fusion_gtir_tmp_1514;

            {
                double __tlet_arg0 = gtir_tmp_1098;
                double __tlet_arg1 = gtir_tmp_1511;
                double __tlet_arg1_0 = gtir_tmp_1104;
                double __tlet_arg1_1 = gtir_tmp_1503;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_820_plus_fused_tlet_823_plus_fused_tlet_826_plus)
                __tlet_result = (((__tlet_arg0 + __tlet_arg1_0) + __tlet_arg1_1) + __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1514 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1514, &__arg1________________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1________________, &__output, 1);

        }
    } else {
        {


            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_____________, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_36_3_5_259(const double&  __arg2________________, const double&  __arg2_________________6d7ee0d2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6e181c2a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const double&  __arg2_________________6dcb292e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6e1820d0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const double&  __arg2_________________6e1816e4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const bool&  __cond, const double&  __map_fusion_gtir_tmp_323, const double&  __map_fusion_gtir_tmp_888, const double&  gtir_tmp_869, const double&  gtir_tmp_872, const double&  gtir_tmp_875, const double&  gtir_tmp_884, double&  __output, double&  __output_6d7edede_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6e181a7c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_6dcb273a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6e181f54_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_6e1814a0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion) {

    if (__cond) {
        {
            double __arg1________________________;
            double __arg1_________________________6e18105e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1_________________________6d7eda88_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6e1818c4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1_________________________6dcb232a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6e181dba_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __map_fusion_gtir_tmp_899;
            double __map_fusion_gtir_tmp_911;
            double __map_fusion_gtir_tmp_917;
            double __map_fusion_gtir_tmp_905;

            {
                double __tlet_arg0 = gtir_tmp_869;
                double __tlet_arg1 = gtir_tmp_884;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_888;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_510_multiplies_fused_tlet_511_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_899 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_899, &__arg1________________________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1________________________, &__output, 1);
            {
                double __tlet_arg0 = __map_fusion_gtir_tmp_323;
                double __tlet_arg1 = gtir_tmp_884;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_888;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_514_multiplies_fused_tlet_515_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_911 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_911, &__arg1_________________________6d7eda88_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6e1818c4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_________________________6d7eda88_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6e1818c4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6d7edede_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6e181a7c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_arg0 = gtir_tmp_875;
                double __tlet_arg1 = gtir_tmp_884;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_888;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_516_multiplies_fused_tlet_517_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_917 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_917, &__arg1_________________________6dcb232a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6e181dba_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_________________________6dcb232a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6e181dba_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6dcb273a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6e181f54_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_arg0 = gtir_tmp_872;
                double __tlet_arg1 = gtir_tmp_884;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_888;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_512_multiplies_fused_tlet_513_divides)
                __tlet_result = ((__tlet_arg0 * __tlet_arg1_0) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_905 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_905, &__arg1_________________________6e18105e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_________________________6e18105e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6e1814a0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

        }
    } else {
        {


            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2________________, &__output, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_________________6e1816e4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6e1814a0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_________________6d7ee0d2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6e181c2a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6d7edede_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6e181a7c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_________________6dcb292e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6e1820d0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6dcb273a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6e181f54_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

        }
    }
}

DACE_DFI void if_stmt_74_3_5_115(const double&  __arg2________________, const bool&  __cond, const double&  gtir_tmp_900, const double&  gtir_tmp_906, const double&  gtir_tmp_912, const double&  gtir_tmp_918, double&  __output) {

    if (__cond) {
        {
            double __arg1________________________;
            double __map_fusion_gtir_tmp_1480;

            {
                double __tlet_arg0 = gtir_tmp_900;
                double __tlet_arg1 = gtir_tmp_918;
                double __tlet_arg1_0 = gtir_tmp_912;
                double __tlet_arg1_1 = gtir_tmp_906;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_809_plus_fused_tlet_810_plus_fused_tlet_811_plus)
                __tlet_result = (((__tlet_arg0 + __tlet_arg1_1) + __tlet_arg1_0) + __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1480 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1480, &__arg1________________________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1________________________, &__output, 1);

        }
    } else {
        {


            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2________________, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_100_0_0_16(const double&  __arg2, const double&  __arg2_676cc204_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const double&  __arg2_67cea33e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const double&  __arg2_6853672c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const double&  __arg2_68f55a78_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6943e3d2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const double&  __arg2_6943dd9c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const double&  __arg2__68a3b5a6_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, const bool&  __cond, const double* __restrict__ p, const double*  q_in_0, const double*  q_in_1, const double*  q_in_2, const double*  q_in_3, const double*  q_in_4, const double*  q_in_5, const double* __restrict__ rho, const double*  te, double&  __output, double&  __output_676cbfd4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_67cea0fa_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_685364e8_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_68a3b376_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_68f5582a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6943e1f2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, double&  __output_6943db80_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, int __p_Cell_range_0, int __p_K_range_0, int __p_K_stride_0, int __q_in_0_Cell_range_0, int __q_in_0_Cell_stride_0, int __q_in_0_K_range_0, int __q_in_0_K_stride_0, int __q_in_1_Cell_range_0, int __q_in_1_Cell_stride_0, int __q_in_1_K_range_0, int __q_in_1_K_stride_0, int __q_in_2_Cell_range_0, int __q_in_2_Cell_stride_0, int __q_in_2_K_range_0, int __q_in_2_K_stride_0, int __q_in_3_Cell_range_0, int __q_in_3_Cell_stride_0, int __q_in_3_K_range_0, int __q_in_3_K_stride_0, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride_0, int __q_in_4_K_range_0, int __q_in_4_K_stride_0, int __q_in_5_Cell_range_0, int __q_in_5_Cell_stride_0, int __q_in_5_K_range_0, int __q_in_5_K_stride_0, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride_0, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride_0, int64_t i_Cell_gtx_horizontal, int64_t i_K_gtx_vertical) {
    bool __map_fusion_gtir_tmp_465;
    double __map_fusion_gtir_tmp_390;
    double gtir_tmp_187;
    bool __map_fusion_gtir_tmp_210;
    double __map_fusion_gtir_tmp_102;
    double gtir_tmp_500;
    double gtir_tmp_869;
    double gtir_tmp_556;
    double gtir_tmp_206;
    double gtir_tmp_548;
    double gtir_tmp_448;
    double gtir_tmp_373;
    double gtir_tmp_350;
    double gtir_tmp_290;
    double gtir_tmp_506;
    double gtir_tmp_492;
    double gtir_tmp_875;
    bool __map_fusion_gtir_tmp_212;
    double __map_fusion_gtir_tmp_398;
    bool __map_fusion_gtir_tmp_52;
    double gtir_tmp_872;

    if (__cond) {
        {
            double gtir_tmp_50_0;
            double gtir_tmp_375_0;
            double gtir_tmp_85_0;
            double __map_fusion_gtir_tmp_100;
            double gtir_tmp_80_0;
            double gtir_tmp_376_0;
            double gtir_tmp_377_0;
            double gtir_tmp_466_0;
            bool __map_fusion_gtir_tmp_468;
            double gtir_tmp_94_0;
            double gtir_tmp_208_0;
            double gtir_tmp_79_0;
            double gtir_tmp_394_0;
            double gtir_tmp_378_0;
            double gtir_tmp_391_0;
            double gtir_tmp_78_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_226_get_value__clone_5aad197e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_378_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_235_get_value__clone_5ac64e80_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-09;
                ///////////////////

                gtir_tmp_394_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_47_get_value__clone_58d32d46_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 21.875;
                ///////////////////

                gtir_tmp_79_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_46_get_value__clone_58ca63d2_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 610.78;
                ///////////////////

                gtir_tmp_78_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_125_get_value__clone_59893ee2_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_208_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1 = gtir_tmp_208_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_126_less)
                __tlet_result = (__tlet_arg0 < __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_210 = __tlet_result;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_233_get_value__clone_5ac10db2_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-12;
                ///////////////////

                gtir_tmp_391_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_224_get_value__clone_5ab7377e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 5.0;
                ///////////////////

                gtir_tmp_376_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_48_get_value__clone_58c2080e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_80_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_223_get_value__clone_5abbfa7a_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 250000.0;
                ///////////////////

                gtir_tmp_375_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_225_get_value__clone_5ab20218_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.304;
                ///////////////////

                gtir_tmp_377_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_375_0;
                double __tlet_arg0_0 = gtir_tmp_376_0;
                double __tlet_arg0_0_0 = gtir_tmp_378_0;
                double __tlet_arg0_1 = gtir_tmp_377_0;
                double __tlet_arg1 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_227_minus_fused_tlet_228_multiplies_fused_tlet_229_exp_fused_tlet_230_multiplies_fused_tlet_231_minimum_fused_tlet_232_divides)
                __tlet_result = (min(__tlet_arg0, (__tlet_arg0_0 * dace::math::exp((__tlet_arg0_1 * (__tlet_arg0_0_0 - __tlet_arg1_0))))) / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_390 = __tlet_result;
            }
            {
                double __tlet_arg0 = gtir_tmp_391_0;
                double __tlet_arg0_0 = q_in_4[((__q_in_4_Cell_stride_0 * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride_0 * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_394_0;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_390;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_234_divides_fused_tlet_236_minimum_fused_tlet_237_maximum)
                __tlet_result = max(__tlet_arg0, min((__tlet_arg0_0 / __tlet_arg1_0), __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_398 = __tlet_result;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_56_get_value__clone_58c630b4_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 461.51;
                ///////////////////

                gtir_tmp_94_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_51_get_value__clone_58cecc42_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 7.66;
                ///////////////////

                gtir_tmp_85_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_78_0;
                double __tlet_arg0_0 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg0_0_0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg0_1 = gtir_tmp_79_0;
                double __tlet_arg0_2 = te[(((- __te_Cell_range_0) + (__te_K_stride_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1 = te[(((- __te_Cell_range_0) + (__te_K_stride_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_0 = gtir_tmp_85_0;
                double __tlet_arg1_1 = gtir_tmp_94_0;
                double __tlet_arg1_2 = gtir_tmp_80_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_49_minus_fused_tlet_50_multiplies_fused_tlet_52_minus_fused_tlet_57_multiplies_fused_tlet_53_divides_fused_tlet_54_exp_fused_tlet_58_multiplies_fused_tlet_55_multiplies_fused_tlet_59_divides)
                __tlet_result = ((__tlet_arg0 * dace::math::exp(((__tlet_arg0_1 * (__tlet_arg0_0_0 - __tlet_arg1_2)) / (__tlet_arg0_2 - __tlet_arg1_0)))) / ((__tlet_arg0_0 * __tlet_arg1_1) * __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_100 = __tlet_result;
            }
            if_stmt_7_3_0_18(__map_fusion_gtir_tmp_210, __map_fusion_gtir_tmp_100, &te[0], gtir_tmp_448, __te_Cell_range_0, __te_K_range_0, __te_K_stride_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);
            {
                double __tlet_arg0 = q_in_0[((__q_in_0_Cell_stride_0 * ((- __q_in_0_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_0_K_stride_0 * ((- __q_in_0_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = __map_fusion_gtir_tmp_100;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_60_minus)
                __tlet_result = (__tlet_arg0 - __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_102 = __tlet_result;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_29_get_value__clone_58a79c76_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_50_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_278_get_value__clone_5b6c479a_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_466_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_4[((__q_in_4_Cell_stride_0 * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride_0 * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_466_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_279_greater)
                __tlet_result = (__tlet_arg0 > __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_468 = __tlet_result;
            }
            if_stmt_9_3_0_43(__map_fusion_gtir_tmp_468, __map_fusion_gtir_tmp_102, __map_fusion_gtir_tmp_398, gtir_tmp_448, &q_in_4[0], &rho[0], gtir_tmp_492, __q_in_4_Cell_range_0, __q_in_4_Cell_stride_0, __q_in_4_K_range_0, __q_in_4_K_stride_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);
            {
                double __tlet_arg0 = q_in_5[((__q_in_5_Cell_stride_0 * ((- __q_in_5_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_5_K_stride_0 * ((- __q_in_5_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0 = q_in_4[((__q_in_4_Cell_stride_0 * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride_0 * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_50_0;
                double __tlet_arg1_0 = q_in_3[((__q_in_3_Cell_stride_0 * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride_0 * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))];
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_27_maximum_fused_tlet_28_maximum_fused_tlet_30_greater)
                __tlet_result = (max(__tlet_arg0, max(__tlet_arg0_0, __tlet_arg1_0)) > __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_52 = __tlet_result;
            }
            {
                bool __tlet_arg0 = __map_fusion_gtir_tmp_210;
                bool __tlet_arg1 = __map_fusion_gtir_tmp_52;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_277_and_)
                __tlet_result = (__tlet_arg0 && __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_465 = __tlet_result;
            }

        }
        if (__map_fusion_gtir_tmp_465) {
            {
                double __arg1_____________________________;
                double __arg1______________________________6c4aa9a8_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
                double gtir_tmp_494_0;
                double __map_fusion_gtir_tmp_498;
                double gtir_tmp_502_0;
                double __map_fusion_gtir_tmp_504;

                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_293_get_value__clone_5b7e249c_1c88_11f1_a73a_1f4e8615a1c5)
                    __tlet_out = 0.0;
                    ///////////////////

                    gtir_tmp_494_0 = __tlet_out;
                }
                {
                    double __tlet_arg0 = gtir_tmp_492;
                    double __tlet_arg1 = gtir_tmp_494_0;
                    double __tlet_result;

                    ///////////////////
                    // Tasklet code (tlet_294_minimum_fused_tlet_295_neg)
                    __tlet_result = (- min(__tlet_arg0, __tlet_arg1));
                    ///////////////////

                    __map_fusion_gtir_tmp_498 = __tlet_result;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__map_fusion_gtir_tmp_498, &__arg1_____________________________, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__arg1_____________________________, &gtir_tmp_500, 1);
                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_297_get_value__clone_5bfe9014_1c88_11f1_a73a_1f4e8615a1c5)
                    __tlet_out = 0.0;
                    ///////////////////

                    gtir_tmp_502_0 = __tlet_out;
                }
                {
                    double __tlet_arg0 = gtir_tmp_492;
                    double __tlet_arg1 = gtir_tmp_502_0;
                    double __tlet_result;

                    ///////////////////
                    // Tasklet code (tlet_298_maximum)
                    __tlet_result = max(__tlet_arg0, __tlet_arg1);
                    ///////////////////

                    __map_fusion_gtir_tmp_504 = __tlet_result;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__map_fusion_gtir_tmp_504, &__arg1______________________________6c4aa9a8_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__arg1______________________________6c4aa9a8_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &gtir_tmp_506, 1);

            }
        } else {
            {
                double __arg2_________;
                double __arg2__________6c4aafde_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
                double gtir_tmp_505_0;
                double gtir_tmp_499_0;

                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_299_get_value__clone_5bec442c_1c88_11f1_a73a_1f4e8615a1c5)
                    __tlet_out = 0.0;
                    ///////////////////

                    gtir_tmp_505_0 = __tlet_out;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_505_0, &__arg2__________6c4aafde_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__arg2__________6c4aafde_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &gtir_tmp_506, 1);
                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_296_get_value__clone_5b780e18_1c88_11f1_a73a_1f4e8615a1c5)
                    __tlet_out = 0.0;
                    ///////////////////

                    gtir_tmp_499_0 = __tlet_out;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_499_0, &__arg2_________, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__arg2_________, &gtir_tmp_500, 1);

            }
        }
        {

            if_stmt_13_3_2_8(__map_fusion_gtir_tmp_210, __map_fusion_gtir_tmp_102, __map_fusion_gtir_tmp_390, gtir_tmp_506, &q_in_1[0], &q_in_4[0], &te[0], gtir_tmp_548, __q_in_1_Cell_range_0, __q_in_1_Cell_stride_0, __q_in_1_K_range_0, __q_in_1_K_stride_0, __q_in_4_Cell_range_0, __q_in_4_Cell_stride_0, __q_in_4_K_range_0, __q_in_4_K_stride_0, __te_Cell_range_0, __te_K_range_0, __te_K_stride_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);

        }
        if (__map_fusion_gtir_tmp_210) {
            {
                double __arg1____________________________;
                double gtir_tmp_550_0;
                double __map_fusion_gtir_tmp_554;

                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_324_get_value__clone_5c0a2910_1c88_11f1_a73a_1f4e8615a1c5)
                    __tlet_out = 30.0;
                    ///////////////////

                    gtir_tmp_550_0 = __tlet_out;
                }
                {
                    double __tlet_arg0 = gtir_tmp_548;
                    double __tlet_arg0_0 = __map_fusion_gtir_tmp_102;
                    double __tlet_arg1 = gtir_tmp_550_0;
                    double __tlet_result;

                    ///////////////////
                    // Tasklet code (tlet_325_divides_fused_tlet_326_minimum)
                    __tlet_result = min(__tlet_arg0, (__tlet_arg0_0 / __tlet_arg1));
                    ///////////////////

                    __map_fusion_gtir_tmp_554 = __tlet_result;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__map_fusion_gtir_tmp_554, &__arg1____________________________, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__arg1____________________________, &gtir_tmp_556, 1);

            }
        } else {
            {
                double __arg2__________________;
                double gtir_tmp_555_0;

                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_327_get_value__clone_5c048de8_1c88_11f1_a73a_1f4e8615a1c5)
                    __tlet_out = 0.0;
                    ///////////////////

                    gtir_tmp_555_0 = __tlet_out;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_555_0, &__arg2__________________, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__arg2__________________, &gtir_tmp_556, 1);

            }
        }
        {
            bool __map_fusion_gtir_tmp_191;
            double gtir_tmp_334_0;
            double gtir_tmp_232_0;
            bool __map_fusion_gtir_tmp_133;
            double gtir_tmp_131_0;
            bool __map_fusion_gtir_tmp_236;
            double gtir_tmp_229_0;
            double gtir_tmp_331_0;
            bool __map_fusion_gtir_tmp_361;
            double gtir_tmp_189_0;
            double gtir_tmp_357_0;
            double gtir_tmp_354_0;
            bool __map_fusion_gtir_tmp_338;

            {
                bool __tlet_arg0 = __map_fusion_gtir_tmp_210;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_127_not_)
                __tlet_result = (! __tlet_arg0);
                ///////////////////

                __map_fusion_gtir_tmp_212 = __tlet_result;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_138_get_value__clone_59f64136_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-06;
                ///////////////////

                gtir_tmp_229_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_140_get_value__clone_59db400c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 236.15;
                ///////////////////

                gtir_tmp_232_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_1[((__q_in_1_Cell_stride_0 * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride_0 * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1 = gtir_tmp_232_0;
                double __tlet_arg1_0 = gtir_tmp_229_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_139_greater_fused_tlet_141_greater_fused_tlet_142_and_)
                __tlet_result = ((__tlet_arg0 > __tlet_arg1_0) && (__tlet_arg0_0 > __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_236 = __tlet_result;
            }
            if_stmt_2_3_4_11(__map_fusion_gtir_tmp_236, &q_in_1[0], &q_in_2[0], gtir_tmp_290, __q_in_1_Cell_range_0, __q_in_1_Cell_stride_0, __q_in_1_K_range_0, __q_in_1_K_stride_0, __q_in_2_Cell_range_0, __q_in_2_Cell_stride_0, __q_in_2_K_range_0, __q_in_2_K_stride_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_211_get_value__clone_5a94c90a_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_354_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_213_get_value__clone_5aa3acae_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 236.15;
                ///////////////////

                gtir_tmp_357_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_1[((__q_in_1_Cell_stride_0 * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride_0 * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1 = gtir_tmp_357_0;
                double __tlet_arg1_0 = q_in_5[((__q_in_5_Cell_stride_0 * ((- __q_in_5_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_5_K_stride_0 * ((- __q_in_5_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1_1 = gtir_tmp_354_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_210_minimum_fused_tlet_212_greater_fused_tlet_214_greater_fused_tlet_215_and_)
                __tlet_result = ((min(__tlet_arg0, __tlet_arg1_0) > __tlet_arg1_1) && (__tlet_arg0_0 > __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_361 = __tlet_result;
            }
            if_stmt_6_3_4_25(__map_fusion_gtir_tmp_361, &q_in_1[0], &q_in_5[0], &rho[0], gtir_tmp_373, __q_in_1_Cell_range_0, __q_in_1_Cell_stride_0, __q_in_1_K_range_0, __q_in_1_K_stride_0, __q_in_5_Cell_range_0, __q_in_5_Cell_stride_0, __q_in_5_K_range_0, __q_in_5_K_stride_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_198_get_value__clone_5a8fa330_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_331_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_114_get_value__clone_5980ba7e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_189_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_3[((__q_in_3_Cell_stride_0 * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride_0 * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_189_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_115_greater)
                __tlet_result = (__tlet_arg0 > __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_191 = __tlet_result;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_200_get_value__clone_5a8a4ae8_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 236.15;
                ///////////////////

                gtir_tmp_334_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_1[((__q_in_1_Cell_stride_0 * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride_0 * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1 = gtir_tmp_334_0;
                double __tlet_arg1_0 = q_in_3[((__q_in_3_Cell_stride_0 * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride_0 * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1_1 = gtir_tmp_331_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_197_minimum_fused_tlet_199_greater_fused_tlet_201_greater_fused_tlet_202_and_)
                __tlet_result = ((min(__tlet_arg0, __tlet_arg1_0) > __tlet_arg1_1) && (__tlet_arg0_0 > __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_338 = __tlet_result;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_79_get_value__clone_594d006c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_131_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_3[((__q_in_3_Cell_stride_0 * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride_0 * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_131_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_80_greater)
                __tlet_result = (__tlet_arg0 > __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_133 = __tlet_result;
            }
            if_stmt_0_3_4_41(__map_fusion_gtir_tmp_133, &q_in_3[0], &rho[0], &te[0], gtir_tmp_187, __q_in_3_Cell_range_0, __q_in_3_Cell_stride_0, __q_in_3_K_range_0, __q_in_3_K_stride_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0, __te_Cell_range_0, __te_K_range_0, __te_K_stride_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);
            if_stmt_1_3_4_9(__map_fusion_gtir_tmp_191, gtir_tmp_187, &q_in_3[0], &rho[0], gtir_tmp_206, __q_in_3_Cell_range_0, __q_in_3_Cell_stride_0, __q_in_3_K_range_0, __q_in_3_K_stride_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);
            if_stmt_5_3_4_27(__map_fusion_gtir_tmp_338, gtir_tmp_187, gtir_tmp_206, &q_in_1[0], gtir_tmp_350, __q_in_1_Cell_range_0, __q_in_1_Cell_stride_0, __q_in_1_K_range_0, __q_in_1_K_stride_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);

        }
        if (__map_fusion_gtir_tmp_212) {
            {
                double __arg1________________;
                double __arg1_______________6bfcd4a8_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
                double __arg1______________6bafe472_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6bfcdc82_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
                double gtir_tmp_874_0;
                double __map_fusion_gtir_tmp_868;
                double gtir_tmp_871_0;

                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_499_get_value__clone_5deeab2a_1c88_11f1_a73a_1f4e8615a1c5)
                    __tlet_out = 0.0;
                    ///////////////////

                    gtir_tmp_874_0 = __tlet_out;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_874_0, &__arg1______________6bafe472_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6bfcdc82_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__arg1______________6bafe472_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6bfcdc82_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &gtir_tmp_875, 1);
                {
                    double __tlet_arg0 = gtir_tmp_290;
                    double __tlet_arg1 = gtir_tmp_373;
                    double __tlet_arg1_0 = gtir_tmp_350;
                    double __tlet_result;

                    ///////////////////
                    // Tasklet code (tlet_496_plus_fused_tlet_497_plus)
                    __tlet_result = ((__tlet_arg0 + __tlet_arg1_0) + __tlet_arg1);
                    ///////////////////

                    __map_fusion_gtir_tmp_868 = __tlet_result;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__map_fusion_gtir_tmp_868, &__arg1________________, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__arg1________________, &gtir_tmp_869, 1);
                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_498_get_value__clone_5df4ab88_1c88_11f1_a73a_1f4e8615a1c5)
                    __tlet_out = 0.0;
                    ///////////////////

                    gtir_tmp_871_0 = __tlet_out;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_871_0, &__arg1_______________6bfcd4a8_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__arg1_______________6bfcd4a8_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &gtir_tmp_872, 1);

            }
        } else {
            {


                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_290, &gtir_tmp_869, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_350, &gtir_tmp_872, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_373, &gtir_tmp_875, 1);

            }
        }
        {
            double __arg1__________________;
            double __arg1___________________676cba2a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1___________________67ce9ca4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1___________________68535fa2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1_______________________68a3aed0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1___________________6943d6ee_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double __arg1___________________68f55366_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6943dff4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion;
            double gtir_tmp_1520_0;
            double __map_fusion_gtir_tmp_1566;
            double gtir_tmp_69_0;
            bool __map_fusion_gtir_tmp_1197;
            double gtir_tmp_1665_0;
            double gtir_tmp_1601;
            double gtir_tmp_1367;
            double __map_fusion_gtir_tmp_323;
            bool __map_fusion_gtir_tmp_1344;
            double gtir_tmp_851;
            double gtir_tmp_837_0;
            double gtir_tmp_1589;
            double gtir_tmp_863;
            double __map_fusion_gtir_tmp_1526;
            double gtir_tmp_410_0;
            double gtir_tmp_1412;
            double __map_fusion_gtir_tmp_1518;
            double gtir_tmp_1605_0;
            double __map_fusion_gtir_tmp_1425;
            double gtir_tmp_1638_0;
            double gtir_tmp_1355;
            double gtir_tmp_1089_0;
            double gtir_tmp_1428_0;
            double gtir_tmp_712;
            double gtir_tmp_292_0;
            double __map_fusion_gtir_tmp_77;
            double gtir_tmp_562_0;
            double gtir_tmp_619_0;
            double __map_fusion_gtir_tmp_1604;
            bool gtir_tmp_825_0;
            double gtir_tmp_1186;
            double gtir_tmp_891_0;
            double gtir_tmp_906;
            double gtir_tmp_54_0;
            double __map_fusion_gtir_tmp_1484;
            double gtir_tmp_53_0;
            double __map_fusion_gtir_tmp_1612;
            double gtir_tmp_321_0;
            double gtir_tmp_1503;
            bool __map_fusion_gtir_tmp_895;
            double gtir_tmp_979;
            double gtir_tmp_1639_0;
            double gtir_tmp_842_0;
            double gtir_tmp_1208;
            double gtir_tmp_1437;
            bool __map_fusion_gtir_tmp_1432;
            double gtir_tmp_1486_0;
            double gtir_tmp_1084_0;
            double gtir_tmp_1651_0;
            bool __map_fusion_gtir_tmp_299;
            double gtir_tmp_822_0;
            double gtir_tmp_1421;
            double gtir_tmp_835;
            double __map_fusion_gtir_tmp_1190;
            double gtir_tmp_60_0;
            double gtir_tmp_319;
            double gtir_tmp_720;
            double gtir_tmp_1606_0;
            double gtir_tmp_886_0;
            double __map_fusion_gtir_tmp_1574;
            double __map_fusion_gtir_tmp_1616;
            double __map_fusion_gtir_tmp_888;
            double gtir_tmp_706;
            double gtir_tmp_1073;
            double gtir_tmp_1464;
            double gtir_tmp_1443;
            double gtir_tmp_1650_0;
            double __map_fusion_gtir_tmp_839;
            double __map_fusion_gtir_tmp_1554;
            double gtir_tmp_1485_0;
            double gtir_tmp_820;
            double __map_fusion_gtir_tmp_1673;
            double gtir_tmp_1445_0;
            double gtir_tmp_1543;
            double gtir_tmp_1214;
            bool __map_fusion_gtir_tmp_1093;
            double gtir_tmp_1597;
            double gtir_tmp_400_0;
            double gtir_tmp_857;
            bool __map_fusion_gtir_tmp_724;
            double gtir_tmp_1519_0;
            double gtir_tmp_558_0;
            double __map_fusion_gtir_tmp_1618;
            double gtir_tmp_918;
            double gtir_tmp_1202;
            double gtir_tmp_295_0;
            bool __map_fusion_gtir_tmp_1276;
            double gtir_tmp_1193_0;
            double gtir_tmp_1660_0;
            double gtir_tmp_1627_0;
            double gtir_tmp_884;
            double gtir_tmp_324_0;
            double gtir_tmp_1626_0;
            double gtir_tmp_414_0;
            bool __map_fusion_gtir_tmp_846;
            double gtir_tmp_55_0;
            double __map_fusion_gtir_tmp_1546;
            double gtir_tmp_1460;
            double gtir_tmp_407_0;
            double gtir_tmp_1567_0;
            double gtir_tmp_1511;
            double __map_fusion_gtir_tmp_1474;
            double gtir_tmp_722_0;
            double gtir_tmp_1188_0;
            double gtir_tmp_559_0;
            bool __map_fusion_gtir_tmp_1152;
            double gtir_tmp_1082;
            double gtir_tmp_1568_0;
            double gtir_tmp_928_0;
            double gtir_tmp_1563;
            double gtir_tmp_806;
            double gtir_tmp_1177;
            bool __map_fusion_gtir_tmp_1217;
            double gtir_tmp_413_0;
            double gtir_tmp_812;
            double gtir_tmp_1623_0;
            double gtir_tmp_923_0;
            double __map_fusion_gtir_tmp_568;
            double gtir_tmp_1349;
            double gtir_tmp_1098;
            double gtir_tmp_1104;
            double gtir_tmp_1423_0;
            double gtir_tmp_900;
            double __map_fusion_gtir_tmp_420;
            bool __map_fusion_gtir_tmp_621;
            double gtir_tmp_1361;
            double gtir_tmp_1340_0;
            double gtir_tmp_1468_0;
            double __map_fusion_gtir_tmp_1086;
            double gtir_tmp_1273;
            double gtir_tmp_1333;
            double gtir_tmp_399_0;
            double gtir_tmp_1149;
            bool __map_fusion_gtir_tmp_982;
            bool __map_fusion_gtir_tmp_932;
            double gtir_tmp_1324;
            double gtir_tmp_1515;
            double gtir_tmp_1539;
            double gtir_tmp_1481;
            double __map_fusion_gtir_tmp_1337;
            double __map_fusion_gtir_tmp_1492;
            double __map_fusion_gtir_tmp_328;
            double gtir_tmp_1547_0;
            double gtir_tmp_1548_0;
            double gtir_tmp_912;
            double gtir_tmp_1335_0;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_851_get_value__clone_619c286a_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1567_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_892_get_value__clone_61bd45ae_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 717.6;
                ///////////////////

                gtir_tmp_1650_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_841_get_value__clone_619734c2_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1547_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_785_get_value__clone_616b33ae_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_1423_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_876_get_value__clone_6208aefe_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_1623_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_795_get_value__clone_61759da8_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1445_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_829_get_value__clone_6188036c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_1520_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_36_get_value__clone_58b08304_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 35.86;
                ///////////////////

                gtir_tmp_60_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_31_get_value__clone_58bdcb36_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 610.78;
                ///////////////////

                gtir_tmp_53_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_744_get_value__clone_60e2422e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_1340_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_41_get_value__clone_58ac2994_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 461.51;
                ///////////////////

                gtir_tmp_69_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_869_get_value__clone_61a5d28e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_1606_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_248_get_value__clone_5ad89e46_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 188.14999999999998;
                ///////////////////

                gtir_tmp_414_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_842_get_value__clone_6191c9ec_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_1548_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_176_get_value__clone_5a236ada_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_292_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_741_get_value__clone_60dd0c0a_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_1335_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_4[((__q_in_4_Cell_stride_0 * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride_0 * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_1335_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_742_divides)
                __tlet_result = (__tlet_arg0 / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1337 = __tlet_result;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_868_get_value__clone_61aad34c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1605_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_828_get_value__clone_618c91b6_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1519_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_504_get_value__clone_5e00bc70_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_886_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_1[((__q_in_1_Cell_stride_0 * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride_0 * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_886_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_505_divides)
                __tlet_result = (__tlet_arg0 / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_888 = __tlet_result;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_33_get_value__clone_58b98472_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_55_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_243_get_value__clone_5ae4a646_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1.0;
                ///////////////////

                gtir_tmp_407_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_813_get_value__clone_617ec964_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1485_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_5[((__q_in_5_Cell_stride_0 * ((- __q_in_5_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_5_K_stride_0 * ((- __q_in_5_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_1423_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_786_divides)
                __tlet_result = (__tlet_arg0 / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1425 = __tlet_result;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_238_get_value__clone_5adf101e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.09;
                ///////////////////

                gtir_tmp_399_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_247_get_value__clone_5ad13408_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0035;
                ///////////////////

                gtir_tmp_413_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_893_get_value__clone_61b44c10_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 690.35;
                ///////////////////

                gtir_tmp_1651_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_329_get_value__clone_5c151e60_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 461.51;
                ///////////////////

                gtir_tmp_559_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_788_get_value__clone_61712606_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_1428_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_852_get_value__clone_61a0f232_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_1568_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_239_get_value__clone_5aea17f2_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_400_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_612_get_value__clone_5f4b9ece_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_1089_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_609_get_value__clone_5f452256_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_1084_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_2[((__q_in_2_Cell_stride_0 * ((- __q_in_2_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_2_K_stride_0 * ((- __q_in_2_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_1084_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_610_divides)
                __tlet_result = (__tlet_arg0 / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1086 = __tlet_result;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_667_get_value__clone_6007d62a_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_1193_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_194_get_value__clone_5a41749e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_324_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_476_get_value__clone_5dd293a4_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_822_0 = __tlet_out;
            }
            {
                bool __tlet_out;

                ///////////////////
                // Tasklet code (tlet_478_get_value__clone_5dd800fa_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = true;
                ///////////////////

                gtir_tmp_825_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_885_get_value__clone_620dea9a_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 2899657.201;
                ///////////////////

                gtir_tmp_1638_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_507_get_value__clone_5e3d13aa_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_891_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_420_get_value__clone_5d2fd498_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_722_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_5[((__q_in_5_Cell_stride_0 * ((- __q_in_5_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_5_K_stride_0 * ((- __q_in_5_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_722_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_421_greater)
                __tlet_result = (__tlet_arg0 > __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_724 = __tlet_result;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_32_get_value__clone_58b52bac_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 17.269;
                ///////////////////

                gtir_tmp_54_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_0[((__q_in_0_Cell_stride_0 * ((- __q_in_0_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_0_K_stride_0 * ((- __q_in_0_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg0_1 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg0_1_0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg0_2 = gtir_tmp_54_0;
                double __tlet_arg0_3 = gtir_tmp_53_0;
                double __tlet_arg1 = te[(((- __te_Cell_range_0) + (__te_K_stride_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_0 = gtir_tmp_69_0;
                double __tlet_arg1_0_0 = gtir_tmp_55_0;
                double __tlet_arg1_1 = gtir_tmp_60_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_34_minus_fused_tlet_37_minus_fused_tlet_35_multiplies_fused_tlet_38_divides_fused_tlet_42_multiplies_fused_tlet_43_multiplies_fused_tlet_39_exp_fused_tlet_40_multiplies_fused_tlet_44_divides_fused_tlet_45_minus)
                __tlet_result = (__tlet_arg0 - ((__tlet_arg0_3 * dace::math::exp(((__tlet_arg0_2 * (__tlet_arg0_1_0 - __tlet_arg1_0_0)) / (__tlet_arg0_0 - __tlet_arg1_1)))) / ((__tlet_arg0_1 * __tlet_arg1_0) * __tlet_arg1)));
                ///////////////////

                __map_fusion_gtir_tmp_77 = __tlet_result;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_664_get_value__clone_600257b8_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_1188_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_3[((__q_in_3_Cell_stride_0 * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride_0 * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_1188_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_665_divides)
                __tlet_result = (__tlet_arg0 / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1190 = __tlet_result;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_898_get_value__clone_62036fe8_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 2784.7141119999997;
                ///////////////////

                gtir_tmp_1660_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_245_get_value__clone_5acb7496_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.075;
                ///////////////////

                gtir_tmp_410_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_399_0;
                double __tlet_arg0_0 = gtir_tmp_413_0;
                double __tlet_arg0_0_0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg0_1 = te[(((- __te_Cell_range_0) + (__te_K_stride_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1 = gtir_tmp_414_0;
                double __tlet_arg1_0 = gtir_tmp_410_0;
                double __tlet_arg1_0_0 = gtir_tmp_400_0;
                double __tlet_arg1_1 = gtir_tmp_407_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_240_minus_fused_tlet_241_multiplies_fused_tlet_242_exp_fused_tlet_244_minimum_fused_tlet_249_minus_fused_tlet_246_maximum_fused_tlet_250_multiplies_fused_tlet_251_maximum)
                __tlet_result = max(max(min(dace::math::exp((__tlet_arg0 * (__tlet_arg0_0_0 - __tlet_arg1_0_0))), __tlet_arg1_1), __tlet_arg1_0), (__tlet_arg0_0 * (__tlet_arg0_1 - __tlet_arg1)));
                ///////////////////

                __map_fusion_gtir_tmp_420 = __tlet_result;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_878_get_value__clone_61fe6f66_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 3135383.2031928;
                ///////////////////

                gtir_tmp_1626_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_520_get_value__clone_5f37f464_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_923_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_523_get_value__clone_5f04c72e_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_928_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_2[((__q_in_2_Cell_stride_0 * ((- __q_in_2_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_2_K_stride_0 * ((- __q_in_2_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0 = __map_fusion_gtir_tmp_77;
                double __tlet_arg1 = gtir_tmp_928_0;
                double __tlet_arg1_0 = q_in_1[((__q_in_1_Cell_stride_0 * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride_0 * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1_1 = gtir_tmp_923_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_521_greater_fused_tlet_522_plus_fused_tlet_524_less_equal_fused_tlet_525_and_)
                __tlet_result = ((__tlet_arg0 > __tlet_arg1_1) && ((__tlet_arg0_0 + __tlet_arg1_0) <= __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_932 = __tlet_result;
            }
            if_stmt_40_3_5_103(__map_fusion_gtir_tmp_932, __map_fusion_gtir_tmp_77, &q_in_2[0], &rho[0], &te[0], gtir_tmp_979, __q_in_2_Cell_range_0, __q_in_2_Cell_stride_0, __q_in_2_K_range_0, __q_in_2_K_stride_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0, __te_Cell_range_0, __te_K_range_0, __te_K_stride_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_328_get_value__clone_5c0f7bc2_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 610.78;
                ///////////////////

                gtir_tmp_558_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_363_get_value__clone_5ccf0406_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_619_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_3[((__q_in_3_Cell_stride_0 * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride_0 * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_619_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_364_greater)
                __tlet_result = (__tlet_arg0 > __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_621 = __tlet_result;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_487_get_value__clone_5de7fa50_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 1e-15;
                ///////////////////

                gtir_tmp_842_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_814_get_value__clone_61836fbe_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_1486_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_178_get_value__clone_5a19ad4c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_295_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_4[((__q_in_4_Cell_stride_0 * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride_0 * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1 = gtir_tmp_295_0;
                double __tlet_arg1_0 = gtir_tmp_292_0;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_179_greater_fused_tlet_177_greater_fused_tlet_180_and_)
                __tlet_result = ((__tlet_arg0 > __tlet_arg1_0) && (__tlet_arg0_0 > __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_299 = __tlet_result;
            }
            if_stmt_4_3_5_71(__map_fusion_gtir_tmp_299, &q_in_1[0], &q_in_4[0], &te[0], gtir_tmp_319, __q_in_1_Cell_range_0, __q_in_1_Cell_stride_0, __q_in_1_K_range_0, __q_in_1_K_stride_0, __q_in_4_Cell_range_0, __q_in_4_Cell_stride_0, __q_in_4_K_range_0, __q_in_4_K_stride_0, __te_Cell_range_0, __te_K_range_0, __te_K_stride_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);
            {
                double __tlet_arg0 = gtir_tmp_319;
                double __tlet_arg1 = gtir_tmp_324_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_195_minimum_fused_tlet_196_neg)
                __tlet_result = (- min(__tlet_arg0, __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_328 = __tlet_result;
            }
            {
                bool __tlet_arg0 = __map_fusion_gtir_tmp_210;
                bool __tlet_arg1 = __map_fusion_gtir_tmp_52;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_676_and_)
                __tlet_result = (__tlet_arg0 && __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1217 = __tlet_result;
            }
            {
                bool __tlet_arg0 = __map_fusion_gtir_tmp_210;
                bool __tlet_arg1 = __map_fusion_gtir_tmp_52;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_710_and_)
                __tlet_result = (__tlet_arg0 && __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1276 = __tlet_result;
            }
            if_stmt_61_3_5_210(__map_fusion_gtir_tmp_1276, __map_fusion_gtir_tmp_420, &q_in_2[0], &q_in_4[0], &q_in_5[0], &rho[0], gtir_tmp_1324, __q_in_2_Cell_range_0, __q_in_2_Cell_stride_0, __q_in_2_K_range_0, __q_in_2_K_stride_0, __q_in_4_Cell_range_0, __q_in_4_Cell_stride_0, __q_in_4_K_range_0, __q_in_4_K_stride_0, __q_in_5_Cell_range_0, __q_in_5_Cell_stride_0, __q_in_5_K_range_0, __q_in_5_K_stride_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);
            {
                bool __tlet_arg0 = __map_fusion_gtir_tmp_210;
                bool __tlet_arg1 = __map_fusion_gtir_tmp_52;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_554_and_)
                __tlet_result = (__tlet_arg0 && __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_982 = __tlet_result;
            }
            {
                bool __tlet_arg0 = __map_fusion_gtir_tmp_210;
                bool __tlet_arg1 = __map_fusion_gtir_tmp_52;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_645_and_)
                __tlet_result = (__tlet_arg0 && __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1152 = __tlet_result;
            }
            if_stmt_51_3_5_126(__map_fusion_gtir_tmp_1152, &q_in_1[0], &q_in_3[0], &rho[0], &te[0], gtir_tmp_1177, __q_in_1_Cell_range_0, __q_in_1_Cell_stride_0, __q_in_1_K_range_0, __q_in_1_K_stride_0, __q_in_3_Cell_range_0, __q_in_3_Cell_stride_0, __q_in_3_K_range_0, __q_in_3_K_stride_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0, __te_Cell_range_0, __te_K_range_0, __te_K_stride_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_805_get_value__clone_617a2788_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_1468_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_331_get_value__clone_5c1a3f30_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 273.15;
                ///////////////////

                gtir_tmp_562_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_0[((__q_in_0_Cell_stride_0 * ((- __q_in_0_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_0_K_stride_0 * ((- __q_in_0_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_0 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride_0 * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg0_1 = gtir_tmp_558_0;
                double __tlet_arg1 = gtir_tmp_562_0;
                double __tlet_arg1_0 = gtir_tmp_559_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_330_multiplies_fused_tlet_332_multiplies_fused_tlet_333_divides_fused_tlet_334_minus)
                __tlet_result = (__tlet_arg0 - (__tlet_arg0_1 / ((__tlet_arg0_0 * __tlet_arg1_0) * __tlet_arg1)));
                ///////////////////

                __map_fusion_gtir_tmp_568 = __tlet_result;
            }
            if_stmt_20_3_5_72(__map_fusion_gtir_tmp_621, __map_fusion_gtir_tmp_102, __map_fusion_gtir_tmp_568, __map_fusion_gtir_tmp_77, gtir_tmp_187, gtir_tmp_206, gtir_tmp_448, gtir_tmp_556, &p[0], &q_in_3[0], &rho[0], &te[0], gtir_tmp_706, __p_Cell_range_0, __p_K_range_0, __p_K_stride_0, __q_in_3_Cell_range_0, __q_in_3_Cell_stride_0, __q_in_3_K_range_0, __q_in_3_K_stride_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0, __te_Cell_range_0, __te_K_range_0, __te_K_stride_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);
            if_stmt_25_3_5_15(__map_fusion_gtir_tmp_724, __map_fusion_gtir_tmp_102, __map_fusion_gtir_tmp_568, __map_fusion_gtir_tmp_77, &p[0], &q_in_5[0], &rho[0], &te[0], gtir_tmp_806, __p_Cell_range_0, __p_K_range_0, __p_K_stride_0, __q_in_5_Cell_range_0, __q_in_5_Cell_stride_0, __q_in_5_K_range_0, __q_in_5_K_stride_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0, __te_Cell_range_0, __te_K_range_0, __te_K_stride_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_484_get_value__clone_5de275c6_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 30.0;
                ///////////////////

                gtir_tmp_837_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = q_in_0[((__q_in_0_Cell_stride_0 * ((- __q_in_0_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_0_K_stride_0 * ((- __q_in_0_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_837_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_485_divides)
                __tlet_result = (__tlet_arg0 / __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_839 = __tlet_result;
            }
            if_stmt_44_3_5_168(__map_fusion_gtir_tmp_982, __map_fusion_gtir_tmp_398, __map_fusion_gtir_tmp_77, &q_in_1[0], &q_in_2[0], &q_in_3[0], &q_in_4[0], &rho[0], &te[0], gtir_tmp_1073, __q_in_1_Cell_range_0, __q_in_1_Cell_stride_0, __q_in_1_K_range_0, __q_in_1_K_stride_0, __q_in_2_Cell_range_0, __q_in_2_Cell_stride_0, __q_in_2_K_range_0, __q_in_2_K_stride_0, __q_in_3_Cell_range_0, __q_in_3_Cell_stride_0, __q_in_3_K_range_0, __q_in_3_K_stride_0, __q_in_4_Cell_range_0, __q_in_4_Cell_stride_0, __q_in_4_K_range_0, __q_in_4_K_stride_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0, __te_Cell_range_0, __te_K_range_0, __te_K_stride_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);
            if_stmt_58_3_5_66(__map_fusion_gtir_tmp_1217, __map_fusion_gtir_tmp_398, __map_fusion_gtir_tmp_420, gtir_tmp_187, gtir_tmp_206, gtir_tmp_556, &q_in_4[0], gtir_tmp_1273, __q_in_4_Cell_range_0, __q_in_4_Cell_stride_0, __q_in_4_K_range_0, __q_in_4_K_stride_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);
            if_stmt_21_3_5_184(__map_fusion_gtir_tmp_52, __map_fusion_gtir_tmp_328, __map_fusion_gtir_tmp_568, gtir_tmp_1273, gtir_tmp_1324, gtir_tmp_500, gtir_tmp_706, gtir_tmp_806, &p[0], &q_in_3[0], &q_in_5[0], &rho[0], &te[0], gtir_tmp_712, gtir_tmp_1149, gtir_tmp_1333, gtir_tmp_720, gtir_tmp_812, gtir_tmp_1412, gtir_tmp_820, __p_Cell_range_0, __p_K_range_0, __p_K_stride_0, __q_in_3_Cell_range_0, __q_in_3_Cell_stride_0, __q_in_3_K_range_0, __q_in_3_K_stride_0, __q_in_5_Cell_range_0, __q_in_5_Cell_stride_0, __q_in_5_K_range_0, __q_in_5_K_stride_0, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride_0, __te_Cell_range_0, __te_K_range_0, __te_K_stride_0, i_Cell_gtx_horizontal, i_K_gtx_vertical);
            {
                double __tlet_arg0 = gtir_tmp_1333;
                double __tlet_arg0_0 = q_in_4[((__q_in_4_Cell_stride_0 * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride_0 * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_1340_0;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_1337;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_745_greater_fused_tlet_743_greater_fused_tlet_746_and_)
                __tlet_result = ((__tlet_arg0 > __tlet_arg1_0) && (__tlet_arg0_0 > __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_1344 = __tlet_result;
            }
            if_stmt_63_3_5_167(gtir_tmp_500, gtir_tmp_1324, __map_fusion_gtir_tmp_328, gtir_tmp_1273, __map_fusion_gtir_tmp_1344, __map_fusion_gtir_tmp_1337, __map_fusion_gtir_tmp_328, gtir_tmp_1273, gtir_tmp_1324, gtir_tmp_1333, gtir_tmp_500, gtir_tmp_1349, gtir_tmp_1367, gtir_tmp_1355, gtir_tmp_1361);
            if_stmt_80_3_5_104(gtir_tmp_1333, __map_fusion_gtir_tmp_1344, gtir_tmp_1349, gtir_tmp_1355, gtir_tmp_1361, gtir_tmp_1367, gtir_tmp_1563);
            if_stmt_52_3_5_191(__map_fusion_gtir_tmp_52, gtir_tmp_1149, gtir_tmp_1177, gtir_tmp_1412, gtir_tmp_720, gtir_tmp_820, gtir_tmp_822_0, gtir_tmp_1186, gtir_tmp_1421);
            {
                double __tlet_arg0 = gtir_tmp_1186;
                double __tlet_arg0_0 = q_in_3[((__q_in_3_Cell_stride_0 * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride_0 * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_1193_0;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_1190;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_666_greater_fused_tlet_668_greater_fused_tlet_669_and_)
                __tlet_result = ((__tlet_arg0 > __tlet_arg1_0) && (__tlet_arg0_0 > __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_1197 = __tlet_result;
            }
            if_stmt_53_3_5_149(gtir_tmp_720, gtir_tmp_1149, gtir_tmp_822_0, gtir_tmp_1177, __map_fusion_gtir_tmp_1197, __map_fusion_gtir_tmp_1190, gtir_tmp_1149, gtir_tmp_1177, gtir_tmp_1186, gtir_tmp_720, gtir_tmp_822_0, gtir_tmp_1202, gtir_tmp_1208, gtir_tmp_1539, gtir_tmp_1214);
            if_stmt_79_3_5_262(gtir_tmp_1186, __map_fusion_gtir_tmp_1197, gtir_tmp_1202, gtir_tmp_1208, gtir_tmp_1214, gtir_tmp_1539, gtir_tmp_1543);
            {
                double __tlet_arg0 = gtir_tmp_1421;
                double __tlet_arg0_0 = q_in_5[((__q_in_5_Cell_stride_0 * ((- __q_in_5_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_5_K_stride_0 * ((- __q_in_5_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_1428_0;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_1425;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_789_greater_fused_tlet_787_greater_fused_tlet_790_and_)
                __tlet_result = ((__tlet_arg0 > __tlet_arg1_0) && (__tlet_arg0_0 > __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_1432 = __tlet_result;
            }
            if_stmt_70_3_5_96(gtir_tmp_820, gtir_tmp_822_0, gtir_tmp_1412, gtir_tmp_822_0, __map_fusion_gtir_tmp_1432, __map_fusion_gtir_tmp_1425, gtir_tmp_1412, gtir_tmp_1421, gtir_tmp_820, gtir_tmp_822_0, gtir_tmp_1437, gtir_tmp_1443, gtir_tmp_1597, gtir_tmp_1589);
            if_stmt_83_3_5_54(gtir_tmp_1421, __map_fusion_gtir_tmp_1432, gtir_tmp_1437, gtir_tmp_1443, gtir_tmp_1589, gtir_tmp_1597, gtir_tmp_1601);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_192_get_value__clone_5a3c755c_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_321_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = gtir_tmp_319;
                double __tlet_arg1 = gtir_tmp_321_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_193_maximum)
                __tlet_result = max(__tlet_arg0, __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_323 = __tlet_result;
            }
            if_stmt_28_3_5_163(gtir_tmp_825_0, __map_fusion_gtir_tmp_323, gtir_tmp_1073, gtir_tmp_548, gtir_tmp_712, gtir_tmp_812, gtir_tmp_822_0, gtir_tmp_869, gtir_tmp_872, gtir_tmp_875, gtir_tmp_979, gtir_tmp_835, gtir_tmp_1082, gtir_tmp_884);
            {
                double __tlet_arg0 = gtir_tmp_835;
                double __tlet_arg0_0 = q_in_0[((__q_in_0_Cell_stride_0 * ((- __q_in_0_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_0_K_stride_0 * ((- __q_in_0_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_842_0;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_839;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_488_greater_fused_tlet_486_greater_fused_tlet_489_and_)
                __tlet_result = ((__tlet_arg0 > __tlet_arg1_0) && (__tlet_arg0_0 > __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_846 = __tlet_result;
            }
            if_stmt_29_3_5_180(gtir_tmp_712, gtir_tmp_822_0, gtir_tmp_812, gtir_tmp_548, __map_fusion_gtir_tmp_846, __map_fusion_gtir_tmp_839, gtir_tmp_548, gtir_tmp_712, gtir_tmp_812, gtir_tmp_822_0, gtir_tmp_835, gtir_tmp_851, gtir_tmp_1460, gtir_tmp_863, gtir_tmp_857);
            if_stmt_73_3_5_297(gtir_tmp_835, __map_fusion_gtir_tmp_846, gtir_tmp_1460, gtir_tmp_851, gtir_tmp_857, gtir_tmp_863, gtir_tmp_1464);
            {
                double __tlet_arg0 = gtir_tmp_1082;
                double __tlet_arg0_0 = q_in_2[((__q_in_2_Cell_stride_0 * ((- __q_in_2_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_2_K_stride_0 * ((- __q_in_2_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_1089_0;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_1086;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_613_greater_fused_tlet_611_greater_fused_tlet_614_and_)
                __tlet_result = ((__tlet_arg0 > __tlet_arg1_0) && (__tlet_arg0_0 > __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_1093 = __tlet_result;
            }
            if_stmt_46_3_5_211(gtir_tmp_979, gtir_tmp_822_0, gtir_tmp_822_0, gtir_tmp_1073, __map_fusion_gtir_tmp_1093, __map_fusion_gtir_tmp_1086, gtir_tmp_1073, gtir_tmp_1082, gtir_tmp_822_0, gtir_tmp_979, gtir_tmp_1098, gtir_tmp_1503, gtir_tmp_1511, gtir_tmp_1104);
            {
                double __tlet_arg0 = gtir_tmp_1445_0;
                double __tlet_arg0_0 = q_in_0[((__q_in_0_Cell_stride_0 * ((- __q_in_0_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_0_K_stride_0 * ((- __q_in_0_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_1 = gtir_tmp_1098;
                double __tlet_arg1 = gtir_tmp_1468_0;
                double __tlet_arg1_0 = gtir_tmp_1437;
                double __tlet_arg1_0_0 = gtir_tmp_1202;
                double __tlet_arg1_1 = gtir_tmp_1464;
                double __tlet_arg1_2 = gtir_tmp_1349;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_796_plus_fused_tlet_797_plus_fused_tlet_798_plus_fused_tlet_804_minus_fused_tlet_806_multiplies_fused_tlet_807_plus_fused_tlet_808_maximum)
                __tlet_result = max(__tlet_arg0, (__tlet_arg0_0 + (((((__tlet_arg0_1 + __tlet_arg1_0_0) + __tlet_arg1_2) + __tlet_arg1_0) - __tlet_arg1_1) * __tlet_arg1)));
                ///////////////////

                __map_fusion_gtir_tmp_1474 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1474, &__arg1___________________6943d6ee_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1___________________6943d6ee_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6943db80_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            if_stmt_77_3_5_108(gtir_tmp_1082, __map_fusion_gtir_tmp_1093, gtir_tmp_1098, gtir_tmp_1104, gtir_tmp_1503, gtir_tmp_1511, gtir_tmp_1515);
            {
                double __tlet_arg0 = gtir_tmp_884;
                double __tlet_arg0_0 = q_in_1[((__q_in_1_Cell_stride_0 * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride_0 * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_891_0;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_888;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_506_greater_fused_tlet_508_greater_fused_tlet_509_and_)
                __tlet_result = ((__tlet_arg0 > __tlet_arg1_0) && (__tlet_arg0_0 > __tlet_arg1));
                ///////////////////

                __map_fusion_gtir_tmp_895 = __tlet_result;
            }
            if_stmt_36_3_5_259(gtir_tmp_869, __map_fusion_gtir_tmp_323, gtir_tmp_875, gtir_tmp_872, __map_fusion_gtir_tmp_895, __map_fusion_gtir_tmp_323, __map_fusion_gtir_tmp_888, gtir_tmp_869, gtir_tmp_872, gtir_tmp_875, gtir_tmp_884, gtir_tmp_900, gtir_tmp_912, gtir_tmp_918, gtir_tmp_906);
            {
                double __tlet_arg0 = gtir_tmp_900;
                double __tlet_arg1 = gtir_tmp_1515;
                double __tlet_arg1_0 = gtir_tmp_1443;
                double __tlet_arg1_1 = gtir_tmp_1208;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_818_plus_fused_tlet_819_plus_fused_tlet_827_minus)
                __tlet_result = (((__tlet_arg0 + __tlet_arg1_1) + __tlet_arg1_0) - __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1518 = __tlet_result;
            }
            {
                double __tlet_arg0 = gtir_tmp_1519_0;
                double __tlet_arg0_0 = __map_fusion_gtir_tmp_1518;
                double __tlet_arg0_1 = q_in_2[((__q_in_2_Cell_stride_0 * ((- __q_in_2_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_2_K_stride_0 * ((- __q_in_2_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_1520_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_830_multiplies_fused_tlet_831_plus_fused_tlet_832_maximum)
                __tlet_result = max(__tlet_arg0, (__tlet_arg0_1 + (__tlet_arg0_0 * __tlet_arg1)));
                ///////////////////

                __map_fusion_gtir_tmp_1526 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1526, &__arg1___________________68f55366_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6943dff4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1___________________68f55366_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6943dff4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_68f5582a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6943e1f2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_arg0 = gtir_tmp_857;
                double __tlet_arg1 = gtir_tmp_1563;
                double __tlet_arg1_0 = gtir_tmp_912;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_846_plus_fused_tlet_850_minus)
                __tlet_result = ((__tlet_arg0 + __tlet_arg1_0) - __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1566 = __tlet_result;
            }
            {
                double __tlet_arg0 = gtir_tmp_1567_0;
                double __tlet_arg0_0 = q_in_4[((__q_in_4_Cell_stride_0 * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride_0 * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_1 = __map_fusion_gtir_tmp_1566;
                double __tlet_arg1 = gtir_tmp_1568_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_853_multiplies_fused_tlet_854_plus_fused_tlet_855_maximum)
                __tlet_result = max(__tlet_arg0, (__tlet_arg0_0 + (__tlet_arg0_1 * __tlet_arg1)));
                ///////////////////

                __map_fusion_gtir_tmp_1574 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1574, &__arg1__________________, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1__________________, &__output, 1);
            {
                double __tlet_arg0 = gtir_tmp_863;
                double __tlet_arg1 = gtir_tmp_1601;
                double __tlet_arg1_0 = gtir_tmp_1367;
                double __tlet_arg1_0_0 = gtir_tmp_918;
                double __tlet_arg1_1 = gtir_tmp_1214;
                double __tlet_arg1_2 = gtir_tmp_1104;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_856_plus_fused_tlet_857_plus_fused_tlet_858_plus_fused_tlet_859_plus_fused_tlet_867_minus)
                __tlet_result = (((((__tlet_arg0 + __tlet_arg1_0_0) + __tlet_arg1_2) + __tlet_arg1_1) + __tlet_arg1_0) - __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1604 = __tlet_result;
            }
            {
                double __tlet_arg0 = gtir_tmp_1605_0;
                double __tlet_arg0_0 = q_in_5[((__q_in_5_Cell_stride_0 * ((- __q_in_5_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_5_K_stride_0 * ((- __q_in_5_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_1 = __map_fusion_gtir_tmp_1604;
                double __tlet_arg1 = gtir_tmp_1606_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_870_multiplies_fused_tlet_871_plus_fused_tlet_872_maximum)
                __tlet_result = max(__tlet_arg0, (__tlet_arg0_0 + (__tlet_arg0_1 * __tlet_arg1)));
                ///////////////////

                __map_fusion_gtir_tmp_1612 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1612, &__arg1___________________67ce9ca4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1___________________67ce9ca4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_67cea0fa_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_arg0 = gtir_tmp_851;
                double __tlet_arg1 = gtir_tmp_1543;
                double __tlet_arg1_0 = gtir_tmp_906;
                double __tlet_arg1_1 = gtir_tmp_1361;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_833_plus_fused_tlet_834_plus_fused_tlet_840_minus)
                __tlet_result = (((__tlet_arg0 + __tlet_arg1_0) + __tlet_arg1_1) - __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1546 = __tlet_result;
            }
            {
                double __tlet_arg0 = gtir_tmp_1547_0;
                double __tlet_arg0_0 = q_in_3[((__q_in_3_Cell_stride_0 * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride_0 * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg0_1 = __map_fusion_gtir_tmp_1546;
                double __tlet_arg1 = gtir_tmp_1548_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_843_multiplies_fused_tlet_844_plus_fused_tlet_845_maximum)
                __tlet_result = max(__tlet_arg0, (__tlet_arg0_0 + (__tlet_arg0_1 * __tlet_arg1)));
                ///////////////////

                __map_fusion_gtir_tmp_1554 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1554, &__arg1___________________676cba2a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1___________________676cba2a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_676cbfd4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_arg0 = __map_fusion_gtir_tmp_1554;
                double __tlet_arg1 = __map_fusion_gtir_tmp_1612;
                double __tlet_arg1_0 = __map_fusion_gtir_tmp_1574;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_873_plus_fused_tlet_874_plus)
                __tlet_result = ((__tlet_arg0 + __tlet_arg1_0) + __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1616 = __tlet_result;
            }
            if_stmt_74_3_5_115(gtir_tmp_884, __map_fusion_gtir_tmp_895, gtir_tmp_900, gtir_tmp_906, gtir_tmp_912, gtir_tmp_918, gtir_tmp_1481);
            {
                double __tlet_arg0 = gtir_tmp_1355;
                double __tlet_arg1 = gtir_tmp_1481;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_812_minus)
                __tlet_result = (__tlet_arg0 - __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1484 = __tlet_result;
            }
            {
                double __tlet_arg0 = gtir_tmp_1485_0;
                double __tlet_arg0_0 = __map_fusion_gtir_tmp_1484;
                double __tlet_arg0_1 = q_in_1[((__q_in_1_Cell_stride_0 * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride_0 * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                double __tlet_arg1 = gtir_tmp_1486_0;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_815_multiplies_fused_tlet_816_plus_fused_tlet_817_maximum)
                __tlet_result = max(__tlet_arg0, (__tlet_arg0_1 + (__tlet_arg0_0 * __tlet_arg1)));
                ///////////////////

                __map_fusion_gtir_tmp_1492 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1492, &__arg1___________________68535fa2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1___________________68535fa2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_685364e8_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);
            {
                double __tlet_arg0 = __map_fusion_gtir_tmp_1492;
                double __tlet_arg1 = __map_fusion_gtir_tmp_1526;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_875_plus)
                __tlet_result = (__tlet_arg0 + __tlet_arg1);
                ///////////////////

                __map_fusion_gtir_tmp_1618 = __tlet_result;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_886_get_value__clone_61af8540_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 700.05;
                ///////////////////

                gtir_tmp_1639_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_879_get_value__clone_61f968f4_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 2784.7141119999997;
                ///////////////////

                gtir_tmp_1627_0 = __tlet_out;
            }
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_901_get_value__clone_61b8be12_1c88_11f1_a73a_1f4e8615a1c5)
                __tlet_out = 700.05;
                ///////////////////

                gtir_tmp_1665_0 = __tlet_out;
            }
            {
                double __tlet_arg0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg0_0 = gtir_tmp_1660_0;
                double __tlet_arg0_0_0 = __map_fusion_gtir_tmp_1474;
                double __tlet_arg0_0_1 = __map_fusion_gtir_tmp_1566;
                double __tlet_arg0_1 = gtir_tmp_1665_0;
                double __tlet_arg0_1_0 = gtir_tmp_1626_0;
                double __tlet_arg0_2 = gtir_tmp_1650_0;
                double __tlet_arg0_2_0 = gtir_tmp_1638_0;
                double __tlet_arg0_3 = gtir_tmp_1651_0;
                double __tlet_arg0_3_0 = gtir_tmp_1627_0;
                double __tlet_arg0_4 = gtir_tmp_1639_0;
                double __tlet_arg0_5 = __map_fusion_gtir_tmp_1484;
                double __tlet_arg0_6 = gtir_tmp_1623_0;
                double __tlet_arg1 = __map_fusion_gtir_tmp_1616;
                double __tlet_arg1_0 = te[(((- __te_Cell_range_0) + (__te_K_stride_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_arg1_0_0 = __map_fusion_gtir_tmp_1616;
                double __tlet_arg1_0_0_0 = __map_fusion_gtir_tmp_1546;
                double __tlet_arg1_1 = __map_fusion_gtir_tmp_1618;
                double __tlet_arg1_1_0 = __map_fusion_gtir_tmp_1518;
                double __tlet_arg1_2 = __map_fusion_gtir_tmp_1618;
                double __tlet_arg1_2_0 = __map_fusion_gtir_tmp_1604;
                double __tlet_arg1_3 = te[(((- __te_Cell_range_0) + (__te_K_stride_0 * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_883_plus_fused_tlet_884_plus_fused_tlet_887_multiplies_fused_tlet_880_multiplies_fused_tlet_888_minus_fused_tlet_877_plus_fused_tlet_881_minus_fused_tlet_882_multiplies_fused_tlet_889_multiplies_fused_tlet_890_plus_fused_tlet_894_plus_fused_tlet_895_plus_fused_tlet_896_multiplies_fused_tlet_897_plus_fused_tlet_899_multiplies_fused_tlet_900_plus_fused_tlet_902_multiplies_fused_tlet_891_multiplies_fused_tlet_903_plus_fused_tlet_904_divides_fused_tlet_905_plus)
                __tlet_result = (__tlet_arg0 + ((__tlet_arg0_6 * (((__tlet_arg0_5 + __tlet_arg1_1_0) * (__tlet_arg0_1_0 - (__tlet_arg0_3_0 * __tlet_arg1_0))) + (((__tlet_arg0_0_1 + __tlet_arg1_0_0_0) + __tlet_arg1_2_0) * (__tlet_arg0_2_0 - (__tlet_arg0_4 * __tlet_arg1_3))))) / (((__tlet_arg0_2 + (__tlet_arg0_3 * ((__tlet_arg0_0_0 + __tlet_arg1_0_0) + __tlet_arg1_2))) + (__tlet_arg0_0 * __tlet_arg1_1)) + (__tlet_arg0_1 * __tlet_arg1))));
                ///////////////////

                __map_fusion_gtir_tmp_1673 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__map_fusion_gtir_tmp_1673, &__arg1_______________________68a3aed0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg1_______________________68a3aed0_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_68a3b376_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

        }
    } else {
        {


            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2, &__output, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_676cc204_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_676cbfd4_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_67cea33e_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_67cea0fa_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_6853672c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_685364e8_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2__68a3b5a6_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_68a3b376_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_6943dd9c_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_6943db80_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__arg2_68f55a78_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6943e3d2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, &__output_68f5582a_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion_6943e1f2_1c88_11f1_a73a_1f4e8615a1c5_from_cb_fusion, 1);

        }
    }
}

DACE_DFI void if_stmt_87_215_1_7(const bool&  __cond, const double&  __ct_el_25, const double&  gtir_var_2, const double&  gtir_var_87, const double&  gtir_var_95, const double&  rho_, double&  __output) {

    if (__cond) {
        {
            double gtir_tmp_1731;

            {
                double __tlet_arg0 = gtir_var_95;
                double __tlet_arg0_0 = gtir_var_87;
                double __tlet_arg0_1 = gtir_var_2;
                double __tlet_arg0_2 = __ct_el_25;
                double __tlet_arg1 = gtir_var_2;
                double __tlet_arg1_0 = rho_;
                double __tlet_arg1_1 = gtir_var_87;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_949_multiplies_fused_tlet_950_plus_fused_tlet_951_multiplies_fused_tlet_952_plus_fused_tlet_953_multiplies_fused_tlet_954_minus_fused_tlet_946_plus_fused_tlet_947_multiplies_fused_tlet_948_divides_fused_tlet_955_power_fused_tlet_957_multiplies_fused_tlet_958_multiplies_fused_tlet_956_multiplies_fused_tlet_959_divides_fused_tlet_944_multiplies_fused_tlet_945_maximum_fused_tlet_942_multiplies_fused_tlet_943_minimum_fused_tlet_960_maximum_fused_tlet_961_minimum)
                __tlet_result = min(min((__tlet_arg0 * 762750000.0), 1000000000.0), max(max((__tlet_arg0 * 3813750.0), 1000000.0), ((dace::math::pow((((__tlet_arg0_2 + 2e-06) * __tlet_arg1_0) / 0.069), (4.0 - (((((__tlet_arg0_0 * 9.6e-05) + 0.0119) * __tlet_arg1_1) + 1.42) * 3.0))) * 13.5) / ((__tlet_arg0_1 * __tlet_arg1) * __tlet_arg1))));
                ///////////////////

                gtir_tmp_1731 = __tlet_result;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1731, &__output, 1);

        }
    } else {
        {
            double gtir_tmp_1732;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_962_write)
                __tlet_out = 800000.0;
                ///////////////////

                gtir_tmp_1732 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1732, &__output, 1);

        }
    }
}

DACE_DFI void if_stmt_88_215_1_28(const bool&  __cond, const double&  __ct_el_22, const double&  __ct_el_23, const double&  __ct_el_24, const double&  __ct_el_25, const double&  __ct_el_26, const double&  __ct_el_27, const double&  dt_, const double&  dz_, const bool&  gtir_var_3, const double&  gtir_var_89, const double&  gtir_var_90, const double&  gtir_var_93, const double&  gtir_var_96, const bool&  mask_g, const bool&  mask_i, const bool&  mask_r, const bool&  mask_s, const double&  previous_level_0_0, const double&  previous_level_0_1, const double&  previous_level_0_2, const bool&  previous_level_0_3, const double&  previous_level_1_0, const double&  previous_level_1_1, const double&  previous_level_1_2, const bool&  previous_level_1_3, const double&  previous_level_2_0, const double&  previous_level_2_1, const double&  previous_level_2_2, const bool&  previous_level_2_3, const double&  previous_level_3_0, const double&  previous_level_3_1, const double&  previous_level_3_2, const bool&  previous_level_3_3, const double&  previous_level_4_1, const bool&  previous_level_4_2, const double&  previous_level_5, const double&  rho_, const double&  t_, const double&  t_kp1, double&  __output_0_0, double&  __output_0_1, double&  __output_0_2, bool&  __output_0_3, double&  __output_1_0, double&  __output_1_1, double&  __output_1_2, bool&  __output_1_3, double&  __output_2_0, double&  __output_2_1, double&  __output_2_2, bool&  __output_2_3, double&  __output_3_0, double&  __output_3_1, double&  __output_3_2, bool&  __output_3_3, double&  __output_4_0, double&  __output_4_1, bool&  __output_4_2) {
    bool gtir_tmp_1745;
    double gtir_tmp_1749;
    double gtir_tmp_1755;
    double gtir_tmp_1776;
    double gtir_tmp_1777;
    double gtir_tmp_1778;
    bool gtir_tmp_1779;
    bool gtir_tmp_1780;
    double gtir_tmp_1784;
    double gtir_tmp_1790;
    double gtir_tmp_1811;
    double gtir_tmp_1812;
    double gtir_tmp_1813;
    bool gtir_tmp_1814;
    bool gtir_tmp_1815;
    double gtir_tmp_1819;
    double gtir_tmp_1825;
    double gtir_tmp_1846;
    double gtir_tmp_1847;
    double gtir_tmp_1848;
    bool gtir_tmp_1849;
    bool gtir_tmp_1850;
    double gtir_tmp_1854;
    double gtir_tmp_1860;
    double gtir_tmp_1881;
    double gtir_tmp_1882;
    double gtir_tmp_1883;
    bool gtir_tmp_1884;
    double gtir_tmp_1885;
    double gtir_tmp_1887;
    bool gtir_tmp_1888;
    double gtir_tmp_1945;
    double gtir_tmp_1946;
    double gtir_tmp_1947;
    bool gtir_tmp_1948;
    double gtir_tmp_1949;
    double gtir_tmp_1950;
    double gtir_tmp_1951;
    bool gtir_tmp_1952;
    double gtir_tmp_1953;
    double gtir_tmp_1954;
    double gtir_tmp_1955;
    bool gtir_tmp_1956;
    double gtir_tmp_1957;
    double gtir_tmp_1958;
    double gtir_tmp_1959;
    bool gtir_tmp_1960;
    double gtir_tmp_1961;
    double gtir_tmp_1962;
    bool gtir_tmp_1963;
    double gtir_tmp_1764;
    double gtir_tmp_1799;
    double gtir_tmp_1834;
    double gtir_tmp_1869;

    if (__cond) {
        {
            double gtir_tmp_1746;

            {
                bool __tlet_arg0 = previous_level_0_3;
                bool __tlet_arg1 = mask_r;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_974_or_)
                __tlet_result = (__tlet_arg0 || __tlet_arg1);
                ///////////////////

                gtir_tmp_1745 = __tlet_result;
            }
            {
                double __tlet_arg0 = __ct_el_24;
                double __tlet_arg1 = rho_;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_975_multiplies)
                __tlet_result = (__tlet_arg0 * __tlet_arg1);
                ///////////////////

                gtir_tmp_1746 = __tlet_result;
            }
            {
                double __tlet_arg0 = previous_level_0_1;
                double __tlet_arg0_0 = gtir_tmp_1746;
                double __tlet_arg1 = gtir_var_96;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_977_divides_fused_tlet_976_multiplies_fused_tlet_978_plus)
                __tlet_result = ((__tlet_arg0 * 2.0) + (__tlet_arg0_0 / __tlet_arg1));
                ///////////////////

                gtir_tmp_1749 = __tlet_result;
            }
            {
                double __tlet_arg0 = gtir_tmp_1746;
                double __tlet_arg0_0 = gtir_tmp_1746;
                double __tlet_arg1 = gtir_tmp_1749;
                double __tlet_arg1_0 = gtir_var_93;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_979_multiplies_fused_tlet_980_multiplies_fused_tlet_981_plus_fused_tlet_982_power_fused_tlet_983_multiplies_fused_tlet_984_minimum)
                __tlet_result = min((((__tlet_arg0 * __tlet_arg1_0) * 14.58) * dace::math::pow((__tlet_arg0_0 + 1e-12), 0.111)), __tlet_arg1);
                ///////////////////

                gtir_tmp_1755 = __tlet_result;
            }

        }
        if (gtir_tmp_1745) {
            if (previous_level_0_3) {
                {
                    double gtir_tmp_1762;

                    {
                        double __tlet_arg0 = previous_level_0_2;
                        double __tlet_arg0_0 = previous_level_0_0;
                        double __tlet_arg1 = previous_level_5;
                        double __tlet_arg1_0 = __ct_el_24;
                        double __tlet_result;

                        ///////////////////
                        // Tasklet code (tlet_986_plus_fused_tlet_987_multiplies_fused_tlet_988_multiplies_fused_tlet_989_plus_fused_tlet_990_power_fused_tlet_985_multiplies_fused_tlet_991_multiplies)
                        __tlet_result = ((__tlet_arg0 * 14.58) * dace::math::pow(((((__tlet_arg0_0 + __tlet_arg1_0) * 0.5) * __tlet_arg1) + 1e-12), 0.111));
                        ///////////////////

                        gtir_tmp_1762 = __tlet_result;
                    }

                    dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                    &gtir_tmp_1762, &gtir_tmp_1764, 1);

                }
            } else {
                {
                    double gtir_tmp_1763;

                    {
                        double __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_992_write)
                        __tlet_out = 0.0;
                        ///////////////////

                        gtir_tmp_1763 = __tlet_out;
                    }

                    dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                    &gtir_tmp_1763, &gtir_tmp_1764, 1);

                }
            }
            {
                double gtir_tmp_1774;
                double gtir_tmp_1770;


                dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1745, &gtir_tmp_1779, 1);
                {
                    double __tlet_arg0 = gtir_tmp_1749;
                    double __tlet_arg0_0 = gtir_var_96;
                    double __tlet_arg1 = rho_;
                    double __tlet_arg1_0 = gtir_tmp_1755;
                    double __tlet_arg1_1 = gtir_var_96;
                    double __tlet_arg1_2 = gtir_tmp_1764;
                    double __tlet_result;

                    ///////////////////
                    // Tasklet code (tlet_995_multiplies_fused_tlet_996_plus_fused_tlet_993_minus_fused_tlet_994_multiplies_fused_tlet_997_multiplies_fused_tlet_998_divides)
                    __tlet_result = (((__tlet_arg0 - __tlet_arg1_0) * __tlet_arg1_1) / (((__tlet_arg0_0 * __tlet_arg1_2) + 1.0) * __tlet_arg1));
                    ///////////////////

                    gtir_tmp_1770 = __tlet_result;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1770, &gtir_tmp_1776, 1);
                {
                    double __tlet_arg0 = gtir_tmp_1770;
                    double __tlet_arg1 = gtir_tmp_1755;
                    double __tlet_arg1_0 = rho_;
                    double __tlet_arg1_1 = gtir_tmp_1764;
                    double __tlet_result;

                    ///////////////////
                    // Tasklet code (tlet_999_multiplies_fused_tlet_1000_multiplies_fused_tlet_1001_plus_fused_tlet_1002_multiplies)
                    __tlet_result = ((((__tlet_arg0 * __tlet_arg1_0) * __tlet_arg1_1) + __tlet_arg1) * 0.5);
                    ///////////////////

                    gtir_tmp_1774 = __tlet_result;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1774, &gtir_tmp_1777, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_var_93, &gtir_tmp_1778, 1);

            }
        } else {
            {
                double gtir_tmp_1775;

                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_1003_write)
                    __tlet_out = 0.0;
                    ///////////////////

                    gtir_tmp_1775 = __tlet_out;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1775, &gtir_tmp_1777, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__ct_el_24, &gtir_tmp_1776, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_var_93, &gtir_tmp_1778, 1);

                dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1745, &gtir_tmp_1779, 1);

            }
        }
        {
            double gtir_tmp_1781;

            {
                double __tlet_arg0 = __ct_el_25;
                double __tlet_arg1 = rho_;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_1005_multiplies)
                __tlet_result = (__tlet_arg0 * __tlet_arg1);
                ///////////////////

                gtir_tmp_1781 = __tlet_result;
            }
            {
                double __tlet_arg0 = previous_level_1_1;
                double __tlet_arg0_0 = gtir_tmp_1781;
                double __tlet_arg1 = gtir_var_96;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_1006_multiplies_fused_tlet_1007_divides_fused_tlet_1008_plus)
                __tlet_result = ((__tlet_arg0 * 2.0) + (__tlet_arg0_0 / __tlet_arg1));
                ///////////////////

                gtir_tmp_1784 = __tlet_result;
            }
            {
                double __tlet_arg0 = gtir_tmp_1781;
                double __tlet_arg0_0 = gtir_tmp_1781;
                double __tlet_arg1 = gtir_tmp_1784;
                double __tlet_arg1_0 = gtir_var_90;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_1009_multiplies_fused_tlet_1011_plus_fused_tlet_1012_power_fused_tlet_1010_multiplies_fused_tlet_1013_multiplies_fused_tlet_1014_minimum)
                __tlet_result = min((((__tlet_arg0 * __tlet_arg1_0) * 57.8) * dace::math::pow((__tlet_arg0_0 + 1e-12), 0.16666666666666666)), __tlet_arg1);
                ///////////////////

                gtir_tmp_1790 = __tlet_result;
            }
            {
                bool __tlet_arg0 = previous_level_1_3;
                bool __tlet_arg1 = mask_s;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_1004_or_)
                __tlet_result = (__tlet_arg0 || __tlet_arg1);
                ///////////////////

                gtir_tmp_1780 = __tlet_result;
            }

        }
        if (gtir_tmp_1780) {
            if (previous_level_1_3) {
                {
                    double gtir_tmp_1797;

                    {
                        double __tlet_arg0 = previous_level_1_2;
                        double __tlet_arg0_0 = previous_level_1_0;
                        double __tlet_arg1 = previous_level_5;
                        double __tlet_arg1_0 = __ct_el_25;
                        double __tlet_result;

                        ///////////////////
                        // Tasklet code (tlet_1016_plus_fused_tlet_1017_multiplies_fused_tlet_1018_multiplies_fused_tlet_1019_plus_fused_tlet_1020_power_fused_tlet_1015_multiplies_fused_tlet_1021_multiplies)
                        __tlet_result = ((__tlet_arg0 * 57.8) * dace::math::pow(((((__tlet_arg0_0 + __tlet_arg1_0) * 0.5) * __tlet_arg1) + 1e-12), 0.16666666666666666));
                        ///////////////////

                        gtir_tmp_1797 = __tlet_result;
                    }

                    dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                    &gtir_tmp_1797, &gtir_tmp_1799, 1);

                }
            } else {
                {
                    double gtir_tmp_1798;

                    {
                        double __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_1022_write)
                        __tlet_out = 0.0;
                        ///////////////////

                        gtir_tmp_1798 = __tlet_out;
                    }

                    dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                    &gtir_tmp_1798, &gtir_tmp_1799, 1);

                }
            }
            {
                double gtir_tmp_1809;
                double gtir_tmp_1805;


                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_var_90, &gtir_tmp_1813, 1);

                dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1780, &gtir_tmp_1814, 1);
                {
                    double __tlet_arg0 = gtir_tmp_1784;
                    double __tlet_arg0_0 = gtir_var_96;
                    double __tlet_arg1 = rho_;
                    double __tlet_arg1_0 = gtir_tmp_1790;
                    double __tlet_arg1_1 = gtir_var_96;
                    double __tlet_arg1_2 = gtir_tmp_1799;
                    double __tlet_result;

                    ///////////////////
                    // Tasklet code (tlet_1025_multiplies_fused_tlet_1026_plus_fused_tlet_1023_minus_fused_tlet_1024_multiplies_fused_tlet_1027_multiplies_fused_tlet_1028_divides)
                    __tlet_result = (((__tlet_arg0 - __tlet_arg1_0) * __tlet_arg1_1) / (((__tlet_arg0_0 * __tlet_arg1_2) + 1.0) * __tlet_arg1));
                    ///////////////////

                    gtir_tmp_1805 = __tlet_result;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1805, &gtir_tmp_1811, 1);
                {
                    double __tlet_arg0 = gtir_tmp_1805;
                    double __tlet_arg1 = gtir_tmp_1790;
                    double __tlet_arg1_0 = gtir_tmp_1799;
                    double __tlet_arg1_1 = rho_;
                    double __tlet_result;

                    ///////////////////
                    // Tasklet code (tlet_1029_multiplies_fused_tlet_1030_multiplies_fused_tlet_1031_plus_fused_tlet_1032_multiplies)
                    __tlet_result = ((((__tlet_arg0 * __tlet_arg1_1) * __tlet_arg1_0) + __tlet_arg1) * 0.5);
                    ///////////////////

                    gtir_tmp_1809 = __tlet_result;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1809, &gtir_tmp_1812, 1);

            }
        } else {
            {
                double gtir_tmp_1810;

                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_1033_write)
                    __tlet_out = 0.0;
                    ///////////////////

                    gtir_tmp_1810 = __tlet_out;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1810, &gtir_tmp_1812, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__ct_el_25, &gtir_tmp_1811, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_var_90, &gtir_tmp_1813, 1);

                dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1780, &gtir_tmp_1814, 1);

            }
        }
        {
            double gtir_tmp_1816;

            {
                double __tlet_arg0 = __ct_el_26;
                double __tlet_arg1 = rho_;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_1035_multiplies)
                __tlet_result = (__tlet_arg0 * __tlet_arg1);
                ///////////////////

                gtir_tmp_1816 = __tlet_result;
            }
            {
                double __tlet_arg0 = previous_level_2_1;
                double __tlet_arg0_0 = gtir_tmp_1816;
                double __tlet_arg1 = gtir_var_96;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_1036_multiplies_fused_tlet_1037_divides_fused_tlet_1038_plus)
                __tlet_result = ((__tlet_arg0 * 2.0) + (__tlet_arg0_0 / __tlet_arg1));
                ///////////////////

                gtir_tmp_1819 = __tlet_result;
            }
            {
                double __tlet_arg0 = gtir_tmp_1816;
                double __tlet_arg1 = gtir_tmp_1819;
                double __tlet_arg1_1 = gtir_var_89;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_1041_plus_fused_tlet_1042_power_fused_tlet_1039_multiplies_fused_tlet_1040_multiplies_fused_tlet_1043_multiplies_fused_tlet_1044_minimum)
                __tlet_result = min((((__tlet_arg0 * __tlet_arg1_1) * 1.25) * dace::math::pow((__tlet_arg0 + 1e-12), 0.16)), __tlet_arg1);
                ///////////////////

                gtir_tmp_1825 = __tlet_result;
            }
            {
                bool __tlet_arg0 = previous_level_2_3;
                bool __tlet_arg1 = mask_i;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_1034_or_)
                __tlet_result = (__tlet_arg0 || __tlet_arg1);
                ///////////////////

                gtir_tmp_1815 = __tlet_result;
            }

        }
        if (gtir_tmp_1815) {
            if (previous_level_2_3) {
                {
                    double gtir_tmp_1832;

                    {
                        double __tlet_arg0 = previous_level_2_2;
                        double __tlet_arg0_0 = previous_level_2_0;
                        double __tlet_arg1 = previous_level_5;
                        double __tlet_arg1_0 = __ct_el_26;
                        double __tlet_result;

                        ///////////////////
                        // Tasklet code (tlet_1046_plus_fused_tlet_1047_multiplies_fused_tlet_1048_multiplies_fused_tlet_1049_plus_fused_tlet_1050_power_fused_tlet_1045_multiplies_fused_tlet_1051_multiplies)
                        __tlet_result = ((__tlet_arg0 * 1.25) * dace::math::pow(((((__tlet_arg0_0 + __tlet_arg1_0) * 0.5) * __tlet_arg1) + 1e-12), 0.16));
                        ///////////////////

                        gtir_tmp_1832 = __tlet_result;
                    }

                    dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                    &gtir_tmp_1832, &gtir_tmp_1834, 1);

                }
            } else {
                {
                    double gtir_tmp_1833;

                    {
                        double __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_1052_write)
                        __tlet_out = 0.0;
                        ///////////////////

                        gtir_tmp_1833 = __tlet_out;
                    }

                    dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                    &gtir_tmp_1833, &gtir_tmp_1834, 1);

                }
            }
            {
                double gtir_tmp_1844;
                double gtir_tmp_1840;

                {
                    double __tlet_arg0 = gtir_tmp_1819;
                    double __tlet_arg0_0 = gtir_var_96;
                    double __tlet_arg1 = rho_;
                    double __tlet_arg1_0 = gtir_tmp_1825;
                    double __tlet_arg1_1 = gtir_var_96;
                    double __tlet_arg1_2 = gtir_tmp_1834;
                    double __tlet_result;

                    ///////////////////
                    // Tasklet code (tlet_1055_multiplies_fused_tlet_1056_plus_fused_tlet_1053_minus_fused_tlet_1054_multiplies_fused_tlet_1057_multiplies_fused_tlet_1058_divides)
                    __tlet_result = (((__tlet_arg0 - __tlet_arg1_0) * __tlet_arg1_1) / (((__tlet_arg0_0 * __tlet_arg1_2) + 1.0) * __tlet_arg1));
                    ///////////////////

                    gtir_tmp_1840 = __tlet_result;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1840, &gtir_tmp_1846, 1);
                {
                    double __tlet_arg0 = gtir_tmp_1840;
                    double __tlet_arg1 = gtir_tmp_1825;
                    double __tlet_arg1_0 = rho_;
                    double __tlet_arg1_1 = gtir_tmp_1834;
                    double __tlet_result;

                    ///////////////////
                    // Tasklet code (tlet_1059_multiplies_fused_tlet_1060_multiplies_fused_tlet_1061_plus_fused_tlet_1062_multiplies)
                    __tlet_result = ((((__tlet_arg0 * __tlet_arg1_0) * __tlet_arg1_1) + __tlet_arg1) * 0.5);
                    ///////////////////

                    gtir_tmp_1844 = __tlet_result;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1844, &gtir_tmp_1847, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_var_89, &gtir_tmp_1848, 1);

                dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1815, &gtir_tmp_1849, 1);

            }
        } else {
            {
                double gtir_tmp_1845;

                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_1063_write)
                    __tlet_out = 0.0;
                    ///////////////////

                    gtir_tmp_1845 = __tlet_out;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1845, &gtir_tmp_1847, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__ct_el_26, &gtir_tmp_1846, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_var_89, &gtir_tmp_1848, 1);

                dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1815, &gtir_tmp_1849, 1);

            }
        }
        {
            double gtir_tmp_1851;

            {
                bool __tlet_arg0 = previous_level_3_3;
                bool __tlet_arg1 = mask_g;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_1064_or_)
                __tlet_result = (__tlet_arg0 || __tlet_arg1);
                ///////////////////

                gtir_tmp_1850 = __tlet_result;
            }
            {
                double __tlet_arg0 = __ct_el_27;
                double __tlet_arg1 = rho_;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_1065_multiplies)
                __tlet_result = (__tlet_arg0 * __tlet_arg1);
                ///////////////////

                gtir_tmp_1851 = __tlet_result;
            }
            {
                double __tlet_arg0 = previous_level_3_1;
                double __tlet_arg0_0 = gtir_tmp_1851;
                double __tlet_arg1 = gtir_var_96;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_1066_multiplies_fused_tlet_1067_divides_fused_tlet_1068_plus)
                __tlet_result = ((__tlet_arg0 * 2.0) + (__tlet_arg0_0 / __tlet_arg1));
                ///////////////////

                gtir_tmp_1854 = __tlet_result;
            }
            {
                double __tlet_arg0 = gtir_tmp_1851;
                double __tlet_arg0_0 = gtir_tmp_1851;
                double __tlet_arg1 = gtir_tmp_1854;
                double __tlet_arg1_0 = gtir_var_93;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_1069_multiplies_fused_tlet_1071_plus_fused_tlet_1072_power_fused_tlet_1070_multiplies_fused_tlet_1073_multiplies_fused_tlet_1074_minimum)
                __tlet_result = min((((__tlet_arg0 * __tlet_arg1_0) * 12.24) * dace::math::pow((__tlet_arg0_0 + 1e-08), 0.217)), __tlet_arg1);
                ///////////////////

                gtir_tmp_1860 = __tlet_result;
            }

        }
        if (gtir_tmp_1850) {
            if (previous_level_3_3) {
                {
                    double gtir_tmp_1867;

                    {
                        double __tlet_arg0 = previous_level_3_2;
                        double __tlet_arg0_0 = previous_level_3_0;
                        double __tlet_arg1 = previous_level_5;
                        double __tlet_arg1_0 = __ct_el_27;
                        double __tlet_result;

                        ///////////////////
                        // Tasklet code (tlet_1076_plus_fused_tlet_1077_multiplies_fused_tlet_1078_multiplies_fused_tlet_1079_plus_fused_tlet_1080_power_fused_tlet_1075_multiplies_fused_tlet_1081_multiplies)
                        __tlet_result = ((__tlet_arg0 * 12.24) * dace::math::pow(((((__tlet_arg0_0 + __tlet_arg1_0) * 0.5) * __tlet_arg1) + 1e-08), 0.217));
                        ///////////////////

                        gtir_tmp_1867 = __tlet_result;
                    }

                    dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                    &gtir_tmp_1867, &gtir_tmp_1869, 1);

                }
            } else {
                {
                    double gtir_tmp_1868;

                    {
                        double __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_1082_write)
                        __tlet_out = 0.0;
                        ///////////////////

                        gtir_tmp_1868 = __tlet_out;
                    }

                    dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                    &gtir_tmp_1868, &gtir_tmp_1869, 1);

                }
            }
            {
                double gtir_tmp_1879;
                double gtir_tmp_1875;

                {
                    double __tlet_arg0 = gtir_tmp_1854;
                    double __tlet_arg0_0 = gtir_var_96;
                    double __tlet_arg1 = rho_;
                    double __tlet_arg1_0 = gtir_var_96;
                    double __tlet_arg1_1 = gtir_tmp_1860;
                    double __tlet_arg1_2 = gtir_tmp_1869;
                    double __tlet_result;

                    ///////////////////
                    // Tasklet code (tlet_1085_multiplies_fused_tlet_1086_plus_fused_tlet_1083_minus_fused_tlet_1087_multiplies_fused_tlet_1084_multiplies_fused_tlet_1088_divides)
                    __tlet_result = (((__tlet_arg0 - __tlet_arg1_1) * __tlet_arg1_0) / (((__tlet_arg0_0 * __tlet_arg1_2) + 1.0) * __tlet_arg1));
                    ///////////////////

                    gtir_tmp_1875 = __tlet_result;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1875, &gtir_tmp_1881, 1);
                {
                    double __tlet_arg0 = gtir_tmp_1875;
                    double __tlet_arg1 = gtir_tmp_1860;
                    double __tlet_arg1_0 = rho_;
                    double __tlet_arg1_1 = gtir_tmp_1869;
                    double __tlet_result;

                    ///////////////////
                    // Tasklet code (tlet_1089_multiplies_fused_tlet_1090_multiplies_fused_tlet_1091_plus_fused_tlet_1092_multiplies)
                    __tlet_result = ((((__tlet_arg0 * __tlet_arg1_0) * __tlet_arg1_1) + __tlet_arg1) * 0.5);
                    ///////////////////

                    gtir_tmp_1879 = __tlet_result;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1879, &gtir_tmp_1882, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_var_93, &gtir_tmp_1883, 1);

                dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1850, &gtir_tmp_1884, 1);

            }
        } else {
            {
                double gtir_tmp_1880;

                {
                    double __tlet_out;

                    ///////////////////
                    // Tasklet code (tlet_1093_write)
                    __tlet_out = 0.0;
                    ///////////////////

                    gtir_tmp_1880 = __tlet_out;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1880, &gtir_tmp_1882, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &__ct_el_27, &gtir_tmp_1881, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_var_93, &gtir_tmp_1883, 1);

                dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1850, &gtir_tmp_1884, 1);

            }
        }
        {

            {
                double __tlet_arg0 = __ct_el_23;
                double __tlet_arg1 = gtir_tmp_1776;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_1094_plus)
                __tlet_result = (__tlet_arg0 + __tlet_arg1);
                ///////////////////

                gtir_tmp_1885 = __tlet_result;
            }
            {
                double __tlet_arg0 = gtir_tmp_1811;
                double __tlet_arg1 = gtir_tmp_1881;
                double __tlet_arg1_0 = gtir_tmp_1846;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_1095_plus_fused_tlet_1096_plus)
                __tlet_result = ((__tlet_arg0 + __tlet_arg1_0) + __tlet_arg1);
                ///////////////////

                gtir_tmp_1887 = __tlet_result;
            }
            {
                bool __tlet_arg0 = previous_level_4_2;
                bool __tlet_arg1 = gtir_var_3;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_1097_or_)
                __tlet_result = (__tlet_arg0 || __tlet_arg1);
                ///////////////////

                gtir_tmp_1888 = __tlet_result;
            }

        }
        if (gtir_tmp_1888) {
            {
                double gtir_tmp_1901;
                double gtir_tmp_1903;
                double gtir_tmp_1905;
                double gtir_tmp_1944;


                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1882, &gtir_tmp_1946, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1847, &gtir_tmp_1950, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1777, &gtir_tmp_1954, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1812, &gtir_tmp_1958, 1);
                {
                    double __tlet_arg0 = __ct_el_23;
                    double __tlet_arg1 = __ct_el_24;
                    double __tlet_result;

                    ///////////////////
                    // Tasklet code (tlet_1112_plus)
                    __tlet_result = (__tlet_arg0 + __tlet_arg1);
                    ///////////////////

                    gtir_tmp_1903 = __tlet_result;
                }
                {
                    double __tlet_arg0 = __ct_el_25;
                    double __tlet_arg1 = __ct_el_27;
                    double __tlet_arg1_0 = __ct_el_26;
                    double __tlet_result;

                    ///////////////////
                    // Tasklet code (tlet_1113_plus_fused_tlet_1114_plus)
                    __tlet_result = ((__tlet_arg0 + __tlet_arg1_0) + __tlet_arg1);
                    ///////////////////

                    gtir_tmp_1905 = __tlet_result;
                }
                {
                    double __tlet_arg0 = t_;
                    double __tlet_arg0_0 = t_kp1;
                    double __tlet_arg0_1 = t_;
                    double __tlet_arg0_2 = gtir_tmp_1812;
                    double __tlet_arg1 = gtir_tmp_1882;
                    double __tlet_arg1_0 = gtir_tmp_1777;
                    double __tlet_arg1_0_0 = gtir_tmp_1847;
                    double __tlet_result;

                    ///////////////////
                    // Tasklet code (tlet_1106_multiplies_fused_tlet_1105_multiplies_fused_tlet_1107_minus_fused_tlet_1103_plus_fused_tlet_1104_plus_fused_tlet_1108_minus_fused_tlet_1109_multiplies_fused_tlet_1099_multiplies_fused_tlet_1098_multiplies_fused_tlet_1100_minus_fused_tlet_1101_minus_fused_tlet_1102_multiplies_fused_tlet_1110_plus)
                    __tlet_result = (((((__tlet_arg0 * 4192.6641119999995) - (__tlet_arg0_0 * 717.6)) - 3135383.2031928) * __tlet_arg1_0) + (((__tlet_arg0_2 + __tlet_arg1_0_0) + __tlet_arg1) * (((__tlet_arg0_1 * 2108.0) - (__tlet_arg0_0 * 717.6)) - 2899657.201)));
                    ///////////////////

                    gtir_tmp_1901 = __tlet_result;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1901, &gtir_tmp_1962, 1);
                {
                    double __tlet_arg0 = dt_;
                    double __tlet_arg0_0 = rho_;
                    double __tlet_arg0_0_0 = gtir_tmp_1887;
                    double __tlet_arg0_0_0_0 = __ct_el_22;
                    double __tlet_arg0_0_0_1 = gtir_tmp_1905;
                    double __tlet_arg0_0_1 = gtir_tmp_1887;
                    double __tlet_arg0_0_1_0 = gtir_tmp_1903;
                    double __tlet_arg0_0_2 = gtir_tmp_1885;
                    double __tlet_arg0_0_2_0 = gtir_tmp_1903;
                    double __tlet_arg0_0_3 = dt_;
                    double __tlet_arg0_1 = gtir_tmp_1885;
                    double __tlet_arg0_1_0 = rho_;
                    double __tlet_arg0_2 = gtir_tmp_1885;
                    double __tlet_arg0_2_0 = gtir_tmp_1903;
                    double __tlet_arg1 = dz_;
                    double __tlet_arg1_0 = dz_;
                    double __tlet_arg1_0_0 = rho_;
                    double __tlet_arg1_0_0_0 = dz_;
                    double __tlet_arg1_0_1 = gtir_tmp_1887;
                    double __tlet_arg1_0_1_0 = t_;
                    double __tlet_arg1_0_2 = gtir_tmp_1905;
                    double __tlet_arg1_0_3 = previous_level_4_1;
                    double __tlet_arg1_1 = __ct_el_22;
                    double __tlet_arg1_2 = gtir_tmp_1901;
                    double __tlet_result;

                    ///////////////////
                    // Tasklet code (tlet_1116_plus_fused_tlet_1117_plus_fused_tlet_1118_minus_fused_tlet_1119_multiplies_fused_tlet_1120_multiplies_fused_tlet_1121_plus_fused_tlet_1122_multiplies_fused_tlet_1123_plus_fused_tlet_1124_multiplies_fused_tlet_1125_plus_fused_tlet_1127_multiplies_fused_tlet_1126_multiplies_fused_tlet_1128_minus_fused_tlet_1129_multiplies_fused_tlet_1130_minus_fused_tlet_1115_multiplies_fused_tlet_1131_multiplies_fused_tlet_1111_multiplies_fused_tlet_1132_plus_fused_tlet_1133_multiplies_fused_tlet_1134_minus_fused_tlet_1141_plus_fused_tlet_1142_plus_fused_tlet_1143_minus_fused_tlet_1144_multiplies_fused_tlet_1145_multiplies_fused_tlet_1146_plus_fused_tlet_1147_multiplies_fused_tlet_1148_plus_fused_tlet_1149_multiplies_fused_tlet_1150_plus_fused_tlet_1151_multiplies_fused_tlet_1152_multiplies_fused_tlet_1137_multiplies_fused_tlet_1136_multiplies_fused_tlet_1138_plus_fused_tlet_1135_multiplies_fused_tlet_1139_multiplies_fused_tlet_1140_plus_fused_tlet_1153_divides)
                    __tlet_result = (((((__tlet_arg0 * __tlet_arg1_0_3) + ((__tlet_arg0_1_0 * __tlet_arg1_0_0_0) * ((((((((1.0 - ((__tlet_arg0_2_0 + __tlet_arg1_0_2) + __tlet_arg1_1)) * 717.6) + (__tlet_arg0_0_0_0 * 1407.95)) + (__tlet_arg0_0_2_0 * 4192.6641119999995)) + (__tlet_arg0_0_0_1 * 2108.0)) * __tlet_arg1_0_1_0) - (__tlet_arg0_0_1_0 * 3135383.2031928)) - (__tlet_arg0_0_0_1 * 2899657.201)))) - (__tlet_arg0_0_3 * __tlet_arg1_2)) + ((__tlet_arg0_0 * __tlet_arg1_0) * ((__tlet_arg0_1 * 3135383.2031928) + (__tlet_arg0_0_0 * 2899657.201)))) / (((((((1.0 - ((__tlet_arg0_2 + __tlet_arg1_0_1) + __tlet_arg1_1)) * 717.6) + (__tlet_arg0_0_0_0 * 1407.95)) + (__tlet_arg0_0_2 * 4192.6641119999995)) + (__tlet_arg0_0_1 * 2108.0)) * __tlet_arg1_0_0) * __tlet_arg1));
                    ///////////////////

                    gtir_tmp_1944 = __tlet_result;
                }

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1944, &gtir_tmp_1961, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1881, &gtir_tmp_1945, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1883, &gtir_tmp_1947, 1);

                dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1884, &gtir_tmp_1948, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1846, &gtir_tmp_1949, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1848, &gtir_tmp_1951, 1);

                dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1849, &gtir_tmp_1952, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1776, &gtir_tmp_1953, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1778, &gtir_tmp_1955, 1);

                dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1779, &gtir_tmp_1956, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1811, &gtir_tmp_1957, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1813, &gtir_tmp_1959, 1);

                dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1814, &gtir_tmp_1960, 1);

                dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1888, &gtir_tmp_1963, 1);

            }
        } else {
            {


                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1881, &gtir_tmp_1945, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1882, &gtir_tmp_1946, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1883, &gtir_tmp_1947, 1);

                dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1884, &gtir_tmp_1948, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1846, &gtir_tmp_1949, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1847, &gtir_tmp_1950, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1848, &gtir_tmp_1951, 1);

                dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1849, &gtir_tmp_1952, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1776, &gtir_tmp_1953, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1777, &gtir_tmp_1954, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1778, &gtir_tmp_1955, 1);

                dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1779, &gtir_tmp_1956, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1811, &gtir_tmp_1957, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1812, &gtir_tmp_1958, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1813, &gtir_tmp_1959, 1);

                dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1814, &gtir_tmp_1960, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &t_, &gtir_tmp_1961, 1);

                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                &previous_level_4_1, &gtir_tmp_1962, 1);

                dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
                &gtir_tmp_1888, &gtir_tmp_1963, 1);

            }
        }
        {


            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1945, &__output_0_0, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1949, &__output_1_0, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1953, &__output_2_0, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1957, &__output_3_0, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1961, &__output_4_0, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1946, &__output_0_1, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1950, &__output_1_1, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1954, &__output_2_1, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1958, &__output_3_1, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1962, &__output_4_1, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1947, &__output_0_2, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1951, &__output_1_2, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1955, &__output_2_2, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1959, &__output_3_2, 1);

            dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1963, &__output_4_2, 1);

            dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1948, &__output_0_3, 1);

            dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1952, &__output_1_3, 1);

            dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1956, &__output_2_3, 1);

            dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1960, &__output_3_3, 1);

        }
    } else {
        {
            double gtir_tmp_1964;
            bool gtir_tmp_1965;
            double gtir_tmp_1966;
            bool gtir_tmp_1967;
            double gtir_tmp_1968;
            bool gtir_tmp_1969;
            double gtir_tmp_1970;
            bool gtir_tmp_1971;
            bool gtir_tmp_1972;

            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_1154_write)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1964 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1964, &__output_0_1, 1);
            {
                bool __tlet_out;

                ///////////////////
                // Tasklet code (tlet_1155_write)
                __tlet_out = false;
                ///////////////////

                gtir_tmp_1965 = __tlet_out;
            }

            dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1965, &__output_0_3, 1);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_1156_write)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1966 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1966, &__output_1_1, 1);
            {
                bool __tlet_out;

                ///////////////////
                // Tasklet code (tlet_1157_write)
                __tlet_out = false;
                ///////////////////

                gtir_tmp_1967 = __tlet_out;
            }

            dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1967, &__output_1_3, 1);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_1158_write)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1968 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1968, &__output_2_1, 1);
            {
                bool __tlet_out;

                ///////////////////
                // Tasklet code (tlet_1159_write)
                __tlet_out = false;
                ///////////////////

                gtir_tmp_1969 = __tlet_out;
            }

            dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1969, &__output_2_3, 1);
            {
                double __tlet_out;

                ///////////////////
                // Tasklet code (tlet_1160_write)
                __tlet_out = 0.0;
                ///////////////////

                gtir_tmp_1970 = __tlet_out;
            }

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1970, &__output_3_1, 1);
            {
                bool __tlet_out;

                ///////////////////
                // Tasklet code (tlet_1161_write)
                __tlet_out = false;
                ///////////////////

                gtir_tmp_1971 = __tlet_out;
            }

            dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1971, &__output_3_3, 1);
            {
                bool __tlet_out;

                ///////////////////
                // Tasklet code (tlet_1162_write)
                __tlet_out = false;
                ///////////////////

                gtir_tmp_1972 = __tlet_out;
            }

            dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1972, &__output_4_2, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__ct_el_27, &__output_0_0, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_var_93, &__output_0_2, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_var_93, &__output_2_2, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__ct_el_26, &__output_1_0, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_var_89, &__output_1_2, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__ct_el_24, &__output_2_0, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &__ct_el_25, &__output_3_0, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_var_90, &__output_3_2, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &t_, &__output_4_0, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &previous_level_4_1, &__output_4_1, 1);

        }
    }
}

DACE_DFI void scan_0_0_0_27(const double* __restrict__ __ct_flat_el_0_q_, const double* __restrict__ __ct_flat_el_1_q_, const double * __restrict__ __ct_flat_el_2_q_, const double * __restrict__ __ct_flat_el_3_q_, const double * __restrict__ __ct_flat_el_4_q_, const double * __restrict__ __ct_flat_el_5_q_, const double&  __gtir_scan_input_previous_level_0_0, const double&  __gtir_scan_input_previous_level_0_1, const double&  __gtir_scan_input_previous_level_0_2, const bool&  __gtir_scan_input_previous_level_0_3, const double&  __gtir_scan_input_previous_level_1_0, const double&  __gtir_scan_input_previous_level_1_1, const double&  __gtir_scan_input_previous_level_1_2, const bool&  __gtir_scan_input_previous_level_1_3, const double&  __gtir_scan_input_previous_level_2_0, const double&  __gtir_scan_input_previous_level_2_1, const double&  __gtir_scan_input_previous_level_2_2, const bool&  __gtir_scan_input_previous_level_2_3, const double&  __gtir_scan_input_previous_level_3_0, const double&  __gtir_scan_input_previous_level_3_1, const double&  __gtir_scan_input_previous_level_3_2, const bool&  __gtir_scan_input_previous_level_3_3, const double&  __gtir_scan_input_previous_level_4_1, const bool&  __gtir_scan_input_previous_level_4_2, const double&  __gtir_scan_input_previous_level_5, const double&  dt_, const double* __restrict__ dz_, const double *  gtir_tmp_1675, const double *  gtir_tmp_1675_0, const bool * __restrict__ mask_g, const bool * __restrict__ mask_i, const bool * __restrict__ mask_r, const bool * __restrict__ mask_s, const double* __restrict__ rho_, const double *  t_, double* __restrict__ __gtir_scan_output_previous_level_0_0, double* __restrict__ __gtir_scan_output_previous_level_0_1, double* __restrict__ __gtir_scan_output_previous_level_1_0, double* __restrict__ __gtir_scan_output_previous_level_1_1, double* __restrict__ __gtir_scan_output_previous_level_2_0, double* __restrict__ __gtir_scan_output_previous_level_2_1, double* __restrict__ __gtir_scan_output_previous_level_3_0, double* __restrict__ __gtir_scan_output_previous_level_3_1, double* __restrict__ __gtir_scan_output_previous_level_4_0, double* __restrict__ __gtir_scan_output_previous_level_4_1, double* __restrict__ __gtir_scan_output_previous_level_6, int ____ct_flat_el_0_q__Cell_range_0, int ____ct_flat_el_0_q__K_range_0, int ____ct_flat_el_1_q__Cell_range_0, int ____ct_flat_el_1_q__K_range_0, int ____ct_flat_el_2_q__Cell_range_0, int ____ct_flat_el_2_q__K_range_0, int ____ct_flat_el_3_q__Cell_range_0, int ____ct_flat_el_3_q__K_range_0, int ____ct_flat_el_4_q__Cell_range_0, int ____ct_flat_el_4_q__K_range_0, int ____ct_flat_el_5_q__Cell_range_0, int ____ct_flat_el_5_q__K_range_0, int __dz_K_stride, int __dz__Cell_range_0, int __dz__K_range_0, int __mask_g_Cell_range_0, int __mask_g_K_range_0, int __mask_i_Cell_range_0, int __mask_i_K_range_0, int __mask_r_Cell_range_0, int __mask_r_K_range_0, int __mask_s_Cell_range_0, int __mask_s_K_range_0, int __pflx_K_stride_0, int __q_out_0_Cell_stride_0, int __q_out_0_K_stride_0, int __q_out_1_Cell_stride_0, int __q_out_1_K_stride_0, int __q_out_2_K_stride_0, int __q_out_3_K_stride_0, int __q_out_4_K_stride_0, int __q_out_5_K_stride_0, int __rho_K_stride, int __rho__Cell_range_0, int __rho__K_range_0, int __t__Cell_range_0, int __t__K_range_0, int __t_kp1_Cell_range_0, int __t_kp1_K_range_0, int __t_out_K_stride_0, int i_Cell_gtx_horizontal) {
    double previous_level_0_0;
    double previous_level_0_1;
    double previous_level_0_2;
    bool previous_level_0_3;
    double previous_level_1_0;
    double previous_level_1_1;
    double previous_level_1_2;
    bool previous_level_1_3;
    double previous_level_2_0;
    double previous_level_2_1;
    double previous_level_2_2;
    bool previous_level_2_3;
    double previous_level_3_0;
    double previous_level_3_1;
    double previous_level_3_2;
    bool previous_level_3_3;
    double previous_level_4_1;
    bool previous_level_4_2;
    double previous_level_5;
    double gtir_tmp_1973;
    double gtir_tmp_1974;
    double gtir_tmp_1975;
    bool gtir_tmp_1976;
    double gtir_tmp_1977;
    double gtir_tmp_1978;
    double gtir_tmp_1979;
    bool gtir_tmp_1980;
    double gtir_tmp_1981;
    double gtir_tmp_1982;
    double gtir_tmp_1983;
    bool gtir_tmp_1984;
    double gtir_tmp_1985;
    double gtir_tmp_1986;
    double gtir_tmp_1987;
    bool gtir_tmp_1988;
    double gtir_tmp_1990;
    bool gtir_tmp_1991;
    double gtir_tmp_1995;
    int64_t i_K_gtx_vertical;

    {


        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
        &__gtir_scan_input_previous_level_0_0, &previous_level_0_0, 1);

        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
        &__gtir_scan_input_previous_level_0_1, &previous_level_0_1, 1);

        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
        &__gtir_scan_input_previous_level_0_2, &previous_level_0_2, 1);

        dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
        &__gtir_scan_input_previous_level_0_3, &previous_level_0_3, 1);

        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
        &__gtir_scan_input_previous_level_1_0, &previous_level_1_0, 1);

        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
        &__gtir_scan_input_previous_level_1_1, &previous_level_1_1, 1);

        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
        &__gtir_scan_input_previous_level_1_2, &previous_level_1_2, 1);

        dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
        &__gtir_scan_input_previous_level_1_3, &previous_level_1_3, 1);

        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
        &__gtir_scan_input_previous_level_2_0, &previous_level_2_0, 1);

        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
        &__gtir_scan_input_previous_level_2_1, &previous_level_2_1, 1);

        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
        &__gtir_scan_input_previous_level_2_2, &previous_level_2_2, 1);

        dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
        &__gtir_scan_input_previous_level_2_3, &previous_level_2_3, 1);

        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
        &__gtir_scan_input_previous_level_3_0, &previous_level_3_0, 1);

        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
        &__gtir_scan_input_previous_level_3_1, &previous_level_3_1, 1);

        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
        &__gtir_scan_input_previous_level_3_2, &previous_level_3_2, 1);

        dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
        &__gtir_scan_input_previous_level_3_3, &previous_level_3_3, 1);

        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
        &__gtir_scan_input_previous_level_4_1, &previous_level_4_1, 1);

        dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
        &__gtir_scan_input_previous_level_4_2, &previous_level_4_2, 1);

        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
        &__gtir_scan_input_previous_level_5, &previous_level_5, 1);

    }
    for (i_K_gtx_vertical = 0; (i_K_gtx_vertical < 70); i_K_gtx_vertical = (i_K_gtx_vertical + 1)) {
        {
            double gtir_tmp_1698;
            double gtir_tmp_1700;
            double gtir_tmp_1703;
            double gtir_tmp_1708;
            double gtir_tmp_1710;
            bool gtir_tmp_1711;
            double gtir_tmp_1733;
            double gtir_tmp_1735;
            double gtir_tmp_1736;
            bool gtir_tmp_1739;
            bool gtir_tmp_1744;
            double __gt4py_concat_where_mapper_temp_t_kp1;


            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            rho_ + (((__rho_K_stride * ((- __rho__K_range_0) + i_K_gtx_vertical)) - __rho__Cell_range_0) + i_Cell_gtx_horizontal), &gtir_tmp_1995, 1);
            {
                double __tlet_arg0 = t_[((((- __t__Cell_range_0) - (160 * __t__K_range_0)) + i_Cell_gtx_horizontal) + (160 * i_K_gtx_vertical))];
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_931_minimum_fused_tlet_932_maximum_fused_tlet_933_minus)
                __tlet_result = (max(min(__tlet_arg0, 273.15), 233.14999999999998) - 273.15);
                ///////////////////

                gtir_tmp_1703 = __tlet_result;
            }
            {
                double __tlet_arg0 = gtir_tmp_1703;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_939_multiplies_fused_tlet_940_exp)
                __tlet_result = dace::math::exp((__tlet_arg0 * -0.107));
                ///////////////////

                gtir_tmp_1710 = __tlet_result;
            }
            {
                double __tlet_arg0 = gtir_tmp_1703;
                double __tlet_arg1 = gtir_tmp_1703;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_934_multiplies_fused_tlet_935_plus_fused_tlet_936_multiplies_fused_tlet_937_minus_fused_tlet_938_power)
                __tlet_result = dace::math::pow(10.0, ((((__tlet_arg0 * 0.000327) + 0.0545) * __tlet_arg1) - 1.65));
                ///////////////////

                gtir_tmp_1708 = __tlet_result;
            }
            {
                double __tlet_arg0 = __ct_flat_el_3_q_[((((- ____ct_flat_el_3_q__Cell_range_0) - (160 * ____ct_flat_el_3_q__K_range_0)) + i_Cell_gtx_horizontal) + (160 * i_K_gtx_vertical))];
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_941_greater)
                __tlet_result = (__tlet_arg0 > 1e-15);
                ///////////////////

                gtir_tmp_1711 = __tlet_result;
            }
            {
                double __tlet_arg0 = dt_;
                double __tlet_arg0_0 = dz_[(((__dz_K_stride * ((- __dz__K_range_0) + i_K_gtx_vertical)) - __dz__Cell_range_0) + i_Cell_gtx_horizontal)];
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_927_multiplies_fused_tlet_928_divides)
                __tlet_result = (__tlet_arg0 / (__tlet_arg0_0 * 2.0));
                ///////////////////

                gtir_tmp_1698 = __tlet_result;
            }
            {
                double __tlet_arg1 = rho_[(((__rho_K_stride * ((- __rho__K_range_0) + i_K_gtx_vertical)) - __rho__Cell_range_0) + i_Cell_gtx_horizontal)];
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_929_divides_fused_tlet_930_sqrt)
                __tlet_result = dace::math::sqrt((1.225 / __tlet_arg1));
                ///////////////////

                gtir_tmp_1700 = __tlet_result;
            }
            {
                double __tlet_arg0 = gtir_tmp_1700;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_965_power)
                __tlet_result = dace::math::pow(__tlet_arg0, 0.6666666666666666);
                ///////////////////

                gtir_tmp_1736 = __tlet_result;
            }
            if_stmt_87_215_1_7(gtir_tmp_1711, __ct_flat_el_3_q_[((((- ____ct_flat_el_3_q__Cell_range_0) - (160 * ____ct_flat_el_3_q__K_range_0)) + i_Cell_gtx_horizontal) + (160 * i_K_gtx_vertical))], gtir_tmp_1708, gtir_tmp_1703, gtir_tmp_1710, rho_[(((__rho_K_stride * ((- __rho__K_range_0) + i_K_gtx_vertical)) - __rho__Cell_range_0) + i_Cell_gtx_horizontal)], gtir_tmp_1733);
            {
                double __tlet_arg0 = gtir_tmp_1700;
                double __tlet_arg0_0 = gtir_tmp_1733;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_963_power_fused_tlet_964_multiplies)
                __tlet_result = (__tlet_arg0 * dace::math::pow(__tlet_arg0_0, -0.16666666666666666));
                ///////////////////

                gtir_tmp_1735 = __tlet_result;
            }
            {
                bool __tlet_arg0 = mask_r[((((- __mask_r_Cell_range_0) - (160 * __mask_r_K_range_0)) + i_Cell_gtx_horizontal) + (160 * i_K_gtx_vertical))];
                bool __tlet_arg1 = mask_g[((((- __mask_g_Cell_range_0) - (160 * __mask_g_K_range_0)) + i_Cell_gtx_horizontal) + (160 * i_K_gtx_vertical))];
                bool __tlet_arg1_0 = mask_i[((((- __mask_i_Cell_range_0) - (160 * __mask_i_K_range_0)) + i_Cell_gtx_horizontal) + (160 * i_K_gtx_vertical))];
                bool __tlet_arg1_1 = mask_s[((((- __mask_s_Cell_range_0) - (160 * __mask_s_K_range_0)) + i_Cell_gtx_horizontal) + (160 * i_K_gtx_vertical))];
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_966_or__fused_tlet_967_or__fused_tlet_968_or_)
                __tlet_result = (((__tlet_arg0 || __tlet_arg1_1) || __tlet_arg1_0) || __tlet_arg1);
                ///////////////////

                gtir_tmp_1739 = __tlet_result;
            }
            {
                bool __tlet_arg0 = gtir_tmp_1739;
                bool __tlet_arg0_0 = previous_level_0_3;
                bool __tlet_arg1 = previous_level_4_2;
                bool __tlet_arg1_0 = previous_level_2_3;
                bool __tlet_arg1_1 = previous_level_1_3;
                bool __tlet_arg1_2 = previous_level_3_3;
                bool __tlet_result;

                ///////////////////
                // Tasklet code (tlet_969_or__fused_tlet_970_or__fused_tlet_971_or__fused_tlet_972_or__fused_tlet_973_or_)
                __tlet_result = (__tlet_arg0 || ((((__tlet_arg0_0 || __tlet_arg1_1) || __tlet_arg1_0) || __tlet_arg1_2) || __tlet_arg1));
                ///////////////////

                gtir_tmp_1744 = __tlet_result;
            }
            {
                const double * __in0 = &gtir_tmp_1675[0];
                const double * __in1 = &gtir_tmp_1675_0[0];
                double __out;

                ///////////////////
                // Tasklet code (concat_where_tasklet_t_kp1)
                __out = ((((0 <= ((- __t_kp1_Cell_range_0) + i_Cell_gtx_horizontal)) && (((- __t_kp1_Cell_range_0) + i_Cell_gtx_horizontal) <= 159)) && ((0 <= ((- __t_kp1_K_range_0) + i_K_gtx_vertical)) && (((- __t_kp1_K_range_0) + i_K_gtx_vertical) <= 68))) ? __in0[(((((- __t_kp1_Cell_range_0) - (160 * __t_kp1_K_range_0)) + i_Cell_gtx_horizontal) + (160 * i_K_gtx_vertical)) + 160)] : __in1[((((- __t_kp1_Cell_range_0) - (160 * __t_kp1_K_range_0)) + i_Cell_gtx_horizontal) + (160 * i_K_gtx_vertical))]);
                ///////////////////

                __gt4py_concat_where_mapper_temp_t_kp1 = __out;
            }
            if_stmt_88_215_1_28(gtir_tmp_1744, __ct_flat_el_0_q_[((__q_out_0_Cell_stride_0 * ((- ____ct_flat_el_0_q__Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_out_0_K_stride_0 * ((- ____ct_flat_el_0_q__K_range_0) + i_K_gtx_vertical)))], __ct_flat_el_1_q_[((__q_out_1_Cell_stride_0 * ((- ____ct_flat_el_1_q__Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_out_1_K_stride_0 * ((- ____ct_flat_el_1_q__K_range_0) + i_K_gtx_vertical)))], __ct_flat_el_2_q_[((((- ____ct_flat_el_2_q__Cell_range_0) - (160 * ____ct_flat_el_2_q__K_range_0)) + i_Cell_gtx_horizontal) + (160 * i_K_gtx_vertical))], __ct_flat_el_3_q_[((((- ____ct_flat_el_3_q__Cell_range_0) - (160 * ____ct_flat_el_3_q__K_range_0)) + i_Cell_gtx_horizontal) + (160 * i_K_gtx_vertical))], __ct_flat_el_4_q_[((((- ____ct_flat_el_4_q__Cell_range_0) - (160 * ____ct_flat_el_4_q__K_range_0)) + i_Cell_gtx_horizontal) + (160 * i_K_gtx_vertical))], __ct_flat_el_5_q_[((((- ____ct_flat_el_5_q__Cell_range_0) - (160 * ____ct_flat_el_5_q__K_range_0)) + i_Cell_gtx_horizontal) + (160 * i_K_gtx_vertical))], dt_, dz_[(((__dz_K_stride * ((- __dz__K_range_0) + i_K_gtx_vertical)) - __dz__Cell_range_0) + i_Cell_gtx_horizontal)], gtir_tmp_1739, gtir_tmp_1736, gtir_tmp_1735, gtir_tmp_1700, gtir_tmp_1698, mask_g[((((- __mask_g_Cell_range_0) - (160 * __mask_g_K_range_0)) + i_Cell_gtx_horizontal) + (160 * i_K_gtx_vertical))], mask_i[((((- __mask_i_Cell_range_0) - (160 * __mask_i_K_range_0)) + i_Cell_gtx_horizontal) + (160 * i_K_gtx_vertical))], mask_r[((((- __mask_r_Cell_range_0) - (160 * __mask_r_K_range_0)) + i_Cell_gtx_horizontal) + (160 * i_K_gtx_vertical))], mask_s[((((- __mask_s_Cell_range_0) - (160 * __mask_s_K_range_0)) + i_Cell_gtx_horizontal) + (160 * i_K_gtx_vertical))], previous_level_0_0, previous_level_0_1, previous_level_0_2, previous_level_0_3, previous_level_1_0, previous_level_1_1, previous_level_1_2, previous_level_1_3, previous_level_2_0, previous_level_2_1, previous_level_2_2, previous_level_2_3, previous_level_3_0, previous_level_3_1, previous_level_3_2, previous_level_3_3, previous_level_4_1, previous_level_4_2, previous_level_5, rho_[(((__rho_K_stride * ((- __rho__K_range_0) + i_K_gtx_vertical)) - __rho__Cell_range_0) + i_Cell_gtx_horizontal)], t_[((((- __t__Cell_range_0) - (160 * __t__K_range_0)) + i_Cell_gtx_horizontal) + (160 * i_K_gtx_vertical))], __gt4py_concat_where_mapper_temp_t_kp1, gtir_tmp_1973, gtir_tmp_1974, gtir_tmp_1975, gtir_tmp_1976, gtir_tmp_1977, gtir_tmp_1978, gtir_tmp_1979, gtir_tmp_1980, gtir_tmp_1981, gtir_tmp_1982, gtir_tmp_1983, gtir_tmp_1984, gtir_tmp_1985, gtir_tmp_1986, gtir_tmp_1987, gtir_tmp_1988, __gtir_scan_output_previous_level_4_0[(__t_out_K_stride_0 * i_K_gtx_vertical)], gtir_tmp_1990, gtir_tmp_1991);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1973, __gtir_scan_output_previous_level_3_0 + (__q_out_5_K_stride_0 * i_K_gtx_vertical), 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1977, __gtir_scan_output_previous_level_2_0 + (__q_out_4_K_stride_0 * i_K_gtx_vertical), 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1981, __gtir_scan_output_previous_level_0_0 + (__q_out_2_K_stride_0 * i_K_gtx_vertical), 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1985, __gtir_scan_output_previous_level_1_0 + (__q_out_3_K_stride_0 * i_K_gtx_vertical), 1);
            {
                double __tlet_arg0 = gtir_tmp_1986;
                double __tlet_arg1 = gtir_tmp_1982;
                double __tlet_arg1_0 = gtir_tmp_1974;
                double __tlet_arg1_1 = gtir_tmp_1978;
                double __tlet_result;

                ///////////////////
                // Tasklet code (tlet_1163_plus_fused_tlet_1164_plus_fused_tlet_1165_plus)
                __tlet_result = (((__tlet_arg0 + __tlet_arg1_1) + __tlet_arg1_0) + __tlet_arg1);
                ///////////////////

                __gtir_scan_output_previous_level_6[(__pflx_K_stride_0 * i_K_gtx_vertical)] = __tlet_result;
            }

        }
        {


            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1981, &previous_level_0_0, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1982, &previous_level_0_1, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1983, &previous_level_0_2, 1);

            dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1984, &previous_level_0_3, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1985, &previous_level_1_0, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1986, &previous_level_1_1, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1987, &previous_level_1_2, 1);

            dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1988, &previous_level_1_3, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1977, &previous_level_2_0, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1978, &previous_level_2_1, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1979, &previous_level_2_2, 1);

            dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1980, &previous_level_2_3, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1973, &previous_level_3_0, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1974, &previous_level_3_1, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1975, &previous_level_3_2, 1);

            dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1976, &previous_level_3_3, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1990, &previous_level_4_1, 1);

            dace::CopyND<bool, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1991, &previous_level_4_2, 1);

            dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
            &gtir_tmp_1995, &previous_level_5, 1);

        }

    }
    {


        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
        &gtir_tmp_1982, __gtir_scan_output_previous_level_0_1, 1);

        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
        &gtir_tmp_1986, __gtir_scan_output_previous_level_1_1, 1);

        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
        &gtir_tmp_1978, __gtir_scan_output_previous_level_2_1, 1);

        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
        &gtir_tmp_1974, __gtir_scan_output_previous_level_3_1, 1);

        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
        &gtir_tmp_1990, __gtir_scan_output_previous_level_4_1, 1);

    }
}



int __dace_init_cuda(graupel_run_state_t *__state, int __dz_Cell_range_0, int __dz_K_range_0, int __dz_K_stride, int __p_Cell_range_0, int __p_K_range_0, int __p_K_stride, int __pflx_Cell_range_0, int __pflx_K_range_0, int __pflx_K_stride, int __pg_Cell_range_0, int __pg_K_range_0, int __pg_K_stride, int __pi_Cell_range_0, int __pi_K_range_0, int __pi_K_stride, int __pr_Cell_range_0, int __pr_K_range_0, int __pr_K_stride, int __pre_Cell_range_0, int __pre_K_range_0, int __pre_K_stride, int __ps_Cell_range_0, int __ps_K_range_0, int __ps_K_stride, int __q_in_0_Cell_range_0, int __q_in_0_Cell_stride, int __q_in_0_K_range_0, int __q_in_0_K_stride, int __q_in_1_Cell_range_0, int __q_in_1_Cell_stride, int __q_in_1_K_range_0, int __q_in_1_K_stride, int __q_in_2_Cell_range_0, int __q_in_2_Cell_stride, int __q_in_2_K_range_0, int __q_in_2_K_stride, int __q_in_3_Cell_range_0, int __q_in_3_Cell_stride, int __q_in_3_K_range_0, int __q_in_3_K_stride, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride, int __q_in_4_K_range_0, int __q_in_4_K_stride, int __q_in_5_Cell_range_0, int __q_in_5_Cell_stride, int __q_in_5_K_range_0, int __q_in_5_K_stride, int __q_out_0_Cell_range_0, int __q_out_0_Cell_stride, int __q_out_0_K_range_0, int __q_out_0_K_stride, int __q_out_1_Cell_range_0, int __q_out_1_Cell_stride, int __q_out_1_K_range_0, int __q_out_1_K_stride, int __q_out_2_Cell_range_0, int __q_out_2_Cell_stride, int __q_out_2_K_range_0, int __q_out_2_K_stride, int __q_out_3_Cell_range_0, int __q_out_3_Cell_stride, int __q_out_3_K_range_0, int __q_out_3_K_stride, int __q_out_4_Cell_range_0, int __q_out_4_Cell_stride, int __q_out_4_K_range_0, int __q_out_4_K_stride, int __q_out_5_Cell_range_0, int __q_out_5_Cell_stride, int __q_out_5_K_range_0, int __q_out_5_K_stride, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride, int __t_out_Cell_range_0, int __t_out_K_range_0, int __t_out_K_stride, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride, int gt_metrics_level) {
    int count;

    // Check that we are able to run cuda code
    if (cudaGetDeviceCount(&count) != cudaSuccess)
    {
        printf("ERROR: GPU drivers are not configured or cuda-capable device "
               "not found\n");
        return 1;
    }
    if (count == 0)
    {
        printf("ERROR: No cuda-capable devices found\n");
        return 2;
    }

    // Initialize cuda before we run the application
    float *dev_X;
    DACE_GPU_CHECK(cudaMalloc((void **) &dev_X, 1));
    DACE_GPU_CHECK(cudaFree(dev_X));

    

    __state->gpu_context = new dace::cuda::Context(1, 1);

    // Create cuda streams and events
    for(int i = 0; i < 1; ++i) {
        DACE_GPU_CHECK(cudaStreamCreateWithFlags(&__state->gpu_context->internal_streams[i], cudaStreamNonBlocking));
        __state->gpu_context->streams[i] = __state->gpu_context->internal_streams[i]; // Allow for externals to modify streams
    }
    for(int i = 0; i < 1; ++i) {
        DACE_GPU_CHECK(cudaEventCreateWithFlags(&__state->gpu_context->events[i], cudaEventDisableTiming));
    }

    __dace_gpu_set_all_streams(__state, cudaStreamDefault);


    return 0;
}

int __dace_exit_cuda(graupel_run_state_t *__state) {
    

    // Synchronize and check for CUDA errors
    int __err = static_cast<int>(__state->gpu_context->lasterror);
    if (__err == 0)
        __err = static_cast<int>(cudaDeviceSynchronize());

    // Destroy cuda streams and events
    for(int i = 0; i < 1; ++i) {
        DACE_GPU_CHECK(cudaStreamDestroy(__state->gpu_context->internal_streams[i]));
    }
    for(int i = 0; i < 1; ++i) {
        DACE_GPU_CHECK(cudaEventDestroy(__state->gpu_context->events[i]));
    }

    delete __state->gpu_context;
    return __err;
}

bool __dace_gpu_set_stream(graupel_run_state_t *__state, int streamid, gpuStream_t stream)
{
    if (streamid < 0 || streamid >= 1)
        return false;

    __state->gpu_context->streams[streamid] = stream;

    return true;
}

void __dace_gpu_set_all_streams(graupel_run_state_t *__state, gpuStream_t stream)
{
    for (int i = 0; i < 1; ++i)
        __state->gpu_context->streams[i] = stream;
}

__global__ void __maxnreg__(128)  map_0_fieldop_0_0_115(double * __restrict__ gtir_tmp_1675, double * __restrict__ gtir_tmp_2000, double * __restrict__ gtir_tmp_2002, double * __restrict__ gtir_tmp_2004, double * __restrict__ gtir_tmp_2006, bool * __restrict__ gtir_tmp_2009, bool * __restrict__ gtir_tmp_2012, bool * __restrict__ gtir_tmp_2015, bool * __restrict__ gtir_tmp_2018, const double * __restrict__ p, const double * __restrict__ q_in_0, const double * __restrict__ q_in_1, const double * __restrict__ q_in_2, const double * __restrict__ q_in_3, const double * __restrict__ q_in_4, const double * __restrict__ q_in_5, double * __restrict__ q_out_0, double * __restrict__ q_out_1, const double * __restrict__ rho, const double * __restrict__ te, int __p_Cell_range_0, int __p_K_range_0, int __p_K_stride, int __q_in_0_Cell_range_0, int __q_in_0_Cell_stride, int __q_in_0_K_range_0, int __q_in_0_K_stride, int __q_in_1_Cell_range_0, int __q_in_1_Cell_stride, int __q_in_1_K_range_0, int __q_in_1_K_stride, int __q_in_2_Cell_range_0, int __q_in_2_Cell_stride, int __q_in_2_K_range_0, int __q_in_2_K_stride, int __q_in_3_Cell_range_0, int __q_in_3_Cell_stride, int __q_in_3_K_range_0, int __q_in_3_K_stride, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride, int __q_in_4_K_range_0, int __q_in_4_K_stride, int __q_in_5_Cell_range_0, int __q_in_5_Cell_stride, int __q_in_5_K_range_0, int __q_in_5_K_stride, int __q_out_0_Cell_range_0, int __q_out_0_Cell_stride, int __q_out_0_K_range_0, int __q_out_0_K_stride, int __q_out_1_Cell_range_0, int __q_out_1_Cell_stride, int __q_out_1_K_range_0, int __q_out_1_K_stride, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride) {
    {
        {
            int b_i_Cell_gtx_horizontal = (32 * blockIdx.x);
            int b_i_K_gtx_vertical = (8 * blockIdx.y);
            {
                {
                    {
                        int i_Cell_gtx_horizontal = (threadIdx.x + b_i_Cell_gtx_horizontal);
                        int i_K_gtx_vertical = (threadIdx.y + b_i_K_gtx_vertical);
                        double gtir_tmp_1619;
                        double gtir_tmp_1621;
                        double gtir_tmp_1674;
                        double gtir_tmp_2001;
                        double gtir_tmp_1999;
                        double gtir_tmp_2003;
                        double gtir_tmp_2005;
                        bool __map_fusion_gtir_tmp_45;
                        double gtir_tmp_21_0;
                        bool gtir_tmp_43_0;
                        double gtir_tmp_15_0;
                        double gtir_tmp_8_0;
                        double gtir_tmp_14_0;
                        double gtir_tmp_30_0;
                        double gtir_tmp_16_0;
                        double gtir_tmp_11_0;
                        double gtir_tmp_2010_0;
                        double gtir_tmp_2007_0;
                        double gtir_tmp_2016_0;
                        double gtir_tmp_2013_0;
                        if (i_Cell_gtx_horizontal >= b_i_Cell_gtx_horizontal && i_Cell_gtx_horizontal < (Min(159, (b_i_Cell_gtx_horizontal + 31)) + 1)) {
                            if (i_K_gtx_vertical >= b_i_K_gtx_vertical && i_K_gtx_vertical < (Min(69, (b_i_K_gtx_vertical + 7)) + 1)) {
                                {
                                    double __tlet_out;

                                    ///////////////////
                                    // Tasklet code (tlet_13_get_value__clone_5849bb60_1c88_11f1_a73a_1f4e8615a1c5)
                                    __tlet_out = 7.66;
                                    ///////////////////

                                    gtir_tmp_21_0 = __tlet_out;
                                }
                                {
                                    bool __tlet_out;

                                    ///////////////////
                                    // Tasklet code (tlet_25_get_value__clone_584e5fc6_1c88_11f1_a73a_1f4e8615a1c5)
                                    __tlet_out = false;
                                    ///////////////////

                                    gtir_tmp_43_0 = __tlet_out;
                                }
                                {
                                    double __tlet_out;

                                    ///////////////////
                                    // Tasklet code (tlet_9_get_value__clone_588d4420_1c88_11f1_a73a_1f4e8615a1c5)
                                    __tlet_out = 21.875;
                                    ///////////////////

                                    gtir_tmp_15_0 = __tlet_out;
                                }
                                {
                                    double __tlet_out;

                                    ///////////////////
                                    // Tasklet code (tlet_4_get_value__clone_5891d81e_1c88_11f1_a73a_1f4e8615a1c5)
                                    __tlet_out = 1e-15;
                                    ///////////////////

                                    gtir_tmp_8_0 = __tlet_out;
                                }
                                {
                                    double __tlet_out;

                                    ///////////////////
                                    // Tasklet code (tlet_8_get_value__clone_58964926_1c88_11f1_a73a_1f4e8615a1c5)
                                    __tlet_out = 610.78;
                                    ///////////////////

                                    gtir_tmp_14_0 = __tlet_out;
                                }
                                {
                                    double __tlet_out;

                                    ///////////////////
                                    // Tasklet code (tlet_18_get_value__clone_589a8f2c_1c88_11f1_a73a_1f4e8615a1c5)
                                    __tlet_out = 461.51;
                                    ///////////////////

                                    gtir_tmp_30_0 = __tlet_out;
                                }
                                {
                                    double __tlet_out;

                                    ///////////////////
                                    // Tasklet code (tlet_10_get_value__clone_589ee3ce_1c88_11f1_a73a_1f4e8615a1c5)
                                    __tlet_out = 273.15;
                                    ///////////////////

                                    gtir_tmp_16_0 = __tlet_out;
                                }
                                {
                                    double __tlet_out;

                                    ///////////////////
                                    // Tasklet code (tlet_6_get_value__clone_58a33406_1c88_11f1_a73a_1f4e8615a1c5)
                                    __tlet_out = 248.15;
                                    ///////////////////

                                    gtir_tmp_11_0 = __tlet_out;
                                }
                                {
                                    double __tlet_arg0_3 = gtir_tmp_14_0;
                                    double __tlet_arg0_4 = gtir_tmp_15_0;
                                    bool __tlet_arg1 = gtir_tmp_43_0;
                                    double __tlet_arg1_0 = gtir_tmp_8_0;
                                    double __tlet_arg1_0_0 = gtir_tmp_11_0;
                                    double __tlet_arg1_0_1 = gtir_tmp_30_0;
                                    double __tlet_arg1_0_2 = gtir_tmp_16_0;
                                    double __tlet_arg1_3 = gtir_tmp_21_0;
                                    double __tlet_arg0_0 = q_in_0[((__q_in_0_Cell_stride * ((- __q_in_0_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_0_K_stride * ((- __q_in_0_K_range_0) + i_K_gtx_vertical)))];
                                    double __tlet_arg0 = q_in_1[((__q_in_1_Cell_stride * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))];
                                    double __tlet_arg0_2 = q_in_2[((__q_in_2_Cell_stride * ((- __q_in_2_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_2_K_stride * ((- __q_in_2_K_range_0) + i_K_gtx_vertical)))];
                                    double __tlet_arg1_2 = q_in_3[((__q_in_3_Cell_stride * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))];
                                    double __tlet_arg0_1_0 = q_in_4[((__q_in_4_Cell_stride * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                                    double __tlet_arg0_0_0 = q_in_5[((__q_in_5_Cell_stride * ((- __q_in_5_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_5_K_stride * ((- __q_in_5_K_range_0) + i_K_gtx_vertical)))];
                                    double __tlet_arg0_1_1 = rho[(((- __rho_Cell_range_0) + (__rho_K_stride * ((- __rho_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                                    double __tlet_arg0_0_1 = te[(((- __te_Cell_range_0) + (__te_K_stride * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                                    double __tlet_arg0_1 = te[(((- __te_Cell_range_0) + (__te_K_stride * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                                    double __tlet_arg1_1 = te[(((- __te_Cell_range_0) + (__te_K_stride * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)];
                                    bool __tlet_result;

                                    ///////////////////
                                    // Tasklet code (tlet_11_minus_fused_tlet_12_multiplies_fused_tlet_14_minus_fused_tlet_15_divides_fused_tlet_19_multiplies_fused_tlet_20_multiplies_fused_tlet_16_exp_fused_tlet_17_multiplies_fused_tlet_21_divides_fused_tlet_0_maximum_fused_tlet_1_maximum_fused_tlet_2_maximum_fused_tlet_3_maximum_fused_tlet_22_greater_fused_tlet_7_less_fused_tlet_23_and__fused_tlet_5_greater_fused_tlet_24_or__fused_tlet_26_or_)
                                    __tlet_result = (((max(__tlet_arg0, max(__tlet_arg0_0_0, max(__tlet_arg0_1_0, max(__tlet_arg0_2, __tlet_arg1_2)))) > __tlet_arg1_0) || ((__tlet_arg0_1 < __tlet_arg1_0_0) && (__tlet_arg0_0 > ((__tlet_arg0_3 * dace::math::exp(((__tlet_arg0_4 * (__tlet_arg0_0_1 - __tlet_arg1_0_2)) / (__tlet_arg0_0_1 - __tlet_arg1_3)))) / ((__tlet_arg0_1_1 * __tlet_arg1_0_1) * __tlet_arg1_1))))) || __tlet_arg1);
                                    ///////////////////

                                    __map_fusion_gtir_tmp_45 = __tlet_result;
                                }
                                if_stmt_100_0_0_16(q_in_4[((__q_in_4_Cell_stride * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))], q_in_3[((__q_in_3_Cell_stride * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))], q_in_5[((__q_in_5_Cell_stride * ((- __q_in_5_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_5_K_stride * ((- __q_in_5_K_range_0) + i_K_gtx_vertical)))], q_in_1[((__q_in_1_Cell_stride * ((- __q_in_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_1_K_stride * ((- __q_in_1_K_range_0) + i_K_gtx_vertical)))], q_in_2[((__q_in_2_Cell_stride * ((- __q_in_2_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_2_K_stride * ((- __q_in_2_K_range_0) + i_K_gtx_vertical)))], q_in_0[((__q_in_0_Cell_stride * ((- __q_in_0_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_0_K_stride * ((- __q_in_0_K_range_0) + i_K_gtx_vertical)))], te[(((- __te_Cell_range_0) + (__te_K_stride * ((- __te_K_range_0) + i_K_gtx_vertical))) + i_Cell_gtx_horizontal)], __map_fusion_gtir_tmp_45, &p[0], &q_in_0[0], &q_in_1[0], &q_in_2[0], &q_in_3[0], &q_in_4[0], &q_in_5[0], &rho[0], &te[0], gtir_tmp_2003, gtir_tmp_2001, gtir_tmp_2005, gtir_tmp_1621, gtir_tmp_1674, gtir_tmp_1999, gtir_tmp_1619, __p_Cell_range_0, __p_K_range_0, __p_K_stride, __q_in_0_Cell_range_0, __q_in_0_Cell_stride, __q_in_0_K_range_0, __q_in_0_K_stride, __q_in_1_Cell_range_0, __q_in_1_Cell_stride, __q_in_1_K_range_0, __q_in_1_K_stride, __q_in_2_Cell_range_0, __q_in_2_Cell_stride, __q_in_2_K_range_0, __q_in_2_K_stride, __q_in_3_Cell_range_0, __q_in_3_Cell_stride, __q_in_3_K_range_0, __q_in_3_K_stride, __q_in_4_Cell_range_0, __q_in_4_Cell_stride, __q_in_4_K_range_0, __q_in_4_K_stride, __q_in_5_Cell_range_0, __q_in_5_Cell_stride, __q_in_5_K_range_0, __q_in_5_K_stride, __rho_Cell_range_0, __rho_K_range_0, __rho_K_stride, __te_Cell_range_0, __te_K_range_0, __te_K_stride, i_Cell_gtx_horizontal, i_K_gtx_vertical);

                                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                                &gtir_tmp_2003, gtir_tmp_2004 + (i_Cell_gtx_horizontal + (160 * i_K_gtx_vertical)), 1);

                                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                                &gtir_tmp_2001, gtir_tmp_2002 + (i_Cell_gtx_horizontal + (160 * i_K_gtx_vertical)), 1);

                                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                                &gtir_tmp_2005, gtir_tmp_2006 + (i_Cell_gtx_horizontal + (160 * i_K_gtx_vertical)), 1);

                                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                                &gtir_tmp_1621, q_out_1 + ((__q_out_1_Cell_stride * ((- __q_out_1_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_out_1_K_stride * ((- __q_out_1_K_range_0) + i_K_gtx_vertical))), 1);

                                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                                &gtir_tmp_1674, gtir_tmp_1675 + (i_Cell_gtx_horizontal + (160 * i_K_gtx_vertical)), 1);

                                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                                &gtir_tmp_1999, gtir_tmp_2000 + (i_Cell_gtx_horizontal + (160 * i_K_gtx_vertical)), 1);

                                dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                                &gtir_tmp_1619, q_out_0 + ((__q_out_0_Cell_stride * ((- __q_out_0_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_out_0_K_stride * ((- __q_out_0_K_range_0) + i_K_gtx_vertical))), 1);
                                {
                                    double __tlet_out;

                                    ///////////////////
                                    // Tasklet code (tlet_1170_get_value__clone_6212f13e_1c88_11f1_a73a_1f4e8615a1c5)
                                    __tlet_out = 1e-15;
                                    ///////////////////

                                    gtir_tmp_2010_0 = __tlet_out;
                                }
                                {
                                    double __tlet_arg1 = gtir_tmp_2010_0;
                                    double __tlet_arg0 = q_in_3[((__q_in_3_Cell_stride * ((- __q_in_3_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_3_K_stride * ((- __q_in_3_K_range_0) + i_K_gtx_vertical)))];
                                    bool __tlet_result;

                                    ///////////////////
                                    // Tasklet code (tlet_1171_greater)
                                    __tlet_result = (__tlet_arg0 > __tlet_arg1);
                                    ///////////////////

                                    gtir_tmp_2012[(i_Cell_gtx_horizontal + (160 * i_K_gtx_vertical))] = __tlet_result;
                                }
                                {
                                    double __tlet_out;

                                    ///////////////////
                                    // Tasklet code (tlet_1168_get_value__clone_62468cf6_1c88_11f1_a73a_1f4e8615a1c5)
                                    __tlet_out = 1e-15;
                                    ///////////////////

                                    gtir_tmp_2007_0 = __tlet_out;
                                }
                                {
                                    double __tlet_arg1 = gtir_tmp_2007_0;
                                    double __tlet_arg0 = q_in_2[((__q_in_2_Cell_stride * ((- __q_in_2_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_2_K_stride * ((- __q_in_2_K_range_0) + i_K_gtx_vertical)))];
                                    bool __tlet_result;

                                    ///////////////////
                                    // Tasklet code (tlet_1169_greater)
                                    __tlet_result = (__tlet_arg0 > __tlet_arg1);
                                    ///////////////////

                                    gtir_tmp_2009[(i_Cell_gtx_horizontal + (160 * i_K_gtx_vertical))] = __tlet_result;
                                }
                                {
                                    double __tlet_out;

                                    ///////////////////
                                    // Tasklet code (tlet_1174_get_value__clone_6253e8ec_1c88_11f1_a73a_1f4e8615a1c5)
                                    __tlet_out = 1e-15;
                                    ///////////////////

                                    gtir_tmp_2016_0 = __tlet_out;
                                }
                                {
                                    double __tlet_arg1 = gtir_tmp_2016_0;
                                    double __tlet_arg0 = q_in_5[((__q_in_5_Cell_stride * ((- __q_in_5_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_5_K_stride * ((- __q_in_5_K_range_0) + i_K_gtx_vertical)))];
                                    bool __tlet_result;

                                    ///////////////////
                                    // Tasklet code (tlet_1175_greater)
                                    __tlet_result = (__tlet_arg0 > __tlet_arg1);
                                    ///////////////////

                                    gtir_tmp_2018[(i_Cell_gtx_horizontal + (160 * i_K_gtx_vertical))] = __tlet_result;
                                }
                                {
                                    double __tlet_out;

                                    ///////////////////
                                    // Tasklet code (tlet_1172_get_value__clone_6291f95c_1c88_11f1_a73a_1f4e8615a1c5)
                                    __tlet_out = 1e-15;
                                    ///////////////////

                                    gtir_tmp_2013_0 = __tlet_out;
                                }
                                {
                                    double __tlet_arg1 = gtir_tmp_2013_0;
                                    double __tlet_arg0 = q_in_4[((__q_in_4_Cell_stride * ((- __q_in_4_Cell_range_0) + i_Cell_gtx_horizontal)) + (__q_in_4_K_stride * ((- __q_in_4_K_range_0) + i_K_gtx_vertical)))];
                                    bool __tlet_result;

                                    ///////////////////
                                    // Tasklet code (tlet_1173_greater)
                                    __tlet_result = (__tlet_arg0 > __tlet_arg1);
                                    ///////////////////

                                    gtir_tmp_2015[(i_Cell_gtx_horizontal + (160 * i_K_gtx_vertical))] = __tlet_result;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_map_0_fieldop_0_0_115(graupel_run_state_t *__state, double * __restrict__ gtir_tmp_1675, double * __restrict__ gtir_tmp_2000, double * __restrict__ gtir_tmp_2002, double * __restrict__ gtir_tmp_2004, double * __restrict__ gtir_tmp_2006, bool * __restrict__ gtir_tmp_2009, bool * __restrict__ gtir_tmp_2012, bool * __restrict__ gtir_tmp_2015, bool * __restrict__ gtir_tmp_2018, const double * __restrict__ p, const double * __restrict__ q_in_0, const double * __restrict__ q_in_1, const double * __restrict__ q_in_2, const double * __restrict__ q_in_3, const double * __restrict__ q_in_4, const double * __restrict__ q_in_5, double * __restrict__ q_out_0, double * __restrict__ q_out_1, const double * __restrict__ rho, const double * __restrict__ te, int __p_Cell_range_0, int __p_K_range_0, int __p_K_stride, int __q_in_0_Cell_range_0, int __q_in_0_Cell_stride, int __q_in_0_K_range_0, int __q_in_0_K_stride, int __q_in_1_Cell_range_0, int __q_in_1_Cell_stride, int __q_in_1_K_range_0, int __q_in_1_K_stride, int __q_in_2_Cell_range_0, int __q_in_2_Cell_stride, int __q_in_2_K_range_0, int __q_in_2_K_stride, int __q_in_3_Cell_range_0, int __q_in_3_Cell_stride, int __q_in_3_K_range_0, int __q_in_3_K_stride, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride, int __q_in_4_K_range_0, int __q_in_4_K_stride, int __q_in_5_Cell_range_0, int __q_in_5_Cell_stride, int __q_in_5_K_range_0, int __q_in_5_K_stride, int __q_out_0_Cell_range_0, int __q_out_0_Cell_stride, int __q_out_0_K_range_0, int __q_out_0_K_stride, int __q_out_1_Cell_range_0, int __q_out_1_Cell_stride, int __q_out_1_K_range_0, int __q_out_1_K_stride, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride);
void __dace_runkernel_map_0_fieldop_0_0_115(graupel_run_state_t *__state, double * __restrict__ gtir_tmp_1675, double * __restrict__ gtir_tmp_2000, double * __restrict__ gtir_tmp_2002, double * __restrict__ gtir_tmp_2004, double * __restrict__ gtir_tmp_2006, bool * __restrict__ gtir_tmp_2009, bool * __restrict__ gtir_tmp_2012, bool * __restrict__ gtir_tmp_2015, bool * __restrict__ gtir_tmp_2018, const double * __restrict__ p, const double * __restrict__ q_in_0, const double * __restrict__ q_in_1, const double * __restrict__ q_in_2, const double * __restrict__ q_in_3, const double * __restrict__ q_in_4, const double * __restrict__ q_in_5, double * __restrict__ q_out_0, double * __restrict__ q_out_1, const double * __restrict__ rho, const double * __restrict__ te, int __p_Cell_range_0, int __p_K_range_0, int __p_K_stride, int __q_in_0_Cell_range_0, int __q_in_0_Cell_stride, int __q_in_0_K_range_0, int __q_in_0_K_stride, int __q_in_1_Cell_range_0, int __q_in_1_Cell_stride, int __q_in_1_K_range_0, int __q_in_1_K_stride, int __q_in_2_Cell_range_0, int __q_in_2_Cell_stride, int __q_in_2_K_range_0, int __q_in_2_K_stride, int __q_in_3_Cell_range_0, int __q_in_3_Cell_stride, int __q_in_3_K_range_0, int __q_in_3_K_stride, int __q_in_4_Cell_range_0, int __q_in_4_Cell_stride, int __q_in_4_K_range_0, int __q_in_4_K_stride, int __q_in_5_Cell_range_0, int __q_in_5_Cell_stride, int __q_in_5_K_range_0, int __q_in_5_K_stride, int __q_out_0_Cell_range_0, int __q_out_0_Cell_stride, int __q_out_0_K_range_0, int __q_out_0_K_stride, int __q_out_1_Cell_range_0, int __q_out_1_Cell_stride, int __q_out_1_K_range_0, int __q_out_1_K_stride, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride, int __te_Cell_range_0, int __te_K_range_0, int __te_K_stride)
{

    void  *map_0_fieldop_0_0_115_args[] = { (void *)&gtir_tmp_1675, (void *)&gtir_tmp_2000, (void *)&gtir_tmp_2002, (void *)&gtir_tmp_2004, (void *)&gtir_tmp_2006, (void *)&gtir_tmp_2009, (void *)&gtir_tmp_2012, (void *)&gtir_tmp_2015, (void *)&gtir_tmp_2018, (void *)&p, (void *)&q_in_0, (void *)&q_in_1, (void *)&q_in_2, (void *)&q_in_3, (void *)&q_in_4, (void *)&q_in_5, (void *)&q_out_0, (void *)&q_out_1, (void *)&rho, (void *)&te, (void *)&__p_Cell_range_0, (void *)&__p_K_range_0, (void *)&__p_K_stride, (void *)&__q_in_0_Cell_range_0, (void *)&__q_in_0_Cell_stride, (void *)&__q_in_0_K_range_0, (void *)&__q_in_0_K_stride, (void *)&__q_in_1_Cell_range_0, (void *)&__q_in_1_Cell_stride, (void *)&__q_in_1_K_range_0, (void *)&__q_in_1_K_stride, (void *)&__q_in_2_Cell_range_0, (void *)&__q_in_2_Cell_stride, (void *)&__q_in_2_K_range_0, (void *)&__q_in_2_K_stride, (void *)&__q_in_3_Cell_range_0, (void *)&__q_in_3_Cell_stride, (void *)&__q_in_3_K_range_0, (void *)&__q_in_3_K_stride, (void *)&__q_in_4_Cell_range_0, (void *)&__q_in_4_Cell_stride, (void *)&__q_in_4_K_range_0, (void *)&__q_in_4_K_stride, (void *)&__q_in_5_Cell_range_0, (void *)&__q_in_5_Cell_stride, (void *)&__q_in_5_K_range_0, (void *)&__q_in_5_K_stride, (void *)&__q_out_0_Cell_range_0, (void *)&__q_out_0_Cell_stride, (void *)&__q_out_0_K_range_0, (void *)&__q_out_0_K_stride, (void *)&__q_out_1_Cell_range_0, (void *)&__q_out_1_Cell_stride, (void *)&__q_out_1_K_range_0, (void *)&__q_out_1_K_stride, (void *)&__rho_Cell_range_0, (void *)&__rho_K_range_0, (void *)&__rho_K_stride, (void *)&__te_Cell_range_0, (void *)&__te_K_range_0, (void *)&__te_K_stride };
    gpuError_t __err = cudaLaunchKernel((void*)map_0_fieldop_0_0_115, dim3(5, 9, 1), dim3(32, 8, 1), map_0_fieldop_0_0_115_args, 0, nullptr);
    DACE_KERNEL_LAUNCH_CHECK(__err, "map_0_fieldop_0_0_115", 5, 9, 1, 32, 8, 1);
}
__global__ void __maxnreg__(128)  map_689_fieldop_0_0_117(const double * __restrict__ dz, const double * __restrict__ gtir_tmp_1675, const double * __restrict__ gtir_tmp_2000, const double * __restrict__ gtir_tmp_2002, const double * __restrict__ gtir_tmp_2004, const double * __restrict__ gtir_tmp_2006, const bool * __restrict__ gtir_tmp_2009, const bool * __restrict__ gtir_tmp_2012, const bool * __restrict__ gtir_tmp_2015, const bool * __restrict__ gtir_tmp_2018, double * __restrict__ pflx, double * __restrict__ pg, double * __restrict__ pi, double * __restrict__ pr, double * __restrict__ pre, double * __restrict__ ps, const double * __restrict__ q_out_0, const double * __restrict__ q_out_1, double * __restrict__ q_out_2, double * __restrict__ q_out_3, double * __restrict__ q_out_4, double * __restrict__ q_out_5, const double * __restrict__ rho, double * __restrict__ t_out, int __dz_Cell_range_0, int __dz_K_range_0, int __dz_K_stride, int __pflx_Cell_range_0, int __pflx_K_range_0, int __pflx_K_stride, int __pg_Cell_range_0, int __pg_K_range_0, int __pg_K_stride, int __pi_Cell_range_0, int __pi_K_range_0, int __pi_K_stride, int __pr_Cell_range_0, int __pr_K_range_0, int __pr_K_stride, int __pre_Cell_range_0, int __pre_K_range_0, int __pre_K_stride, int __ps_Cell_range_0, int __ps_K_range_0, int __ps_K_stride, int __q_out_0_Cell_range_0, int __q_out_0_Cell_stride, int __q_out_0_K_range_0, int __q_out_0_K_stride, int __q_out_1_Cell_range_0, int __q_out_1_Cell_stride, int __q_out_1_K_range_0, int __q_out_1_K_stride, int __q_out_2_Cell_range_0, int __q_out_2_Cell_stride, int __q_out_2_K_range_0, int __q_out_2_K_stride, int __q_out_3_Cell_range_0, int __q_out_3_Cell_stride, int __q_out_3_K_range_0, int __q_out_3_K_stride, int __q_out_4_Cell_range_0, int __q_out_4_Cell_stride, int __q_out_4_K_range_0, int __q_out_4_K_stride, int __q_out_5_Cell_range_0, int __q_out_5_Cell_stride, int __q_out_5_K_range_0, int __q_out_5_K_stride, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride, int __t_out_Cell_range_0, int __t_out_K_range_0, int __t_out_K_stride) {
    {
        int b_i_Cell_gtx_horizontal = (64 * blockIdx.x);
        {
            {
                int i_Cell_gtx_horizontal = (threadIdx.x + b_i_Cell_gtx_horizontal);
                bool gtir_tmp_1679_0;
                bool gtir_tmp_1691_0;
                bool gtir_tmp_1683_0;
                double gtir_tmp_1685_0;
                double gtir_tmp_1682_0;
                double gtir_tmp_1686_0;
                double gtir_tmp_2019_0;
                double gtir_tmp_1693_0;
                double gtir_tmp_1680_0;
                bool gtir_tmp_1687_0;
                bool gtir_tmp_1694_0;
                double gtir_tmp_1678_0;
                double gtir_tmp_1688_0;
                double gtir_tmp_1677_0;
                double gtir_tmp_1689_0;
                double gtir_tmp_1676_0;
                double gtir_tmp_1681_0;
                double gtir_tmp_1690_0;
                double gtir_tmp_1684_0;
                double gtir_tmp_1695_0;
                if (i_Cell_gtx_horizontal >= b_i_Cell_gtx_horizontal && i_Cell_gtx_horizontal < (Min(159, (b_i_Cell_gtx_horizontal + 63)) + 1)) {
                    {
                        bool __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_909_get_value__clone_6217ecd4_1c88_11f1_a73a_1f4e8615a1c5)
                        __tlet_out = false;
                        ///////////////////

                        gtir_tmp_1679_0 = __tlet_out;
                    }
                    {
                        bool __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_921_get_value__clone_621cc77c_1c88_11f1_a73a_1f4e8615a1c5)
                        __tlet_out = false;
                        ///////////////////

                        gtir_tmp_1691_0 = __tlet_out;
                    }
                    {
                        bool __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_913_get_value__clone_62219720_1c88_11f1_a73a_1f4e8615a1c5)
                        __tlet_out = false;
                        ///////////////////

                        gtir_tmp_1683_0 = __tlet_out;
                    }
                    {
                        double __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_915_get_value__clone_6225884e_1c88_11f1_a73a_1f4e8615a1c5)
                        __tlet_out = 0.0;
                        ///////////////////

                        gtir_tmp_1685_0 = __tlet_out;
                    }
                    {
                        double __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_912_get_value__clone_62296f04_1c88_11f1_a73a_1f4e8615a1c5)
                        __tlet_out = 0.0;
                        ///////////////////

                        gtir_tmp_1682_0 = __tlet_out;
                    }
                    {
                        double __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_916_get_value__clone_622d685c_1c88_11f1_a73a_1f4e8615a1c5)
                        __tlet_out = 0.0;
                        ///////////////////

                        gtir_tmp_1686_0 = __tlet_out;
                    }
                    {
                        double __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_1176_get_value__clone_62315156_1c88_11f1_a73a_1f4e8615a1c5)
                        __tlet_out = 30.0;
                        ///////////////////

                        gtir_tmp_2019_0 = __tlet_out;
                    }
                    {
                        double __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_923_get_value__clone_62358e24_1c88_11f1_a73a_1f4e8615a1c5)
                        __tlet_out = 0.0;
                        ///////////////////

                        gtir_tmp_1693_0 = __tlet_out;
                    }
                    {
                        double __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_910_get_value__clone_6239a2a2_1c88_11f1_a73a_1f4e8615a1c5)
                        __tlet_out = 0.0;
                        ///////////////////

                        gtir_tmp_1680_0 = __tlet_out;
                    }
                    {
                        bool __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_917_get_value__clone_623db7de_1c88_11f1_a73a_1f4e8615a1c5)
                        __tlet_out = false;
                        ///////////////////

                        gtir_tmp_1687_0 = __tlet_out;
                    }
                    {
                        bool __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_924_get_value__clone_62421360_1c88_11f1_a73a_1f4e8615a1c5)
                        __tlet_out = false;
                        ///////////////////

                        gtir_tmp_1694_0 = __tlet_out;
                    }
                    {
                        double __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_908_get_value__clone_624b123a_1c88_11f1_a73a_1f4e8615a1c5)
                        __tlet_out = 0.0;
                        ///////////////////

                        gtir_tmp_1678_0 = __tlet_out;
                    }
                    {
                        double __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_918_get_value__clone_624f8392_1c88_11f1_a73a_1f4e8615a1c5)
                        __tlet_out = 0.0;
                        ///////////////////

                        gtir_tmp_1688_0 = __tlet_out;
                    }
                    {
                        double __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_907_get_value__clone_628d5348_1c88_11f1_a73a_1f4e8615a1c5)
                        __tlet_out = 0.0;
                        ///////////////////

                        gtir_tmp_1677_0 = __tlet_out;
                    }
                    {
                        double __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_919_get_value__clone_6296ae70_1c88_11f1_a73a_1f4e8615a1c5)
                        __tlet_out = 0.0;
                        ///////////////////

                        gtir_tmp_1689_0 = __tlet_out;
                    }
                    {
                        double __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_906_get_value__clone_629b0e52_1c88_11f1_a73a_1f4e8615a1c5)
                        __tlet_out = 0.0;
                        ///////////////////

                        gtir_tmp_1676_0 = __tlet_out;
                    }
                    {
                        double __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_911_get_value__clone_629fd34c_1c88_11f1_a73a_1f4e8615a1c5)
                        __tlet_out = 0.0;
                        ///////////////////

                        gtir_tmp_1681_0 = __tlet_out;
                    }
                    {
                        double __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_920_get_value__clone_62a45c50_1c88_11f1_a73a_1f4e8615a1c5)
                        __tlet_out = 0.0;
                        ///////////////////

                        gtir_tmp_1690_0 = __tlet_out;
                    }
                    {
                        double __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_914_get_value__clone_62a8c31c_1c88_11f1_a73a_1f4e8615a1c5)
                        __tlet_out = 0.0;
                        ///////////////////

                        gtir_tmp_1684_0 = __tlet_out;
                    }
                    {
                        double __tlet_out;

                        ///////////////////
                        // Tasklet code (tlet_925_get_value__clone_62ad0670_1c88_11f1_a73a_1f4e8615a1c5)
                        __tlet_out = 0.0;
                        ///////////////////

                        gtir_tmp_1695_0 = __tlet_out;
                    }
                    scan_0_0_0_27(&q_out_0[(((- __q_out_0_Cell_range_0) * __q_out_0_Cell_stride) - (__q_out_0_K_range_0 * __q_out_0_K_stride))], &q_out_1[(((- __q_out_1_Cell_range_0) * __q_out_1_Cell_stride) - (__q_out_1_K_range_0 * __q_out_1_K_stride))], &gtir_tmp_2000[0], &gtir_tmp_2002[0], &gtir_tmp_2004[0], &gtir_tmp_2006[0], gtir_tmp_1676_0, gtir_tmp_1677_0, gtir_tmp_1678_0, gtir_tmp_1679_0, gtir_tmp_1680_0, gtir_tmp_1681_0, gtir_tmp_1682_0, gtir_tmp_1683_0, gtir_tmp_1684_0, gtir_tmp_1685_0, gtir_tmp_1686_0, gtir_tmp_1687_0, gtir_tmp_1688_0, gtir_tmp_1689_0, gtir_tmp_1690_0, gtir_tmp_1691_0, gtir_tmp_1693_0, gtir_tmp_1694_0, gtir_tmp_1695_0, gtir_tmp_2019_0, &dz[0], &gtir_tmp_1675[0], &gtir_tmp_1675[0], &gtir_tmp_2018[0], &gtir_tmp_2015[0], &gtir_tmp_2009[0], &gtir_tmp_2012[0], &rho[0], &gtir_tmp_1675[0], &q_out_2[((__q_out_2_Cell_stride * ((- __q_out_2_Cell_range_0) + i_Cell_gtx_horizontal)) - (__q_out_2_K_range_0 * __q_out_2_K_stride))], &pr[(((- __pr_Cell_range_0) + (__pr_K_stride * (69 - __pr_K_range_0))) + i_Cell_gtx_horizontal)], &q_out_3[((__q_out_3_Cell_stride * ((- __q_out_3_Cell_range_0) + i_Cell_gtx_horizontal)) - (__q_out_3_K_range_0 * __q_out_3_K_stride))], &ps[(((- __ps_Cell_range_0) + (__ps_K_stride * (69 - __ps_K_range_0))) + i_Cell_gtx_horizontal)], &q_out_4[((__q_out_4_Cell_stride * ((- __q_out_4_Cell_range_0) + i_Cell_gtx_horizontal)) - (__q_out_4_K_range_0 * __q_out_4_K_stride))], &pi[(((- __pi_Cell_range_0) + (__pi_K_stride * (69 - __pi_K_range_0))) + i_Cell_gtx_horizontal)], &q_out_5[((__q_out_5_Cell_stride * ((- __q_out_5_Cell_range_0) + i_Cell_gtx_horizontal)) - (__q_out_5_K_range_0 * __q_out_5_K_stride))], &pg[(((- __pg_Cell_range_0) + (__pg_K_stride * (69 - __pg_K_range_0))) + i_Cell_gtx_horizontal)], &t_out[(((- __t_out_Cell_range_0) - (__t_out_K_range_0 * __t_out_K_stride)) + i_Cell_gtx_horizontal)], &pre[(((- __pre_Cell_range_0) + (__pre_K_stride * (69 - __pre_K_range_0))) + i_Cell_gtx_horizontal)], &pflx[(((- __pflx_Cell_range_0) - (__pflx_K_range_0 * __pflx_K_stride)) + i_Cell_gtx_horizontal)], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, __dz_K_stride, __dz_Cell_range_0, __dz_K_range_0, 0, 0, 0, 0, 0, 0, 0, 0, __pflx_K_stride, __q_out_0_Cell_stride, __q_out_0_K_stride, __q_out_1_Cell_stride, __q_out_1_K_stride, __q_out_2_K_stride, __q_out_3_K_stride, __q_out_4_K_stride, __q_out_5_K_stride, __rho_K_stride, __rho_Cell_range_0, __rho_K_range_0, 0, 0, 0, 0, __t_out_K_stride, i_Cell_gtx_horizontal);
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_map_689_fieldop_0_0_117(graupel_run_state_t *__state, const double * __restrict__ dz, const double * __restrict__ gtir_tmp_1675, const double * __restrict__ gtir_tmp_2000, const double * __restrict__ gtir_tmp_2002, const double * __restrict__ gtir_tmp_2004, const double * __restrict__ gtir_tmp_2006, const bool * __restrict__ gtir_tmp_2009, const bool * __restrict__ gtir_tmp_2012, const bool * __restrict__ gtir_tmp_2015, const bool * __restrict__ gtir_tmp_2018, double * __restrict__ pflx, double * __restrict__ pg, double * __restrict__ pi, double * __restrict__ pr, double * __restrict__ pre, double * __restrict__ ps, const double * __restrict__ q_out_0, const double * __restrict__ q_out_1, double * __restrict__ q_out_2, double * __restrict__ q_out_3, double * __restrict__ q_out_4, double * __restrict__ q_out_5, const double * __restrict__ rho, double * __restrict__ t_out, int __dz_Cell_range_0, int __dz_K_range_0, int __dz_K_stride, int __pflx_Cell_range_0, int __pflx_K_range_0, int __pflx_K_stride, int __pg_Cell_range_0, int __pg_K_range_0, int __pg_K_stride, int __pi_Cell_range_0, int __pi_K_range_0, int __pi_K_stride, int __pr_Cell_range_0, int __pr_K_range_0, int __pr_K_stride, int __pre_Cell_range_0, int __pre_K_range_0, int __pre_K_stride, int __ps_Cell_range_0, int __ps_K_range_0, int __ps_K_stride, int __q_out_0_Cell_range_0, int __q_out_0_Cell_stride, int __q_out_0_K_range_0, int __q_out_0_K_stride, int __q_out_1_Cell_range_0, int __q_out_1_Cell_stride, int __q_out_1_K_range_0, int __q_out_1_K_stride, int __q_out_2_Cell_range_0, int __q_out_2_Cell_stride, int __q_out_2_K_range_0, int __q_out_2_K_stride, int __q_out_3_Cell_range_0, int __q_out_3_Cell_stride, int __q_out_3_K_range_0, int __q_out_3_K_stride, int __q_out_4_Cell_range_0, int __q_out_4_Cell_stride, int __q_out_4_K_range_0, int __q_out_4_K_stride, int __q_out_5_Cell_range_0, int __q_out_5_Cell_stride, int __q_out_5_K_range_0, int __q_out_5_K_stride, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride, int __t_out_Cell_range_0, int __t_out_K_range_0, int __t_out_K_stride);
void __dace_runkernel_map_689_fieldop_0_0_117(graupel_run_state_t *__state, const double * __restrict__ dz, const double * __restrict__ gtir_tmp_1675, const double * __restrict__ gtir_tmp_2000, const double * __restrict__ gtir_tmp_2002, const double * __restrict__ gtir_tmp_2004, const double * __restrict__ gtir_tmp_2006, const bool * __restrict__ gtir_tmp_2009, const bool * __restrict__ gtir_tmp_2012, const bool * __restrict__ gtir_tmp_2015, const bool * __restrict__ gtir_tmp_2018, double * __restrict__ pflx, double * __restrict__ pg, double * __restrict__ pi, double * __restrict__ pr, double * __restrict__ pre, double * __restrict__ ps, const double * __restrict__ q_out_0, const double * __restrict__ q_out_1, double * __restrict__ q_out_2, double * __restrict__ q_out_3, double * __restrict__ q_out_4, double * __restrict__ q_out_5, const double * __restrict__ rho, double * __restrict__ t_out, int __dz_Cell_range_0, int __dz_K_range_0, int __dz_K_stride, int __pflx_Cell_range_0, int __pflx_K_range_0, int __pflx_K_stride, int __pg_Cell_range_0, int __pg_K_range_0, int __pg_K_stride, int __pi_Cell_range_0, int __pi_K_range_0, int __pi_K_stride, int __pr_Cell_range_0, int __pr_K_range_0, int __pr_K_stride, int __pre_Cell_range_0, int __pre_K_range_0, int __pre_K_stride, int __ps_Cell_range_0, int __ps_K_range_0, int __ps_K_stride, int __q_out_0_Cell_range_0, int __q_out_0_Cell_stride, int __q_out_0_K_range_0, int __q_out_0_K_stride, int __q_out_1_Cell_range_0, int __q_out_1_Cell_stride, int __q_out_1_K_range_0, int __q_out_1_K_stride, int __q_out_2_Cell_range_0, int __q_out_2_Cell_stride, int __q_out_2_K_range_0, int __q_out_2_K_stride, int __q_out_3_Cell_range_0, int __q_out_3_Cell_stride, int __q_out_3_K_range_0, int __q_out_3_K_stride, int __q_out_4_Cell_range_0, int __q_out_4_Cell_stride, int __q_out_4_K_range_0, int __q_out_4_K_stride, int __q_out_5_Cell_range_0, int __q_out_5_Cell_stride, int __q_out_5_K_range_0, int __q_out_5_K_stride, int __rho_Cell_range_0, int __rho_K_range_0, int __rho_K_stride, int __t_out_Cell_range_0, int __t_out_K_range_0, int __t_out_K_stride)
{

    void  *map_689_fieldop_0_0_117_args[] = { (void *)&dz, (void *)&gtir_tmp_1675, (void *)&gtir_tmp_2000, (void *)&gtir_tmp_2002, (void *)&gtir_tmp_2004, (void *)&gtir_tmp_2006, (void *)&gtir_tmp_2009, (void *)&gtir_tmp_2012, (void *)&gtir_tmp_2015, (void *)&gtir_tmp_2018, (void *)&pflx, (void *)&pg, (void *)&pi, (void *)&pr, (void *)&pre, (void *)&ps, (void *)&q_out_0, (void *)&q_out_1, (void *)&q_out_2, (void *)&q_out_3, (void *)&q_out_4, (void *)&q_out_5, (void *)&rho, (void *)&t_out, (void *)&__dz_Cell_range_0, (void *)&__dz_K_range_0, (void *)&__dz_K_stride, (void *)&__pflx_Cell_range_0, (void *)&__pflx_K_range_0, (void *)&__pflx_K_stride, (void *)&__pg_Cell_range_0, (void *)&__pg_K_range_0, (void *)&__pg_K_stride, (void *)&__pi_Cell_range_0, (void *)&__pi_K_range_0, (void *)&__pi_K_stride, (void *)&__pr_Cell_range_0, (void *)&__pr_K_range_0, (void *)&__pr_K_stride, (void *)&__pre_Cell_range_0, (void *)&__pre_K_range_0, (void *)&__pre_K_stride, (void *)&__ps_Cell_range_0, (void *)&__ps_K_range_0, (void *)&__ps_K_stride, (void *)&__q_out_0_Cell_range_0, (void *)&__q_out_0_Cell_stride, (void *)&__q_out_0_K_range_0, (void *)&__q_out_0_K_stride, (void *)&__q_out_1_Cell_range_0, (void *)&__q_out_1_Cell_stride, (void *)&__q_out_1_K_range_0, (void *)&__q_out_1_K_stride, (void *)&__q_out_2_Cell_range_0, (void *)&__q_out_2_Cell_stride, (void *)&__q_out_2_K_range_0, (void *)&__q_out_2_K_stride, (void *)&__q_out_3_Cell_range_0, (void *)&__q_out_3_Cell_stride, (void *)&__q_out_3_K_range_0, (void *)&__q_out_3_K_stride, (void *)&__q_out_4_Cell_range_0, (void *)&__q_out_4_Cell_stride, (void *)&__q_out_4_K_range_0, (void *)&__q_out_4_K_stride, (void *)&__q_out_5_Cell_range_0, (void *)&__q_out_5_Cell_stride, (void *)&__q_out_5_K_range_0, (void *)&__q_out_5_K_stride, (void *)&__rho_Cell_range_0, (void *)&__rho_K_range_0, (void *)&__rho_K_stride, (void *)&__t_out_Cell_range_0, (void *)&__t_out_K_range_0, (void *)&__t_out_K_stride };
    gpuError_t __err = cudaLaunchKernel((void*)map_689_fieldop_0_0_117, dim3(3, 1, 1), dim3(64, 1, 1), map_689_fieldop_0_0_117_args, 0, nullptr);
    DACE_KERNEL_LAUNCH_CHECK(__err, "map_689_fieldop_0_0_117", 3, 1, 1, 64, 1, 1);
}

