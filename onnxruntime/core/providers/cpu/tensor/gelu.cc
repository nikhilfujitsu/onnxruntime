// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"

#include "core/platform/threadpool.h"
#include <unsupported/Eigen/SpecialFunctions>
#include "core/providers/cpu/element_wise_ranged_transform.h"
#include "core/providers/cpu/tensor/gelu.h"

using onnxruntime::narrow;
using namespace onnxruntime::common;

namespace onnxruntime {

// May revisit the implementations to support inplace computation, if needed.

// ONNX_CPU_OPERATOR_KERNEL(
//     Gelu,
//     20,
//     KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
//     Gelu<float>);
#define ADD_TYPED_GELU_OP(data_type)                                     \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                        \
      Gelu,                                                              \
      20,                                                                \
      data_type,                                                         \
      KernelDefBuilder()                                                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      Gelu<data_type>)
ADD_TYPED_GELU_OP(float);
ADD_TYPED_GELU_OP(MLFloat16);

#ifndef DISABLE_CONTRIB_OPS
namespace contrib {
ONNX_OPERATOR_KERNEL_EX(
    Gelu,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gelu<float>);
}
#endif

template <typename T>
Status Gelu<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const T* input_data = input->Data<T>();

  Tensor* output = context->Output(0, input->Shape());
  T* output_data = output->MutableData<T>();

  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  int64_t elem_count = input->Shape().Size();
  constexpr int64_t length_per_task = 4096;  // this number comes from FastGelu.
  int64_t task_count = (elem_count + length_per_task - 1) / length_per_task;

  if (approximation_algorithm_ == "tanh") {
    // FastGelu allows optional bias. Here we split input data into chunks. Each chunk
    // has N elements (except the last chunk), and use thread pool to parallel chunks.
    // N = 4096 is selected based on performance test results on input shape 1x128x768.
    // FastGelu uses approximation for Gelu. The formula is 0.5 * (1 + Tanh(x * (C * x * x + B))) * x.
    static constexpr float B = 0.7978845608028654f;    // sqrt(2.0 / M_PI)
    static constexpr float C = 0.035677408136300125f;  // 0.044715 * sqrt(2.0 / M_PI)

    concurrency::ThreadPool::TryBatchParallelFor(
        tp, static_cast<int32_t>(task_count),
        [&](ptrdiff_t task_idx) {
          const auto start = task_idx * length_per_task;
          const T* p_input = input_data + start;
          T* p_output = output_data + start;
          int64_t count = std::min(length_per_task, elem_count - start);

          for (int64_t i = 0; i < count; i++) {
            T value = p_input[i];
            p_output[i] = value * (static_cast<T>(C) * value * value + static_cast<T>(B));
          }

          MlasComputeTanh(p_output, p_output, narrow<size_t>(count));

          for (int64_t i = 0; i < count; i++) {
            p_output[i] = 0.5f * p_input[i] * (p_output[i] + 1.0f);
          }
        },
        0);
    return Status::OK();
  } else if (approximation_algorithm_ == "none") {
    concurrency::ThreadPool::TryBatchParallelFor(
        tp, static_cast<int32_t>(task_count),
        [&](ptrdiff_t task_idx) {
          const auto start = task_idx * length_per_task;
          const T* p_input = input_data + start;
          T* p_output = output_data + start;
          int64_t count = std::min(length_per_task, elem_count - start);

          for (int64_t i = 0; i < count; i++) {
            T value = p_input[i];
            p_output[i] = value * static_cast<T>(M_SQRT1_2);
          }

          MlasComputeErf(p_output, p_output, narrow<size_t>(count));

          for (int64_t i = 0; i < count; i++) {
            p_output[i] = 0.5f * p_input[i] * (p_output[i] + 1.0f);
          }
                  },
        0);
    return Status::OK();
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported approximation_algorithm: ", approximation_algorithm_);
}
#if defined(MLAS_USE_SVE)
#include <arm_sve.h>
void ComputeGeluFp16_SVE(const MLFloat16* input, MLFloat16* output, MLFloat16* temp, int64_t count, const std::string& algo) {
  const svfloat16_t v_half = svdup_f16(0.5f);
  const svfloat16_t v_one = svdup_f16(1.0f);
  const svfloat16_t v_sqrt1_2 = svdup_f16(static_cast<float>(M_SQRT1_2));
  const svfloat16_t v_B = svdup_f16(0.7979f);
  const svfloat16_t v_C = svdup_f16(0.03568f);
  if (algo == "tanh") {
    int64_t i = 0;
    while (i < (count)) {
      svbool_t pg = svwhilelt_b16(i, count);
      svfloat16_t v_x = svld1(pg, reinterpret_cast<const __fp16*>(input + i));
      svfloat16_t v_x2 = svmul_f16_m(pg, v_x, v_x);
      svfloat16_t v_inner = svadd_f16_m(pg, v_B, svmul_f16_m(pg, v_C, v_x2));
      svfloat16_t v_tanh_arg = svmul_f16_m(pg, v_x, v_inner);
      v_tanh_arg = svmax_f16_m(pg, svdup_f16(-5.0f), svmin_f16_m(pg, v_tanh_arg, svdup_f16(5.0f)));
      svst1(pg, reinterpret_cast<__fp16*>(temp + i), v_tanh_arg);
      i += svcnth();
    }
    MlasComputeTanh<MLAS_FP16>(reinterpret_cast<const MLAS_FP16*>(temp),
                               reinterpret_cast<MLAS_FP16*>(temp),
                               count);
    int64_t j = 0;
    while (j < (count)) {
      svbool_t pg = svwhilelt_b16(j, count);
      svfloat16_t v_x = svld1(pg, reinterpret_cast<const __fp16*>(input + j));
      svfloat16_t v_tanh = svld1(pg, reinterpret_cast<const __fp16*>(temp + j));
      svfloat16_t v_result = svmul_f16_m(pg, v_half, svmul_f16_m(pg, v_x, svadd_f16_m(pg, v_one, v_tanh)));
      svst1(pg, reinterpret_cast<__fp16*>(output + j), v_result);
      j += svcnth();
    }
  } else if (algo == "none") {
    int64_t i = 0;
    while (i < (count)) {
      svbool_t pg = svwhilelt_b16(i, count);
      svfloat16_t v_x = svld1(pg, reinterpret_cast<const __fp16*>(input + i));
      svfloat16_t v_scaled = svmul_f16_m(pg, v_x, v_sqrt1_2);
      svst1(pg, reinterpret_cast<__fp16*>(temp + i), v_scaled);
      i += svcnth();
    }
    MlasSveErfKernelFp16(reinterpret_cast<const _mlas_fp16_*>(temp),
                         reinterpret_cast<_mlas_fp16_*>(temp),
                         count);
    int64_t j = 0;
    while (j < (count)) {
      svbool_t pg = svwhilelt_b16(j, count);
      svfloat16_t v_x = svld1(pg, reinterpret_cast<const __fp16*>(input + j));
      svfloat16_t v_erf = svld1(pg, reinterpret_cast<const __fp16*>(temp + j));
      svfloat16_t v_result = svmul_f16_m(pg, v_half, svmul_f16_m(pg, v_x, svadd_f16_m(pg, v_one, v_erf)));
      svst1(pg, reinterpret_cast<__fp16*>(output + j), v_result);
      j += svcnth();
    }
  }
}
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
void ComputeGeluFp16_NEON(const MLFloat16* input, MLFloat16* output, MLFloat16* temp, int64_t count, const std::string& algo) {
  const float16x8_t v_half = vdupq_n_f16(0.5f);
  const float16x8_t v_one = vdupq_n_f16(1.0f);
  const float16x8_t v_sqrt1_2 = vdupq_n_f16(static_cast<float>(M_SQRT1_2));
  const float16x8_t v_B = vdupq_n_f16(0.7979f);
  const float16x8_t v_C = vdupq_n_f16(0.03568f);
  for (int64_t i = 0; i < count; i += 8) {
    float16x8_t x = vld1q_f16(reinterpret_cast<const __fp16*>(input + i));
    if (algo == "tanh") {
      float16x8_t x2 = vmulq_f16(x, x);
      float16x8_t inner = vfmaq_f16(v_B, v_C, x2);
      float16x8_t tanh_arg = vmulq_f16(x, inner);
      tanh_arg = vmaxq_f16(vdupq_n_f16(-5.0f), vminq_f16(tanh_arg, vdupq_n_f16(5.0f)));
      vst1q_f16(reinterpret_cast<__fp16*>(temp + i), tanh_arg);
    } else if (algo == "none") {
      float16x8_t scaled = vmulq_f16(x, v_sqrt1_2);
      vst1q_f16(reinterpret_cast<__fp16*>(temp + i), scaled);
    }
  }
  if (algo == "tanh") {
    MlasComputeTanh<MLAS_FP16>(reinterpret_cast<const MLAS_FP16*>(temp),
                               reinterpret_cast<MLAS_FP16*>(temp),
                               count);
  } else if (algo == "none") {
    MlasNeonErfKernelFp16(reinterpret_cast<const _mlas_fp16_*>(temp),
                          reinterpret_cast<_mlas_fp16_*>(temp),
                          count);
  }
  for (int64_t j = 0; j < count; j += 8) {
    float16x8_t x = vld1q_f16(reinterpret_cast<const __fp16*>(input + j));
    float16x8_t t = vld1q_f16(reinterpret_cast<const __fp16*>(temp + j));
    float16x8_t result = vmulq_f16(v_half, vmulq_f16(x, vaddq_f16(v_one, t)));
    vst1q_f16(reinterpret_cast<__fp16*>(output + j), result);
  }
}
#endif
template <>
Status Gelu<MLFloat16>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const MLFloat16* input_data = input->Data<MLFloat16>();
  Tensor* output = context->Output(0, input->Shape());
  MLFloat16* output_data = output->MutableData<MLFloat16>();
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  int64_t elem_count = input->Shape().Size();
  constexpr int64_t length_per_task = 4096;
  int64_t task_count = (elem_count + length_per_task - 1) / length_per_task;
  if (approximation_algorithm_ != "tanh" && approximation_algorithm_ != "none") {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported approximation_algorithm: ", approximation_algorithm_);
  }
  // Alignment and buffer size for aligned_alloc
  constexpr size_t alignment = 64;  // 64-byte alignment for SIMD
  size_t buffer_size = elem_count * sizeof(MLFloat16);
  size_t aligned_size = ((buffer_size + alignment - 1) / alignment) * alignment;
  MLFloat16* temp_fp16_aligned = reinterpret_cast<MLFloat16*>(std::aligned_alloc(alignment, aligned_size));
  if (temp_fp16_aligned == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to allocate aligned temporary buffer.");
  }
  concurrency::ThreadPool::TryBatchParallelFor(
      tp,
      static_cast<int32_t>(task_count),
      [&](ptrdiff_t task_idx) {
        const auto start = task_idx * length_per_task;
        const MLFloat16* p_input = input_data + start;
        MLFloat16* p_output = output_data + start;
        
        int64_t count = std::min(length_per_task, elem_count - start);
#if defined(MLAS_USE_SVE) || defined(MLAS_NEON_INTRINSICS)
MLFloat16* p_temp = temp_fp16_aligned + start;
      if(MLAS_CPUIDINFO::GetCPUIDInfo().HasArmSve())
       {
        ComputeGeluFp16_SVE(p_input, p_output, p_temp, count, approximation_algorithm_);
      }
        else
        {
        ComputeGeluFp16_NEON(p_input, p_output, p_temp, count, approximation_algorithm_);
       }
#else
        // Fallback scalar implementation (optional)
        for (int64_t i = 0; i < count; ++i) {
          float x = static_cast<float>(p_input[i]);
          float gelu_val;
          if (approximation_algorithm_ == "tanh") {
            // GELU approx with tanh
            const float B = 0.7978845608f;  // sqrt(2/pi)
            const float C = 0.044715f * B;
            float tanh_arg = x * (B + C * x * x);
            float tanh_res = std::tanh(tanh_arg);
            gelu_val = 0.5f * x * (1 + tanh_res);
          } else {  // "none"
            gelu_val = 0.5f * x * (1 + std::erf(x * static_cast<float>(M_SQRT1_2)));
          }
          p_output[i] = MLFloat16(gelu_val);
        }
#endif
        },
        0);
          std::free(temp_fp16_aligned);
    return Status::OK();
  }
  /*template <>
Status Gelu<MLFloat16>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const MLFloat16* input_data = input->Data<MLFloat16>();
  Tensor* output = context->Output(0, input->Shape());
  MLFloat16* output_data = output->MutableData<MLFloat16>();
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  int64_t elem_count = input->Shape().Size();
  constexpr int64_t length_per_task = 4096;
  int64_t task_count = (elem_count + length_per_task - 1) / length_per_task;
  if (approximation_algorithm_ != "tanh" && approximation_algorithm_ != "none") {
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported approximation_algorithm: ", approximation_algorithm_);
}
  // Constants for "tanh" approx
  //static constexpr float B = 0.7978845608028654f;   // sqrt(2/pi)
  //static constexpr float C = 0.035677408136300125f; // 0.044715 * sqrt(2/pi)
  const svfloat16_t v_half = svdup_f16(0.5f);
  const svfloat16_t v_one = svdup_f16(1.0f);
  const svfloat16_t v_sqrt1_2 = svdup_f16(static_cast<float>(M_SQRT1_2)); 
  // Alignment and buffer size for aligned_alloc
  constexpr size_t alignment = 64;  // 64-byte alignment for SIMD
  size_t buffer_size = elem_count * sizeof(MLFloat16);
  size_t aligned_size = ((buffer_size + alignment - 1) / alignment) * alignment;
  MLFloat16* temp_fp16_aligned = reinterpret_cast<MLFloat16*>(std::aligned_alloc(alignment, aligned_size));
  if (temp_fp16_aligned == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to allocate aligned temporary buffer.");
  }
  concurrency::ThreadPool::TryBatchParallelFor(
      tp,
      static_cast<int32_t>(task_count),
      [&](ptrdiff_t task_idx) {
        const auto start = task_idx * length_per_task;
        const MLFloat16* p_input = input_data + start;
        MLFloat16* p_output = output_data + start;
        MLFloat16* p_temp = temp_fp16_aligned + start;
        int64_t count = std::min(length_per_task, elem_count - start);
       if (approximation_algorithm_ == "tanh") {
  // Constants (precomputed in FP16)
  const svfloat16_t v_B = svdup_f16(0.7979f);     // sqrt(2/pi)
  const svfloat16_t v_C = svdup_f16(0.03568f);    // 0.044715 * sqrt(2/pi)
 // const svfloat16_t v_half = svdup_f16(0.5f);
 // const svfloat16_t v_one = svdup_f16(1.0f);
  size_t i = 0;
  while (i < static_cast<size_t>(count)) {
    svbool_t pg = svwhilelt_b16(i, static_cast<size_t>(count));
    // Load x
    svfloat16_t v_x = svld1(pg, reinterpret_cast<const __fp16*>(p_input + i));
    // Compute x^2 and x^3
    svfloat16_t v_x2 = svmul_f16_m(pg, v_x, v_x);
    //svfloat16_t v_x3 = svmul_f16_m(pg, v_x2, v_x); // x^3
    // Compute inner: tanh_arg = x * (B + C * x^2)
    svfloat16_t v_cx2 = svmul_f16_m(pg, v_C, v_x2);
    svfloat16_t v_inner = svadd_f16_m(pg, v_B, v_cx2);
    svfloat16_t v_tanh_arg = svmul_f16_m(pg, v_x, v_inner);
    // Optional: clamp tanh_arg to avoid overflow in tanh
    svfloat16_t v_clamp_hi = svdup_f16(5.0f);
    svfloat16_t v_clamp_lo = svdup_f16(-5.0f);
    v_tanh_arg = svmin_f16_m(pg, v_tanh_arg, v_clamp_hi);
    v_tanh_arg = svmax_f16_m(pg, v_tanh_arg, v_clamp_lo);
    // Store tanh_arg to temp buffer
    svst1(pg, reinterpret_cast<__fp16*>(p_temp + i), v_tanh_arg);
    i += svcnth();
  }
  // Step 2: tanh in-place on temp buffer
  MlasComputeTanh<MLAS_FP16>(reinterpret_cast<const MLAS_FP16*>(p_temp),
                         reinterpret_cast<MLAS_FP16*>(p_temp),
                         count);
  // Step 3: GELU = 0.5 * x * (1 + tanh)
  size_t j = 0;
  while (j < static_cast<size_t>(count)) {
    svbool_t pg = svwhilelt_b16(j, static_cast<size_t>(count));
    svfloat16_t v_x = svld1(pg, reinterpret_cast<const __fp16*>(p_input + j));
    svfloat16_t v_tanh = svld1(pg, reinterpret_cast<const __fp16*>(p_temp + j));
    svfloat16_t v_sum = svadd_f16_m(pg, v_one, v_tanh);
    svfloat16_t v_result = svmul_f16_m(pg, v_half, svmul_f16_m(pg, v_x, v_sum));
    svst1(pg, reinterpret_cast<__fp16*>(p_output + j), v_result);
    j += svcnth();
  }
}
else if (approximation_algorithm_ == "none") {
          // Step 1: x * sqrt(1/2)
          size_t i = 0;
          while (i < static_cast<size_t>(count)) {
            svbool_t pg = svwhilelt_b16(i, static_cast<size_t>(count));
            svfloat16_t v_x = svld1(pg, reinterpret_cast<const __fp16*>(p_input + i));
            svfloat16_t v_scaled = svmul_f16_m(pg, v_x, v_sqrt1_2);
            svst1(pg, reinterpret_cast<__fp16*>(p_temp + i), v_scaled);
            i += svcnth();
          }
          // Step 2: Apply erf kernel in-place on temp buffer
          MlasSveErfKernelFp16(reinterpret_cast<const _mlas_fp16_*>(p_temp),
                               reinterpret_cast<_mlas_fp16_*>(p_temp),
                               count);
          // Step 3: GELU output = 0.5 * x * (1 + erf(x * sqrt(1/2)))
          size_t j = 0;
          while (j < static_cast<size_t>(count)) {
            svbool_t pg = svwhilelt_b16(j, static_cast<size_t>(count));
            svfloat16_t v_x = svld1(pg, reinterpret_cast<const __fp16*>(p_input + j));
            svfloat16_t v_erf = svld1(pg, reinterpret_cast<const __fp16*>(p_temp + j));
            svfloat16_t v_one_plus_erf = svadd_f16_m(pg, v_one, v_erf);
            svfloat16_t v_result = svmul_f16_m(pg, v_half, svmul_f16_m(pg, v_x, v_one_plus_erf));
            svst1(pg, reinterpret_cast<__fp16*>(p_output + j), v_result);
            j += svcnth();
          }
        }
      },
      0);
  std::free(temp_fp16_aligned);
  return Status::OK();
}*/

}  // namespace onnxruntime
