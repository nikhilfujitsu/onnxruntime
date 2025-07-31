// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

enum class MoEActivationType {
  Relu,
  Gelu,
  Silu,
  Identity,
  SwiGLU,

};

enum class MoEQuantType {
  None = 0,
  UINT4 = 1,
  UINT8 = 2,
};

enum class MoEParallelType {
  None = 0,
  EP = 1,
  TP = 2,
  EPAndTP = 3,
};

struct MoEParameters {
  MoEParameters() {}
  explicit MoEParameters(int64_t tensor_shards) : tensor_shards(tensor_shards) {}
  int64_t num_rows;
  int64_t num_experts;
  int64_t local_num_experts;
  int64_t hidden_size;
  int64_t inter_size;
  MoEParallelType parallel_type;
  int64_t tensor_shards{1};
};

Status CheckInputs(MoEParameters& parameters, MoEQuantType& quant_type, const Tensor* input,
                   const Tensor* router_probs, const Tensor* fc1_experts_weights,
                   const Tensor* fc1_experts_bias_optional, const Tensor* fc2_experts_weights,
                   const Tensor* fc2_experts_bias_optional, const Tensor* fc3_experts_weights_optional,
                   const Tensor* fc3_experts_bias_optional);

Status CheckInputScales(const Tensor* fc1_experts_scales, const Tensor* fc2_experts_scales, const Tensor* fc3_experts_scales_optional,
                        int64_t num_experts, int64_t hidden_size, int64_t inter_size, MoEActivationType activation_type);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
