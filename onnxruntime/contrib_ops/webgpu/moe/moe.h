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
  Relu, Gelu, Silu, Identity,
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

class MoEProgram final : public Program<MoEProgram> {
 public:
  MoEProgram(TensorShape output_shape) : Program<MoEProgram>{"MoE"}, output_shape_{output_shape} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32});

 private:
  TensorShape output_shape_;
};

class MoE final : public WebGpuKernel {
 public:
  MoE(const OpKernelInfo& info) : WebGpuKernel(info) {
    k_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("k", 128));
    normalize_routing_weights_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("normalize_routing_weights", 0)) == 1;
    use_sparse_mixer_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("use_sparse_mixer", 0)) == 1;
    std::string activation_type = info.GetAttrOrDefault<std::string>("activation_type", "relu");
    if (activation_type == "relu") {
      activation_type_ = MoEActivationType::Relu;
    } else if (activation_type == "gelu") {
      activation_type_ = MoEActivationType::Gelu;
    } else if (activation_type == "silu") {
      activation_type_ = MoEActivationType::Silu;
    } else if (activation_type == "identity") {
      activation_type_ = MoEActivationType::Identity;
    } else {
      ORT_THROW("Unsupported MoE activation type: ", activation_type);
    }
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int k_;
  bool normalize_routing_weights_;
  bool use_sparse_mixer_;
  MoEActivationType activation_type_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
