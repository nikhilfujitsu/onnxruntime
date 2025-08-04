// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "contrib_ops/webgpu/moe/moe_base.h"
#include "contrib_ops/webgpu/moe/moe.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

class QMoEProgram final : public Program<QMoEProgram> {
 public:
  QMoEProgram(TensorShape output_shape,  MoEActivationType activation_type) :
    Program<QMoEProgram>{"QMoE"}, output_shape_{output_shape}, activation_type_{activation_type} {};

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32});

 private:
  TensorShape output_shape_;
  MoEActivationType activation_type_;
};

class QMoE final : public MoE {
 public:
  QMoE(const OpKernelInfo& info) : MoE(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("expert_weight_bits", &expert_weight_bits_).IsOK());
    ORT_ENFORCE(expert_weight_bits_ == 8 || expert_weight_bits_ == 4,
       "expert_weight_bits must be 4 or 8, but got ", expert_weight_bits_);
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t expert_weight_bits_;
  float activation_alpha_;
  float activation_beta_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
