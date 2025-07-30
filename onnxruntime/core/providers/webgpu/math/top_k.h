// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace webgpu {

class TopK final : public WebGpuKernel {
 public:
  TopK(const OpKernelInfo& info) : WebGpuKernel{info} {
    axis_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("axis", -1));
    largest_ = info.GetAttrOrDefault<int64_t>("largest", 1) != 0;
    sorted_ = info.GetAttrOrDefault<int64_t>("sorted", 1) != 0;
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t axis_;
  bool largest_;
  bool sorted_;
};

class TopKProgram final : public Program<TopKProgram> {
 public:
  TopKProgram(uint32_t wg)
      : Program{"TopK"}, wg_{wg} {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Int32});

 private:
  uint32_t wg_;
};

}  // namespace webgpu
}  // namespace onnxruntime
