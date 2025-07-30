// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include "core/common/inlined_containers.h"
#include "core/providers/common.h"
#include "core/providers/webgpu/math/top_k.h"
#include "core/providers/webgpu/tensor/transpose.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_variable.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"
namespace onnxruntime {
namespace webgpu {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    TopK,
    kOnnxDomain,
    1, 9,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    TopK);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    TopK,
    kOnnxDomain,
    10, 10,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    TopK);

ONNX_OPERATOR_KERNEL_EX(
    TopK,
    kOnnxDomain,
    11,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    TopK);


Status TopKProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Add input and output variables
  const auto& input = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  shader.AddOutput("result", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  int components = input.NumComponents();

  // Define shared memory for row max and row sum
  shader.AdditionalImplementation()
      << "var<workgroup> row_max_shared : x_value_t;\n"
      << "var<workgroup> row_sum_shared : x_value_t;\n"
      << "var<workgroup> thread_shared : array<x_value_t, " << wg_ << ">;\n";

  // Define helper functions to get and set values
  shader.AdditionalImplementation()
      << "fn getValue(row: i32, col: i32, row_stride: i32) -> x_value_t {\n"
      << "  let index = row * row_stride + col;\n"
      << "  return x[index];\n"
      << "}\n"
      << "fn setValue(row: i32, col: i32, row_stride: i32, value: x_value_t) {\n"
      << "  let index = row * row_stride + col;\n"
      << "  result[index] = value;\n"
      << "}\n";

  // Main function body
  shader.MainFunctionBody()
      << "  let gindex = i32(global_idx);\n"
      << "  let lindex = i32(local_idx);\n"
      << "  const wg = " << wg_ << ";\n"
      << "  let row = gindex / wg;\n"
      << "  let cols = uniforms.packedCols;\n"
      << "  let row_stride : i32 = uniforms.packedCols;\n"

      // Calculate the final value for each element in the row
      << "  for (var col = lindex; col < cols; col += wg) {\n"
      << "    var value = exp(getValue(row, col, row_stride) - row_max_shared) / row_sum_shared;\n"
      << "    // max operation protects against NaN since all values should be >=0\n"
      << "    value = max(value, x_value_t(0.0));\n"
      << "    setValue(row, col, row_stride, value);\n"
      << "  }\n";

  return Status::OK();
}

Status TopK::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  const TensorShape& input_shape = input_tensor->Shape();
  size_t input_rank = input_shape.NumDimensions();
  auto* output_tensor = context.Output(0, input_shape);

  // normalize axis
  size_t axis = static_cast<size_t>(HandleNegativeAxis(axis_, input_rank));
  // The `axis` attribute of the opset lower than version 13 describes the axis of the inputs when coerced to 2D,
  // the 0th axis most likely describes the batch_size, so transpose is not required on old opset versions.
  int64_t cols = 4;
  int64_t rows = 4;
  const int64_t components = GetMaxComponents(cols);
  const auto packed_cols = cols / components;
  uint32_t workgroup_size = rows == 1 ? 256 : 64;

  TopKProgram program{workgroup_size};
  program
      .AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components)}})
      .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components)}})
      .CacheHint(std::to_string(components), std::to_string(workgroup_size))
      .SetWorkgroupSize(workgroup_size)
      .SetDispatchGroupSize(static_cast<uint32_t>(rows))
      .AddUniformVariables({{static_cast<int32_t>(packed_cols)}});

  ORT_RETURN_IF_ERROR(context.RunProgram(program));

  return Status::OK();
}
}  // namespace webgpu
}  // namespace onnxruntime
