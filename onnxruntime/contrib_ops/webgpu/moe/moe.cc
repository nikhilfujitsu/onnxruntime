// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "contrib_ops/webgpu/moe/moe.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

static Status CheckInputs(MoEParameters& parameters, MoEQuantType& quant_type, const Tensor* input,
                          const Tensor* router_probs, const Tensor* fc1_experts_weights,
                          const Tensor* fc1_experts_bias_optional, const Tensor* fc2_experts_weights,
                          const Tensor* fc2_experts_bias_optional, const Tensor* fc3_experts_weights_optional,
                          const Tensor* fc3_experts_bias_optional) {
  const auto& input_dims = input->Shape().GetDims();
  const auto& router_probs_dims = router_probs->Shape().GetDims();
  const auto& fc1_experts_weights_dims = fc1_experts_weights->Shape().GetDims();
  const auto& fc2_experts_weights_dims = fc2_experts_weights->Shape().GetDims();

  int64_t num_rows = input_dims.size() == 2 ? input_dims[0] : input_dims[0] * input_dims[1];
  int64_t hidden_size = input_dims[input_dims.size() - 1];
  int64_t local_num_experts = fc1_experts_weights_dims[0];
  int64_t num_experts = router_probs_dims[1];
  int64_t inter_size = fc2_experts_weights_dims[1];

  if (fc1_experts_weights_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc1_experts_weights_dims must be 3D, got ",
                           fc1_experts_weights_dims.size());
  }
  if (fc2_experts_weights_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc2_experts_weights_dims must be 3D, got ",
                           fc2_experts_weights_dims.size());
  }
  if (fc1_experts_weights_dims[1] != hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "fc1_experts_weights_dims[1] must be equal to hidden_size, got ",
                           fc1_experts_weights_dims[1], " and ", hidden_size);
  }
  if (fc2_experts_weights_dims[1] != inter_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "fc2_experts_weights_dims[1] must be equal to inter_size, got ",
                           fc2_experts_weights_dims[1], " and ", inter_size);
  }

  const int64_t coe = quant_type == MoEQuantType::UINT4 ? 2 : 1;
  if (fc1_experts_weights_dims[2] != inter_size / coe) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "fc1_experts_weights_dims[2] must be equal to inter_size, got ",
                           fc1_experts_weights_dims[2], " and ", inter_size);
  }
  if (fc2_experts_weights_dims[2] != hidden_size / coe) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "fc2_experts_weights_dims[2] must be equal to hidden_size, got ",
                           fc2_experts_weights_dims[2], " and ", hidden_size);
  }

  if (router_probs_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "router_probs_dims must be 2D, got ",
                           router_probs_dims.size());
  }
  if (router_probs_dims[0] != num_rows) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "router_probs_dims[0] must be equal to num_rows, got ",
                           router_probs_dims[0], " and ", num_rows);
  }
  if (fc1_experts_bias_optional != nullptr && fc2_experts_bias_optional != nullptr) {
    const auto& fc1_experts_bias_dims = fc1_experts_bias_optional->Shape().GetDims();
    const auto& fc2_experts_bias_dims = fc2_experts_bias_optional->Shape().GetDims();
    if (fc1_experts_bias_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc1_experts_bias_dims must be 2D, got ",
                             fc1_experts_bias_dims.size());
    }
    if (fc2_experts_bias_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc2_experts_bias_dims must be 2D, got ",
                             fc2_experts_bias_dims.size());
    }
    if (fc1_experts_bias_dims[0] != local_num_experts) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "fc1_experts_bias_dims[0] must be equal to local_num_experts, got ",
                             fc1_experts_bias_dims[0], " and ", local_num_experts);
    }
    if (fc2_experts_bias_dims[0] != num_experts) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "fc2_experts_bias_dims[0] must be equal to num_experts, got ", fc2_experts_bias_dims[0],
                             " and ", num_experts);
    }
    if (fc1_experts_bias_dims[1] != inter_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "fc1_experts_bias_dims[1] must be equal to inter_size, got ", fc1_experts_bias_dims[1],
                             " and ", inter_size);
    }
    if (fc2_experts_bias_dims[1] != hidden_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "fc2_experts_bias_dims[1] must be equal to hidden_size, got ", fc2_experts_bias_dims[1],
                             " and ", hidden_size);
    }
  }

  if (fc3_experts_weights_optional != nullptr &&
      fc3_experts_weights_optional->Shape().GetDims() != fc1_experts_weights_dims) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "fc3_experts_weights_dims must be equal to fc1_experts_weights_dims, got ",
                           fc3_experts_weights_optional->Shape(), " and ", TensorShape(fc1_experts_weights_dims));
  }

  if (fc3_experts_bias_optional != nullptr && fc1_experts_bias_optional != nullptr &&
      fc3_experts_bias_optional->Shape().GetDims() != fc1_experts_bias_optional->Shape().GetDims()) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT, "fc3_experts_bias_dims must be equal to fc1_experts_bias_dims, got ",
        fc3_experts_bias_optional->Shape(), " and ", fc1_experts_bias_optional->Shape());
  }

  parameters.num_rows = num_rows;
  parameters.num_experts = num_experts;
  parameters.local_num_experts = local_num_experts;
  parameters.hidden_size = hidden_size;
  parameters.inter_size = inter_size;
  if (num_experts == local_num_experts) {
      if (parameters.tensor_shards == 1) {
        parameters.parallel_type = MoEParallelType::None;
      } else {
        parameters.parallel_type = MoEParallelType::TP;
      }
    } else if (num_experts > local_num_experts) {
      if (parameters.tensor_shards == 1) {
        parameters.parallel_type = MoEParallelType::EP;
      } else {
        parameters.parallel_type = MoEParallelType::EPAndTP;
      }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "num_experts must be greater than or equal to local_num_experts, got ", num_experts,
                           " and ", local_num_experts);
  }

  return Status::OK();
}

Status MoEProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("input", ShaderUsage::UseElementTypeAlias);
#if 0
  const auto& x_shape = shader.AddIndices("input_shape", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& indices = shader.AddInput("indices", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseIndicesToOffset);
  const auto& scales = shader.AddInput("scales", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride | ShaderUsage::UseValueTypeAlias);

  shader.MainFunctionBody()
      << "var data_indices = input_shape_indices_t(0);\n"
      << "for (var i: u32 = 0; i < uniforms.gather_axis; i++) {\n"
      << "  let index = " << output.IndicesGet("output_indices", "i") << ";\n  "
      << x_shape.IndicesSet("data_indices", "i", "index") << ";\n};\n"
      << "var index_from_indices = " << indices.GetByIndices("indices_indices") << ";\n"
      << "if (index_from_indices < 0) { index_from_indices += " << x_shape_[gather_axis_] << ";}\n"
      << x_shape.IndicesSet("data_indices", "uniforms.gather_axis", "u32(index_from_indices)") << ";\n"
      << "for (var i = uniforms.gather_axis + 1; i < " << output_shape_.NumDimensions() << "; i++) {\n"
      << "  let index = " << output.IndicesGet("output_indices", "i + " + std::to_string(indices_rank_ - 1)) << ";\n  "
      << x_shape.IndicesSet("data_indices", "i", "index") << ";\n};\n"
      << "  let data_offset = " << x_shape.IndicesToOffset("data_indices") << ";\n";

  shader.MainFunctionBody()
      << "  let dequantized_data = (output_value_t(quantized_data) - output_value_t(zero_point)) * scale;\n  "
      << output.SetByOffset("global_idx", "dequantized_data") << ";\n";
#endif
  return Status::OK();
}

Status MoE::ComputeInternal(ComputeContext& context) const {
  const Tensor* input = context.Input(0);                // (num_rows, hidden_size) or (batch_size, sequence_length, hidden_size)
  const Tensor* router_probs = context.Input(1);         // (num_rows, num_experts)
  const Tensor* fc1_experts_weights = context.Input(2);  // (num_experts, hidden_size, inter_size)
  const Tensor* fc1_experts_bias = context.Input(3);     // (num_experts, hidden_size, inter_size)
  const Tensor* fc2_experts_weights = context.Input(4);  // (num_experts, hidden_size, inter_size)
  const Tensor* fc2_experts_bias = context.Input(5);     // (num_experts, hidden_size, inter_size)
  const Tensor* fc3_experts_weights = context.Input(6);  // (num_experts, hidden_size, inter_size)
  const Tensor* fc3_experts_bias = context.Input(7);     // (num_experts, hidden_size, inter_size)

  MoEParameters moe_params;
  MoEQuantType quant_type = MoEQuantType::None;

  ORT_RETURN_IF_ERROR(CheckInputs(moe_params, quant_type, input, router_probs,
                                  fc1_experts_weights, fc1_experts_bias,
                                  fc2_experts_weights, fc2_experts_bias,
                                  fc3_experts_weights, fc3_experts_bias));

  const auto& input_shape = input->Shape();

  auto* output_tensor = context.Output(0, input_shape);
  int output_size = static_cast<int>(input_shape.Size());

  MoEProgram program{input_shape};

  program
      .AddInputs({{input, ProgramTensorMetadataDependency::Type}})
      .AddInputs({{router_probs, ProgramTensorMetadataDependency::Type}})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(output_size)}})
      .CacheHint("hint?");

  return context.RunProgram(program);
}

ONNX_OPERATOR_KERNEL_EX(
    MoE,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    MoE);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
