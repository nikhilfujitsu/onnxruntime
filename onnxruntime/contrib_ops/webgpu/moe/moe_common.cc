#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "contrib_ops/webgpu/moe/moe_base.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

Status CheckInputs(MoEParameters& parameters, MoEQuantType& quant_type, const Tensor* input,
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
};

Status CheckInputScales(const Tensor* fc1_experts_scales, const Tensor* fc2_experts_scales, const Tensor* fc3_experts_scales_optional,
                        int64_t num_experts, int64_t hidden_size, int64_t inter_size, MoEActivationType activation_type) {
  if (fc1_experts_scales == nullptr || fc2_experts_scales == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc1_experts_scales and fc2_experts_scales cannot be null for quantized MoE");
  }

  // SwiGLU should not use separate FC3 scales - weights are concatenated in FC1
  if (activation_type == MoEActivationType::SwiGLU && fc3_experts_scales_optional != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "SwiGLU should not use separate fc3_experts_scales. Gate weights should be concatenated with FC1 weights.");
  }
  if (activation_type != MoEActivationType::SwiGLU && fc3_experts_scales_optional != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "FC3 gating is not yet implemented for non-SwiGLU activations on CPU.");
  }

  const auto& fc1_experts_scales_dims = fc1_experts_scales->Shape().GetDims();
  const auto& fc2_experts_scales_dims = fc2_experts_scales->Shape().GetDims();

  if (fc1_experts_scales_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc1_experts_scales must be 2D, got ",
                           fc1_experts_scales_dims.size());
  }
  if (fc2_experts_scales_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc2_experts_scales must be 2D, got ",
                           fc2_experts_scales_dims.size());
  }
  if (fc1_experts_scales_dims[0] != num_experts) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc1_experts_scales[0] must be equal to num_experts, got ",
                           fc1_experts_scales_dims[0], " and ", num_experts);
  }

  const int64_t act = activation_type == MoEActivationType::SwiGLU ? 2 : 1;  // SwiGLU requires 2x scales
  if (fc1_experts_scales_dims[1] != act * inter_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc1_experts_scales[1] is ", fc1_experts_scales_dims[1],
                           " expected ", act * inter_size);
  }
  if (fc2_experts_scales_dims[0] != num_experts) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc2_experts_scales[0] must be equal to num_experts, got ",
                           fc2_experts_scales_dims[0], " and ", num_experts);
  }
  if (fc2_experts_scales_dims[1] != hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc2_experts_scales[1] must be equal to hidden_size, got ",
                           fc2_experts_scales_dims[1], " and ", hidden_size);
  }

  return Status::OK();
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
