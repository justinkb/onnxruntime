// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/multihead_attention.h"

#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/rocm/bert/attention_impl.h"
#include "contrib_ops/rocm/bert/batched_gemm_softmax_gemm_permute_pipelines.cuh"
#include "core/platform/env_var_utils.h"
#include "core/providers/rocm/rocm_common.h"

using namespace onnxruntime::rocm;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace rocm {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      MultiHeadAttention,                                         \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kRocmExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MultiHeadAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
MultiHeadAttention<T>::MultiHeadAttention(const OpKernelInfo& info)
    : RocmKernel(info) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = static_cast<int>(num_heads);

  mask_filter_value_ = info.GetAttrOrDefault<float>("mask_filter_value", -10000.0f);

  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
}

template <typename T>
Status MultiHeadAttention<T>::ComputeInternal(OpKernelContext* context) const {
  ORT_ENFORCE(
      GetTuningContext()->IsTunableOpEnabled(),
      "MultiHeadAttention of ROCm EP is only supported if tunable op is used and tuning is enabled.");

  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* bias = context->Input<Tensor>(3);
  const Tensor* key_padding_mask = context->Input<Tensor>(4);
  const Tensor* relative_position_bias = context->Input<Tensor>(5);
  const Tensor* past_key = context->Input<Tensor>(6);
  const Tensor* past_value = context->Input<Tensor>(7);

  if (nullptr != bias) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "qkv_bias is not supported on ROCm EP. "
                           "User should fuse the qkv bias to qkv projection instead.");
  }

  // TODO: Add support for key_padding_mask.
  ORT_ENFORCE(key_padding_mask == nullptr, "key_padding_mask is not supported");

  auto& device_prop = GetDeviceProp();
  AttentionParameters attn;
  ORT_RETURN_IF_ERROR(
      multihead_attention_helper::CheckInputs<Tensor>(
          query, key, value, bias,
          key_padding_mask, relative_position_bias,
          past_key, past_value, /*past_seq_len=*/nullptr,
          &attn,
          num_heads_, mask_filter_value_, scale_,
          false, device_prop.maxThreadsPerBlock));
  // TODO: support more qkv formats
  bool is_cross_attention = attn.qkv_format == Q_BSNH_K_V_BNSH_CROSS && attn.pass_past_in_kv;
  if (!(is_cross_attention || attn.qkv_format == QKV_BSN3H || attn.qkv_format == Q_KV_BSNH_BSN2H)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "qkv format is not supported, got ", attn.qkv_format);
  }

  ORT_ENFORCE(false, "qkv format is not blah, got ", attn.qkv_format);

  hipStream_t stream = Stream(context);

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(attn.batch_size);
  output_shape[1] = static_cast<int64_t>(attn.sequence_length);
  output_shape[2] = static_cast<int64_t>(attn.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  std::vector<int64_t> present_dims{
      attn.batch_size,
      attn.num_heads,
      attn.total_sequence_length,
      attn.head_size,
  };
  TensorShape present_shape(present_dims);
  Tensor* present_key = context->Output(1, present_shape);
  Tensor* present_value = context->Output(2, present_shape);

  using HipT = typename ToHipType<T>::MappedType;
  using AttentionTunableOp = GemmSoftmaxGemmPermuteTunableOp<HipT>;
  auto workspace_bytes = AttentionTunableOp::GetWorkspaceNumBytes(&attn);
  auto workspace = GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());

  const HipT* key_buffer = key == nullptr ? nullptr : reinterpret_cast<const HipT*>(key->DataRaw());
  const HipT* value_buffer = value == nullptr ? nullptr : reinterpret_cast<const HipT*>(value->DataRaw());

  if (nullptr != present_key) {
    // const HipT* past_key_buffer = past_key == nullptr ? nullptr : reinterpret_cast<const HipT*>(past_key->DataRaw());
    // const HipT* new_key_buffer = key_buffer;
    auto present_key_buffer = reinterpret_cast<HipT*>(present_key->MutableDataRaw());

    // int present_seqlen;
    if (!attn.past_present_share_buffer) {
      // present_seqlen = attn.total_sequence_length;
      if (attn.qkv_format == Q_K_V_BSNH) {
        // Copy past_k BxS'xNxH) => present_k Shape(BxNxS'xH):Stride(BxNxTxH)

      } else if (attn.qkv_format == Q_K_V_BNSH || attn.qkv_format == Q_BSNH_K_V_BNSH_CROSS) {
        // Copy past_k (BxNxS'xH) => present_k Shape(BxNxS'xH):Stride(BxNxTxH)
      }
    } else {
      // present_seqlen =
    }

    // Concat past_k (BxNxS'xH) + k (BxNxSxH) => present_k (BxNxTxH)

    // update pointers to present_k.
    key_buffer = present_key_buffer;
  }

  if (nullptr != present_value) {
    auto present_value_buffer = reinterpret_cast<HipT*>(present_value->MutableDataRaw());

    const HipT* past_value_buffer = nullptr;
    if (!attn.past_present_share_buffer) {
      ORT_ENFORCE(past_value != nullptr);
      past_value_buffer = reinterpret_cast<const HipT*>(past_value->DataRaw());
    }

    // Concat past_v (BxNxS'xH) + v (BxNxSxH) => present_v (BxNxTxH)
    ORT_RETURN_IF_ERROR(LaunchConcatTensorToTensor(
        stream,
        attn.total_sequence_length, attn.sequence_length, attn.batch_size, attn.v_head_size, attn.num_heads,
        device_prop.maxThreadsPerBlock, /*matrix_num=*/1,
        past_value_buffer, value_buffer, present_value_buffer));

    // update pointers to present_v.
    value_buffer = present_value_buffer;
  }

  GemmSoftmaxGemmPermuteParams<HipT> params;
  params.tuning_ctx = GetTuningContext();
  params.stream = Stream(context);
  params.handle = GetRocblasHandle(context);
  params.attention = &attn;
  params.device_prop = &device_prop;
  params.scale = scale_ == 0 ? 1.0f / sqrt(attn.head_size) : scale_;
  std::tie(params.q_buffer, params.k_buffer, params.v_buffer) = GetQkvBuffers<HipT>(
      &attn,
      reinterpret_cast<const HipT*>(query->DataRaw()),
      key_buffer,
      value_buffer);
  params.out_buffer = reinterpret_cast<HipT*>(output->MutableDataRaw());

  if (relative_position_bias != nullptr) {
    params.bias_buffer = reinterpret_cast<const HipT*>(relative_position_bias->DataRaw());
  }

  params.workspace_buffer = reinterpret_cast<HipT*>(workspace.get());
  return AttentionTunableOp{}(&params);
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
