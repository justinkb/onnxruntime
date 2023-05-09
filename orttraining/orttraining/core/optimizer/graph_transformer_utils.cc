// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include "orttraining/core/optimizer/graph_transformer_utils.h"

#include "core/mlas/inc/mlas.h"
#include "core/optimizer/bias_dropout_fusion.h"
#include "core/optimizer/bias_gelu_fusion.h"
#include "core/optimizer/bias_softmax_fusion.h"
#include "core/optimizer/cast_elimination.h"
#include "core/optimizer/common_subexpression_elimination.h"
#include "core/optimizer/concat_slice_elimination.h"
#include "core/optimizer/constant_folding.h"
#include "core/optimizer/constant_sharing.h"
#include "core/optimizer/conv_activation_fusion.h"
#include "core/optimizer/conv_add_fusion.h"
#include "core/optimizer/conv_bn_fusion.h"
#include "core/optimizer/conv_mul_fusion.h"
#include "core/optimizer/div_mul_fusion.h"
#include "core/optimizer/dropout_elimination.h"
#include "core/optimizer/embed_layer_norm_fusion.h"
#include "core/optimizer/expand_elimination.h"
#include "core/optimizer/fast_gelu_fusion.h"
#include "core/optimizer/free_dim_override_transformer.h"
#include "core/optimizer/gather_fusion.h"
#include "core/optimizer/gelu_approximation.h"
#include "core/optimizer/gelu_fusion.h"
#include "core/optimizer/gemm_activation_fusion.h"
#include "core/optimizer/gemm_sum_fusion.h"
#include "core/optimizer/gemm_transpose_fusion.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/identity_elimination.h"
#include "core/optimizer/isinf_reducesum_fusion.h"
#include "core/optimizer/layer_norm_fusion.h"
#include "core/optimizer/matmul_add_fusion.h"
#include "core/optimizer/matmul_scale_fusion.h"
#include "core/optimizer/matmul_transpose_fusion.h"
#include "core/optimizer/nchwc_transformer.h"
#include "core/optimizer/noop_elimination.h"
#include "core/optimizer/not_where_fusion.h"
#include "core/optimizer/propagate_cast_ops.h"
#include "core/optimizer/quick_gelu_fusion.h"
#include "core/optimizer/relu_clip_fusion.h"
#include "core/optimizer/reshape_fusion.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/skip_layer_norm_fusion.h"
#include "core/optimizer/slice_elimination.h"
#include "core/optimizer/unsqueeze_elimination.h"
#include "core/session/inference_session.h"
#include "orttraining/core/framework/distributed_run_context.h"
#include "orttraining/core/optimizer/batchnorm_replacement.h"
#include "orttraining/core/optimizer/bitmask_dropout_replacement.h"
#include "orttraining/core/optimizer/concat_replacement.h"
#include "orttraining/core/optimizer/graph_transformer_registry.h"
#include "orttraining/core/optimizer/insert_output_rewriter.h"
#include "orttraining/core/optimizer/localized_recompute.h"
#include "orttraining/core/optimizer/loss_rewriter.h"
#include "orttraining/core/optimizer/lstm_replacement.h"
#include "orttraining/core/optimizer/transformer_layer_recompute.h"
#include "orttraining/core/optimizer/qdq_fusion.h"
#include "orttraining/core/optimizer/shape_optimizer.h"
#include "orttraining/core/optimizer/transformer_layer_recompute.h"

// Only enabled in full training build. Not in on device training builds
#ifdef ENABLE_TRAINING
#include "core/optimizer/compute_optimizer/upstream_gather.h"
#include "core/optimizer/compute_optimizer/upstream_reshape.h"
#include "orttraining/core/optimizer/compute_optimizer/sceloss_compute_optimization.h"
#endif

namespace onnxruntime {
namespace training {
namespace transformer_utils {

std::vector<std::unique_ptr<GraphTransformer>> GeneratePreTrainingTransformers(
    TransformerLevel level,
    const std::unordered_set<std::string>& weights_to_train,
    const TrainingGraphTransformerConfiguration& config,
    const IExecutionProvider& execution_provider,
    const std::unordered_set<std::string>& rules_and_transformers_to_disable) {
  std::vector<std::unique_ptr<GraphTransformer>> transformers;
  std::unique_ptr<RuleBasedGraphTransformer> rule_transformer = nullptr;

  // MUST be empty here, because this is called before partition, so the node's execution type is not decided yet.
  // If we give values here, the check in transformer will fail.
  InlinedHashSet<std::string_view> compatible_eps = {};

  switch (level) {
    case TransformerLevel::Level1: {
      rule_transformer =
          std::make_unique<RuleBasedGraphTransformer>(optimizer_utils::GenerateRuleBasedTransformerName(level),
                                                      compatible_eps);
      ORT_THROW_IF_ERROR(rule_transformer->Register(std::make_unique<InsertMaxPoolOutput>()));
      ORT_THROW_IF_ERROR(rule_transformer->Register(std::make_unique<BatchNormReplacement>()));
      ORT_THROW_IF_ERROR(rule_transformer->Register(std::make_unique<UnsqueezeElimination>()));
      ORT_THROW_IF_ERROR(rule_transformer->Register(std::make_unique<ExpandElimination>()));
      ORT_THROW_IF_ERROR(rule_transformer->Register(std::make_unique<CastElimination>()));
      ORT_THROW_IF_ERROR(rule_transformer->Register(std::make_unique<NoopElimination>()));
      ORT_THROW_IF_ERROR(rule_transformer->Register(std::make_unique<DivMulFusion>()));
      ORT_THROW_IF_ERROR(rule_transformer->Register(std::make_unique<EliminateDropout>()));
      ORT_THROW_IF_ERROR(rule_transformer->Register(std::make_unique<GemmSumFusion>()));
      ORT_THROW_IF_ERROR(rule_transformer->Register(std::make_unique<GemmTransposeFusion>()));
      ORT_THROW_IF_ERROR(rule_transformer->Register(std::make_unique<NotWhereFusion>()));
      ORT_THROW_IF_ERROR(rule_transformer->Register(std::make_unique<InsertSoftmaxCrossEntropyLossOutput>()));
      ORT_THROW_IF_ERROR(rule_transformer->Register(std::make_unique<LSTMReplacement>()));

      // Put ConstantSharing before CommonSubexpressionElimination by intention as it can create more opportunities for
      // CSE. For example, if A and B nodes both do Add operation with a same value but different initializers, by
      // default, CSE will not merge them, because the different initializers are represented by different NodeArg.
      transformers.emplace_back(std::make_unique<LayerNormFusion>(compatible_eps));
      transformers.emplace_back(std::make_unique<ConstantSharing>(compatible_eps));
      // Remove duplicate nodes. Must be applied before any recompute transformations.
      if (config.gelu_recompute || config.attn_dropout_recompute || config.transformer_layer_recompute) {
        transformers.emplace_back(std::make_unique<CommonSubexpressionEliminationApplyOnce>(compatible_eps));
      } else {
        transformers.emplace_back(std::make_unique<CommonSubexpressionElimination>(compatible_eps));
      }

      transformers.emplace_back(std::make_unique<GeluFusion>(compatible_eps));
#if defined(USE_CUDA) || defined(USE_ROCM)
      transformers.emplace_back(std::make_unique<SimplifiedLayerNormFusion>(compatible_eps,
                                                                            true /* skip_device_check*/));
#else
      transformers.emplace_back(std::make_unique<SimplifiedLayerNormFusion>(compatible_eps));
#endif
      transformers.emplace_back(std::make_unique<FastGeluFusion>(compatible_eps));
      transformers.emplace_back(std::make_unique<QuickGeluFusion>(compatible_eps));
      transformers.emplace_back(std::make_unique<SoftmaxCrossEntropyLossInternalFusion>(compatible_eps));
      transformers.emplace_back(std::make_unique<GatherToSplitFusion>(compatible_eps));
      transformers.emplace_back(std::make_unique<GatherToSliceFusion>(compatible_eps));
      // If a model with Q, DQ nodes is being used for the purpose of training, it must be for
      // Quantization Aware Training. So, replace QDQ nodes with FakeQuant.
      transformers.emplace_back(std::make_unique<QDQFusion>(compatible_eps));

#if defined(USE_CUDA) || defined(USE_ROCM)
      // We are supposed to use execution provider as indicator, but here we don't have access to the registered EP at this point
      // as the session is not initialized yet. So using macro for now.
      transformers.emplace_back(std::make_unique<BiasGeluFusion>(compatible_eps));
      transformers.emplace_back(std::make_unique<IsInfReduceSumFusion>(compatible_eps));
#endif

      if (config.enable_gelu_approximation) {
        transformers.emplace_back(std::make_unique<GeluApproximation>(compatible_eps));
      }
      InlinedHashSet<std::string> excluded_initializers(weights_to_train.begin(), weights_to_train.end());
      transformers.emplace_back(std::make_unique<ConstantFolding>(
          execution_provider, false /*skip_dequantize_linear*/, compatible_eps, excluded_initializers));
      transformers.emplace_back(std::make_unique<ReshapeFusion>(compatible_eps));
      // Put fine-grained optimizer (e.g. ShapeOptimizer) after ReshapeFusion to avoid it breaks the strong patterns
      // it defines. ReshapeFusion depends on subgraph pattern matching and do replacement accordingly, ShapeOptimizer
      // potentially will optimize out some nodes defined in the subgraph patterns. So we put it after ReshapeFusion.
      transformers.emplace_back(std::make_unique<ShapeOptimizer>(compatible_eps));
      transformers.emplace_back(std::make_unique<ConcatSliceElimination>(compatible_eps));

      if (config.gelu_recompute) {
        transformers.emplace_back(std::make_unique<GeluRecompute>());
      }
      if (config.attn_dropout_recompute) {
        transformers.emplace_back(std::make_unique<AttentionDropoutRecompute>());
      }
      if (config.transformer_layer_recompute) {
        transformers.emplace_back(std::make_unique<TransformerLayerRecompute>(
            config.number_recompute_layers, compatible_eps));
      }
      if (config.propagate_cast_ops_config.level >= 0) {
        const InlinedHashSet<std::string_view> cuda_execution_provider = {onnxruntime::kCudaExecutionProvider,
                                                                          onnxruntime::kRocmExecutionProvider};
        transformers.emplace_back(std::make_unique<PropagateCastOps>(config.propagate_cast_ops_config.strategy,
                                                                     static_cast<size_t>(config.propagate_cast_ops_config.level),
                                                                     config.propagate_cast_ops_config.allow,
                                                                     cuda_execution_provider));
      }

#ifdef ENABLE_TRAINING
      if (config.enable_compute_optimizer) {
        transformers.emplace_back(std::make_unique<UpStreamGatherGraphTransformer>(compatible_eps));
        if (config.enable_label_sparsity_optimization) {
          transformers.emplace_back(std::make_unique<UpStreamReshapeGraphTransformer>(compatible_eps));
          transformers.emplace_back(std::make_unique<InsertGatherBeforeSceLoss>(compatible_eps));
        }
      }
#endif

    } break;

    case TransformerLevel::Level2: {
      rule_transformer =
          std::make_unique<RuleBasedGraphTransformer>(optimizer_utils::GenerateRuleBasedTransformerName(level),
                                                      compatible_eps);
      ORT_THROW_IF_ERROR(rule_transformer->Register(std::make_unique<ConcatReplacement>()));
    } break;

    case TransformerLevel::Level3: {
    } break;

    default:
      ORT_ENFORCE(false, "Unsupported level " + std::to_string(static_cast<uint32_t>(level)));
      break;
  }

  // Note that some rule-based transformers are depending on some custom transformers,
  // e.g., ExpandElimination and CastElimination are depending on ConstantFolding to fold the constant first,
  // so we should always push the rule-based transformer to the end, this is especially important when the number of
  // transformation steps is 1.
  if (rule_transformer != nullptr) {
    transformers.emplace_back(std::move(rule_transformer));
  }

  GenerateExternalTransformers(level, true, compatible_eps, transformers);

  if (rules_and_transformers_to_disable.empty()) {
    return transformers;
  } else {
    // filter out any disabled transformers
    std::vector<std::unique_ptr<GraphTransformer>> filtered_list;
    auto end = rules_and_transformers_to_disable.cend();
    std::for_each(transformers.begin(), transformers.end(),
                  [&](std::unique_ptr<GraphTransformer>& item) {
                    if ((item != nullptr) && (rules_and_transformers_to_disable.find(item->Name()) == end)) {
                      filtered_list.push_back(std::move(item));
                    }
                  });

    return filtered_list;
  }
}

InlinedVector<std::unique_ptr<GraphTransformer>> GenerateTransformers(
    TransformerLevel level,
    const std::unordered_set<std::string>& weights_to_train,
    gsl::span<const FreeDimensionOverride> free_dimension_overrides,
    const InlinedHashSet<std::string>& rules_and_transformers_to_disable) {
  InlinedVector<std::unique_ptr<GraphTransformer>> transformers;
  std::unique_ptr<RuleBasedGraphTransformer> rule_transformer = nullptr;
  switch (level) {
    case TransformerLevel::Level1: {
      InlinedHashSet<std::string_view> l1_execution_providers = {};
      InlinedHashSet<std::string_view> cuda_rocm_execution_providers = {onnxruntime::kCudaExecutionProvider,
                                                                        onnxruntime::kRocmExecutionProvider};

      // TODO hack - constant folding currently doesn't work after mixed precision transformation so it's disabled for now
      //             ORT uses CPU kernels to evaluate constant values but some of them don't support fp16
      // transformers.emplace_back(std::make_unique<ConstantFolding>(l1_execution_providers));
      transformers.emplace_back(std::make_unique<MatMulAddFusion>(l1_execution_providers));
      transformers.emplace_back(std::make_unique<FreeDimensionOverrideTransformer>(free_dimension_overrides));
      transformers.emplace_back(std::make_unique<MatmulTransposeFusion>(cuda_rocm_execution_providers));
      transformers.emplace_back(std::make_unique<BiasDropoutFusion>(cuda_rocm_execution_providers));
      transformers.emplace_back(std::make_unique<BitmaskDropoutReplacement>(cuda_rocm_execution_providers));
      transformers.emplace_back(std::make_unique<BiasSoftmaxFusion>(l1_execution_providers));
      InlinedHashSet<std::string> excluded_initializers(weights_to_train.begin(), weights_to_train.end());
      transformers.emplace_back(std::make_unique<MatMulScaleFusion>(l1_execution_providers, excluded_initializers));

      rule_transformer = optimizer_utils::GenerateRuleBasedGraphTransformer(level, rules_and_transformers_to_disable,
                                                                            l1_execution_providers);
    } break;

    case TransformerLevel::Level2: {
      InlinedHashSet<std::string_view> cpu_execution_providers = {onnxruntime::kCpuExecutionProvider};

      // create rule based transformer consisting of all the level2 rewrite rules
      rule_transformer = optimizer_utils::GenerateRuleBasedGraphTransformer(level, rules_and_transformers_to_disable,
                                                                            cpu_execution_providers);

      transformers.emplace_back(std::make_unique<GemmActivationFusion>(cpu_execution_providers));
      transformers.emplace_back(std::make_unique<ConvActivationFusion>(cpu_execution_providers));
    } break;

    case TransformerLevel::Level3: {
      // Register the NCHWc layout transformer if supported by the platform.
      if (MlasNchwcGetBlockSize() > 1) {
        transformers.emplace_back(std::make_unique<NchwcTransformer>());
      }
    } break;

    default:
      ORT_ENFORCE(false, "Unsupported level " + std::to_string(static_cast<uint32_t>(level)));
      break;
  }

  if (rule_transformer != nullptr) {
    transformers.emplace_back(std::move(rule_transformer));
  }

  // if the custom list to enable transformers\rules is empty then return the default generated transformers and rules
  // otherwise generate a filtered list based on the provided custom list.
  if (rules_and_transformers_to_disable.empty()) {
    return transformers;
  } else {
    InlinedVector<std::unique_ptr<GraphTransformer>> filtered_list;
    auto end = rules_and_transformers_to_disable.cend();
    std::for_each(transformers.begin(), transformers.end(),
                  [&](std::unique_ptr<GraphTransformer>& item) {
                    if ((item != nullptr) && (rules_and_transformers_to_disable.find(item->Name()) == end)) {
                      filtered_list.push_back(std::move(item));
                    }
                  });

    return filtered_list;
  }
}

}  // namespace transformer_utils
}  // namespace training
}  // namespace onnxruntime
