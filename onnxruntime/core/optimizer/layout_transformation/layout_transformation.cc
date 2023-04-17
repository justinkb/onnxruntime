// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// The ONNX Runtime specific implementation of the generic transpose optimizer API.

#include "core/optimizer/layout_transformation/layout_transformation.h"

#include "core/common/common.h"
#include "core/optimizer/transpose_optimization/ort_transpose_optimizer.h"
#include "core/optimizer/transpose_optimization/ort_optimizer_utils.h"

using namespace onnx_transpose_optimization;

namespace onnxruntime {
namespace layout_transformation {

bool IsNhwcExecutionProvider(std::string_view ep) {
  static const std::unordered_set<std::string_view> nhwc_eps = {
      kNnapiExecutionProvider,
      kQnnExecutionProvider,
      kXnnpackExecutionProvider,
  };

  return nhwc_eps.find(ep) != nhwc_eps.end();
}

// Layout sensitive NCHW ops. TransformLayoutForEP will wrap these with Transpose nodes to convert the input
// data to NHWC and output data back to NCHW, and move the op to the internal NHWC domain (kMSInternalNHWCDomain).
// The EP requesting these ops MUST be able to handle the node with the operator in the kMSInternalNHWCDomain.
// Once all the layout sensitive ops requested by the EP are wrapped the transpose optimizer will attempt to remove
// as many of the layout transposes as possible.
const std::unordered_set<std::string_view>& GetORTLayoutSensitiveOps() {
  static std::unordered_set<std::string_view> ort_layout_sensitive_ops = []() {
    const auto& layout_sensitive_ops = onnx_transpose_optimization::GetLayoutSensitiveOps();
    std::unordered_set<std::string_view> ort_specific_ops =
        {"FusedConv",
         "QLinearAveragePool",
         "QLinearGlobalAveragePool"};

    ort_specific_ops.insert(layout_sensitive_ops.cbegin(), layout_sensitive_ops.cend());
    return ort_specific_ops;
  }();

  return ort_layout_sensitive_ops;
}

const std::unordered_set<std::string_view> GetEPLayoutSensitiveOps(const IExecutionProvider& execution_provider) {
  std::unordered_set<std::string_view> layout_sensitive_ops = GetORTLayoutSensitiveOps();

  const auto& ep = execution_provider.Type();

  // EPs where the Resize implementation only handles one layout - either NCHW or NHWC. The ONNX spec for Resize is
  // not layout specific.
  const auto& layout_sensitive_eps = EPsWithLayoutSensitiveResize();
  if (layout_sensitive_eps.find(ep) != layout_sensitive_eps.end()) {
    layout_sensitive_ops.insert("Resize");
  }

  return layout_sensitive_ops;
}

// Cost check for aggressively pushing the Transpose nodes involved in the layout transformation further out.
static CostCheckResult
PostLayoutTransformCostCheck(const api::GraphRef& graph, const api::NodeRef& node,
                             const std::vector<int64_t>& perm,
                             const std::unordered_set<std::string>& outputs_leading_to_transpose) {
  // we aggressively push the layout transpose nodes.
  // Exception: pushing through a Concat can result in Transpose nodes being added to multiple other inputs which
  // can potentially be worse for performance. Use the cost check in that case.
  if (node.OpType() != "Concat" &&
      (perm == ChannelFirstToLastPerm(perm.size()) || perm == ChannelLastToFirstPerm(perm.size()))) {
    return CostCheckResult::kPushTranspose;
  }

  // for other nodes use the default ORT cost check
  return OrtEPCostCheck(graph, node, perm, outputs_leading_to_transpose);
}

Status TransformLayoutForEP(Graph& graph, bool& modified, const IExecutionProvider& execution_provider,
                            const DebugGraphFn& debug_graph_fn) {
  // sub graph recurse will be added later
  auto api_graph = MakeApiGraph(graph, execution_provider.GetAllocator(OrtMemTypeDefault), nullptr);
  const auto& layout_sensitive_ops = GetEPLayoutSensitiveOps(execution_provider);

  CostCheckFn cost_check;

  // if convert to NHWC we need to wrap layout sensitive nodes to Transpose from NCHW to NHWC and back.
  if (execution_provider.GetPreferredLayout() == DataLayout::NHWC) {
    for (auto& node : api_graph->Nodes()) {
      if (layout_sensitive_ops.count(node->OpType())) {
        if (node->GetExecutionProviderType() != execution_provider.Type()) {
          continue;
        }

        auto domain = node->Domain();
        // Skip if domain is incorrect
        if (domain != kOnnxDomain && domain != kMSDomain) {
          continue;
        }

        // if already transformed then change the domain to kMSInternalNHWCDomain this way the EP
        // knows this op is in the expected format.
        if (node->GetAttributeIntDefault("channels_last", 0) == 1) {
          SwapNodeOpTypeAndDomain(*api_graph, *node, node->OpType(), kMSInternalNHWCDomain);
          // Changing the domain for the node requires creating a new node and replacing the old one
          // therefore set the modified flag.
          modified = true;
          continue;
        }

        // Skip if unknown rank
        auto shape = api_graph->GetValueInfo(node->Inputs()[0])->Shape();
        if (!shape.has_value()) {
          continue;
        }

        // Convert to channels last
        size_t rank = shape->size();

        bool has_channel_last_attr = node->GetAttributeInt("channels_last").has_value() ? true : false;
        if (has_channel_last_attr) {
          node->SetAttributeInt("channels_last", 1);
        }

        auto input_perm = onnx_transpose_optimization::ChannelFirstToLastPerm(rank);
        auto output_perm = onnx_transpose_optimization::ChannelLastToFirstPerm(rank);

        // Except for resize and convolution ops, all the other layout sensitive ops only require layout transformation
        // for 0th input and output. For resize, add the other relevant inputs which need conversion. For Conv - layout
        // transformer only converts layout for 0th input, weights should be handled by every EP.
        if (node->OpType() == "Resize") {
          // Older versions of resize have a bug where ROI and Scales cannot be made empty inputs. To handle this case,
          // we need to jump a few extra hoops to make sure its inputs are correctly handled.
          //
          // Current code skips layout conversion for ROI because it needs special handling as ROI size is 2*rank.
          // Enable passing in ROI for layout conversion when an EP which supports ROI starts using layout transformer.
          // NNAPI which currently uses layout transformer does not support it.
          std::vector<const std::vector<int64_t>*> input_perms{&input_perm, nullptr};
          for (size_t i = 2; i < node->Inputs().size(); i++) {
            auto constant = api_graph->GetConstant(node->Inputs()[i]);
            if (constant != nullptr && constant->Data().size() > 0) {
              input_perms.push_back(&input_perm);
            } else {
              // TODO: Fix inconsistency. We should Transpose the non-const inputs so that the result of our changes
              // is consistent - all layout specific inputs are in NHWC format when we're done.
              // This may need to check the opset to see if it's safe so that an empty non-constant input doesn't
              // have an invalid Transpose added to it.
              // Caveat: Typically `scales` and `sizes` are constants so this may not happen in a production model.
              input_perms.push_back(nullptr);
            }
          }
          WrapTransposesAroundNode(*api_graph, *node, input_perms, {&output_perm});
        } else {
          WrapTransposesAroundNode(*api_graph, *node, {&input_perm}, {&output_perm});
        }

        SwapNodeOpTypeAndDomain(*api_graph, *node, node->OpType(), kMSInternalNHWCDomain);
        modified = true;
      }
    }
    cost_check = PostLayoutTransformCostCheck;

    // debug the changes made inserting Transpose nodes around layout sensitive ops.
    if (debug_graph_fn) {
      debug_graph_fn(graph);
    }

  } else {
    // layout is fine for the EP but we still want to run the transpose optimizer one more time for the EP specific
    // Transpose -> Resize logic.
    cost_check = OrtEPCostCheck;
  }

  const auto max_node_idx = graph.MaxNodeIndex();
  OptimizeResult result =
      onnx_transpose_optimization::Optimize(*api_graph, execution_provider.Type(),
                                            cost_check,
                                            OrtHandlers(),
                                            layout_sensitive_ops);
  if (result.error_msg) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Layout/Transpose optimization for ", execution_provider.Type(),
                           " failed: ", result.error_msg.value());
  }

  modified = modified || (graph.MaxNodeIndex() > max_node_idx);

  // debug optimization of the new Transpose nodes using PostLayoutTransformCostCheck
  if (modified && debug_graph_fn) {
    debug_graph_fn(graph);
  }

  return Status::OK();
}

bool IsSupportedOpset(const Graph& graph) {
  const auto& version_map = graph.DomainToVersionMap();
  const auto& onnx_version = version_map.find(kOnnxDomain);
  return (onnx_version != version_map.end() &&
          onnx_version->second >= onnx_transpose_optimization::kMinSupportedOpset &&
          onnx_version->second <= kMaxSupportedOpset);
}

void WrapTransposesAroundNode(api::GraphRef& graph, api::NodeRef& node,
                              const std::vector<const std::vector<int64_t>*>& input_perms,
                              const std::vector<const std::vector<int64_t>*>& output_perms) {
  for (size_t i = 0; i < input_perms.size(); ++i) {
    const std::vector<int64_t>* input_perm = input_perms[i];
    if (input_perm != nullptr) {
      TransposeInput(graph, node, i, *input_perm, InvertPerm(*input_perm));
    }
  }
  for (size_t i = 0; i < output_perms.size(); ++i) {
    const std::vector<int64_t>* output_perm = output_perms[i];
    if (output_perm != nullptr) {
      TransposeOutput(graph, node, i, *output_perm, InvertPerm(*output_perm));
    }
  }
}
}  // namespace layout_transformation
}  // namespace onnxruntime
