#pragma once

#include <llga/llga.hpp>
#include <torch/csrc/jit/codegen/onednn/operator.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

struct OpPartitionMap {
  void add(uint64_t opId, uint64_t partitionId) {
    opmap[opId] = partitionId;
  }
  void add(Node* n, uint64_t partitionId) {
    add(Operator::getId(n), partitionId);
  }
  bool has(uint64_t opId) {
    return opmap.count(opId) > 0;
  }
  bool has(Node* n) {
    return has(Operator::getId(n));
  }
  uint64_t get(uint64_t opId) {
    return opmap[opId];
  }
  uint64_t get(Node* n) {
    auto opId = Operator::getId(n);
    TORCH_CHECK(
        has(opId),
        "Node ",
        n->kind().toQualString(),
        " does not belong to any LLGA partition");
    return get(opId);
  }

 private:
  std::unordered_map<uint64_t, uint64_t> opmap;
};

class LlgaGraphHelper {
 public:
  LlgaGraphHelper(
      const std::shared_ptr<Graph>& graph,
      dnnl::graph::partition::policy policy =
          dnnl::graph::partition::policy::fusion);

  bool shouldMerge(Node* toMerge, Node* subgraph);

  bool shouldConsiderForMerge(Node* node);

  Node* createSingletonSubgraph(Node* n, AliasDb& db);

  void mergeNodeIntoSubgraph(Node* toMerge, Node* subgraphNode, AliasDb& db);

  void unmergeIfAnyNodeIsMissing(Node* subgraphNode);

  static bool isLlgaSubgraph(const Node* node);

  std::vector<dnnl::graph::partition> getPartitions() const;

 private:
  size_t countSupportedOps(const std::shared_ptr<Graph>& graph) const;

  OpPartitionMap opToOwningPartition;
  std::vector<dnnl::graph::partition> partitions;
};

class LlgaNodeWrapper {
 public:
  LlgaNodeWrapper(const Node* node);

  void setOpaqueLayout(size_t offset);

  bool useOpaqueLayout(size_t offset) const;

  friend class LlgaGraphHelper;

 private:
  void initOutputLayouts();

  Node* n;
};

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
