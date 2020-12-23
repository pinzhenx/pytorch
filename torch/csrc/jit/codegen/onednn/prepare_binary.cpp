#include <torch/csrc/jit/codegen/onednn/prepare_binary.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

bool compareConstValue(Value* v, double d) {
  auto ival = toIValue(v);
  return ival.has_value() &&
      ((ival->isInt() && ival->toInt() == static_cast<int>(d)) ||
       (ival->isDouble() && ival->toDouble() == d));
}

void mayConvertScalarInputToTensor(Node* node) {
  // We do not handle binary ops with two scalar inputs,
  // and we assume scalar is always at the second place.
  if (node->input(0)->type()->isSubtypeOf(TensorType::get()) &&
      (node->input(1)->type()->isSubtypeOf(FloatType::get()) ||
       node->input(1)->type()->isSubtypeOf(IntType::get()))) {
    auto scalar = node->input(1);
    WithInsertPoint guard(node);
    auto g = node->owningGraph();
    // 42 : Scalar  -->  tensor(42.0) : Float([])
    auto t = g->insert(
        aten::as_tensor, {scalar}, {{"dtype", at::ScalarType::Float}});
    // tensor(42.0) : Float([])  -->  tensor([42.0]) : Float([1])
    auto unsqueezed = g->insert(aten::unsqueeze, {t, 0});
    node->replaceInput(1, unsqueezed);
  }
}

static void ConvertScalarToTensor(Block* block) {
  for (auto node : block->nodes()) {
    for (auto sub : node->blocks()) {
      ConvertScalarToTensor(sub);
    }

    if (node->kind() == aten::add || node->kind() == aten::mul) {
      mayConvertScalarInputToTensor(node);
    }
  }
}

void mayDecomposeAdd(Node* node) {
  if (node->inputs().size() < 3)
    return; // aten::add(int, int) may have only two inputs

  auto alphaEqualsOne = compareConstValue(node->input(2), 1.0);
  if (!alphaEqualsOne) {
    WithInsertPoint guard(node);
    auto g = node->owningGraph();
    auto mul = g->insert(aten::mul, {node->input(1), node->input(2)});
    node->replaceInput(1, mul);
    auto one = g->insertConstant(1.0);
    node->replaceInput(2, one);
  }
}

static void DecomposeFusedAdd(Block* block) {
  for (auto node : block->nodes()) {
    for (auto sub : node->blocks()) {
      DecomposeFusedAdd(sub);
    }

    if (node->kind() == aten::add) {
      mayDecomposeAdd(node);
    }
  }
}

static void EliminateIdentityMulAdd(Block* block) {
  for (auto node : block->nodes()) {
    for (auto sub : node->blocks()) {
      EliminateIdentityMulAdd(sub);
    }

    if ((node->kind() == aten::add && compareConstValue(node->input(1), 0.0)) ||
        (node->kind() == aten::mul && compareConstValue(node->input(1), 1.0))) {
      node->output()->replaceAllUsesWith(node->input(0));
    }
  }
}

void PrepareBinaryForLLGA(const std::shared_ptr<Graph>& graph) {
  DecomposeFusedAdd(graph->block());
  EliminateIdentityMulAdd(graph->block());
  EliminateDeadCode(graph);
  // ConvertScalarToTensor must be placed after EliminateIdentityMulAdd
  ConvertScalarToTensor(graph->block());
  PropagateInputShapes(graph);
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
