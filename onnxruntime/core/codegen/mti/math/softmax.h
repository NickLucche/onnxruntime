#pragma once
#include <tvm/tvm.h>

#include <string>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Softmax(const tvm::Tensor& input, int64_t axis, const std::string& name = "softmax");

}  // namespace tvm_codegen
}  // namespace onnxruntime
