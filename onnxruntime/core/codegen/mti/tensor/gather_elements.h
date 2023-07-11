// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <tvm/tvm.h>

#include <string>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor GatherElements(const tvm::Tensor& t,
                           int64_t axis,
                           const tvm::Tensor& indices,
                           const std::string& name = "gather_elements");

}  // namespace tvm_codegen
}  // namespace onnxruntime
