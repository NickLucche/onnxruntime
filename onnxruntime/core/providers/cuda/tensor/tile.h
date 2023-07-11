// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/tile.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace cuda {

struct Tile final : CudaKernel {
  explicit Tile(const OpKernelInfo& info) : CudaKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};
}  // namespace cuda
}  // namespace onnxruntime
