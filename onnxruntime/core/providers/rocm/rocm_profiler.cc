// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(USE_ROCM) && defined(ENABLE_ROCM_PROFILING)

#include "core/providers/rocm/rocm_profiler.h"

#include <time.h>

#include <chrono>

namespace onnxruntime {
namespace profiling {

RocmProfiler::RocmProfiler() {
  auto& manager = RoctracerManager::GetInstance();
  client_handle_ = manager.RegisterClient();
}

RocmProfiler::~RocmProfiler() {
  auto& manager = RoctracerManager::GetInstance();
  manager.DeregisterClient(client_handle_);
}

}  // namespace profiling
}  // namespace onnxruntime
#endif
