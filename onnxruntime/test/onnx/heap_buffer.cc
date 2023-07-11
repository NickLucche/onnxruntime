// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "heap_buffer.h"

#include "callback.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
namespace test {
void HeapBuffer::AddDeleter(const OrtCallback& d) {
  deleters_.push_back(d);
}

HeapBuffer::~HeapBuffer() {
  for (auto d : deleters_) {
    d.Run();
  }
}
}  // namespace test
}  // namespace onnxruntime
