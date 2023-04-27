// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "GpuEvent.h"
#include "DmlCommandList.h"

namespace Dml
{
    class DmlAllocator;
    class CommandQueue;

    // A thread-safe object for recording and executing GPU commands. Calls to
    // this class are batched to minimize lock contention, and the batched
    // commands are periodically flushed by a background thread (or by explicitly
    // calling Flush). Unless otherwise stated, it is the caller's responsibility to
    // keep GPU resource arguments (buffers, heaps, DML ops, etc.) alive until they
    // have finished executing on the GPU; the returned GPU event, when signaled,
    // indicates when the submitted work has finished and the associated objects are
    // safe to release.
    class ExecutionContext
    {
    public:
        ExecutionContext(
            ID3D12Device* d3d12_device,
            IDMLDevice* dml_device,
            ID3D12CommandQueue* queue);

        ~ExecutionContext();

        // NOTE: the caller is responsible for keeping the dst_buffer/src_buffer
        // resources alive until the returned GPU event has completed.
        GpuEvent CopyBufferRegionRaw(
            ID3D12Resource* dst_buffer,
            uint64_t dst_offset,
            D3D12_RESOURCE_STATES dst_state,
            ID3D12Resource* src_buffer,
            uint64_t src_offset,
            D3D12_RESOURCE_STATES src_state,
            uint64_t byte_count);

        GpuEvent CopyBufferRegion(
            ID3D12Resource* dst_buffer,
            uint64_t dst_offset,
            D3D12_RESOURCE_STATES dst_state,
            ID3D12Resource* src_buffer,
            uint64_t src_offset,
            D3D12_RESOURCE_STATES src_state,
            uint64_t byte_count);

        // NOTE: the caller is responsible for keeping the dst resource alive until
        // the returned GPU event has completed. A copy of the value span will be
        // made, so the pointed-to value is safe to release immediately after
        // calling this method.
        GpuEvent FillBufferWithPatternRaw(
            ID3D12Resource* dst,
            uint64_t dst_offset,
            uint64_t dst_size_in_bytes,
            absl::Span<const std::byte> value);

        GpuEvent FillBufferWithPattern(
            ID3D12Resource* dstBuffer,
            gsl::span<const std::byte> value /* Data type agnostic value, treated as raw bits */);

        // NOTE: the caller is responsible for keeping the initializer and
        // descriptor_heap alive until the returned GPU event has completed. This
        // class takes ownership of the binding table.
        GpuEvent InitializeOperator(
            IDMLCompiledOperator* op,
            Microsoft::WRL::ComPtr<IDMLBindingTable>&& binding_table,
            ID3D12DescriptorHeap* descriptor_heap);

        // NOTE: the caller is responsible for keeping the op and descriptor_heap
        // alive until the returned GPU event has completed. This class takes
        // ownership of the binding table.
        GpuEvent ExecuteOperator(
            IDMLCompiledOperator* op,
            Microsoft::WRL::ComPtr<IDMLBindingTable>&& binding_table,
            ID3D12DescriptorHeap* descriptor_heap);

        // NOTE: A copy of the barriers span will be made, so the pointed-to value
        // is safe to release immediately after calling this method.
        GpuEvent ResourceBarrier(
            absl::Span<const D3D12_RESOURCE_BARRIER> barriers);

        // A slightly more efficient version of ResourceBarrier when the barrier
        // span only includes a UAV barrier (elides an extra copy).
        GpuEvent UavBarrier();

        // Indicates that any batched commands should be recorded and executed as
        // soon as possible, even if the batch is small. This is a no-op if nothing
        // is batched.
        GpuEvent Flush();

        Status GetCommandRecorderStatus() const;

        GpuEvent GetCurrentCompletionEvent();

        D3D12_COMMAND_LIST_TYPE GetCommandListTypeForQueue() const;

    private:
        static constexpr uint32_t default_batch_flush_size = 100;
        static constexpr uint32_t default_batch_flush_time_us = 1000;

        using Command = std::function<void(DmlCommandList&)>;
        using Batch = absl::InlinedVector<Command, default_batch_flush_size>;

        // State related to the batching of commands, which may be accessed by
        // both external threads (e.g. DML kernels) and the internal execution
        // thread.
        struct BatchState
        {
            std::mutex mutex;
            GpuEvent next_flush_event;
            std::condition_variable command_added;

            // Commands are double buffered: callers extend the "write batch" while
            // the execution thread records and executes the "execute batch".
            Batch batches[2];
            uint32_t write_batch_index = 0;
            Batch& WriteBatch() { return batches[write_batch_index]; }

            bool exit_requested = false;
            bool flush_requested = false;

            Status status;
        };

        std::shared_ptr<BatchState> batch_state_;
        std::shared_ptr<CommandQueue> dml_command_queue_;
        std::shared_ptr<DmlCommandList> dml_command_list_;
        std::thread execution_thread_;

        static void ExecutionThreadProc(
            std::shared_ptr<BatchState> batch_state,
            std::shared_ptr<DmlCommandList> command_list,
            std::shared_ptr<CommandQueue> command_queue,
            uint32_t batch_flush_size,
            uint32_t batch_flush_time_us);
    };
} // namespace Dml
