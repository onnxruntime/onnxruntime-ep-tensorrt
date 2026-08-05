// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <sstream>
#include <string>

#include "onnxruntime_c_api.h"

// AllocatorStats tracks memory usage by an allocator.
// Field names match those expected by ORT's allocator stats reporting.
struct AllocatorStats {
  int64_t num_allocs{0};
  int64_t num_reserves{0};
  int64_t num_arena_extensions{0};
  int64_t num_arena_shrinkages{0};
  int64_t bytes_in_use{0};
  int64_t bytes_requested_in_use{0};
  int64_t total_allocated_bytes{0};
  int64_t max_bytes_in_use{0};
  int64_t max_alloc_size{0};
  int64_t bytes_limit{0};

  void ToKeyValuePairs(const OrtApi& api, OrtKeyValuePairs* kvps) const {
    if (num_allocs > 0 || bytes_limit != 0) {
      api.AddKeyValuePair(kvps, "Limit", std::to_string(bytes_limit).c_str());
      api.AddKeyValuePair(kvps, "InUse", std::to_string(bytes_in_use).c_str());
      api.AddKeyValuePair(kvps, "RequestedInUse", std::to_string(bytes_requested_in_use).c_str());
      api.AddKeyValuePair(kvps, "TotalAllocated", std::to_string(total_allocated_bytes).c_str());
      api.AddKeyValuePair(kvps, "MaxInUse", std::to_string(max_bytes_in_use).c_str());
      api.AddKeyValuePair(kvps, "NumAllocs", std::to_string(num_allocs).c_str());
      api.AddKeyValuePair(kvps, "NumReserves", std::to_string(num_reserves).c_str());
      api.AddKeyValuePair(kvps, "NumArenaExtensions", std::to_string(num_arena_extensions).c_str());
      api.AddKeyValuePair(kvps, "NumArenaShrinkages", std::to_string(num_arena_shrinkages).c_str());
      api.AddKeyValuePair(kvps, "MaxAllocSize", std::to_string(max_alloc_size).c_str());
    }
  }

  std::string DebugString() const {
    std::ostringstream ss;
    ss << "Limit:                    " << bytes_limit << "\n"
       << "InUse:                    " << bytes_in_use << "\n"
       << "RequestedInUse:           " << bytes_requested_in_use << "\n"
       << "TotalAllocated:           " << total_allocated_bytes << "\n"
       << "MaxInUse:                 " << max_bytes_in_use << "\n"
       << "NumAllocs:                " << num_allocs << "\n"
       << "NumReserves:              " << num_reserves << "\n"
       << "NumArenaExtensions:       " << num_arena_extensions << "\n"
       << "NumArenaShrinkages:       " << num_arena_shrinkages << "\n"
       << "MaxAllocSize:             " << max_alloc_size << "\n";
    return ss.str();
  }
};

// BaseAllocator adds a virtual destructor to OrtAllocator (a C struct) so that derived
// types (e.g. CUDAAllocator, ArenaAllocator) can be deleted safely through a base pointer.
struct BaseAllocator : OrtAllocator {
  virtual ~BaseAllocator() = default;
};

using AllocatorUniquePtr = std::unique_ptr<BaseAllocator>;
