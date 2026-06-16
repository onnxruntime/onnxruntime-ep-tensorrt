// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Plugin-side CUDA graph manager. Manages cudaGraph_t / cudaGraphExec_t lifecycle
// for CUDA graph capture and replay in the plugin EP. Aligned with the CUDA plugin EP
// CudaGraphManager implementation.

#pragma once

#include <cuda_runtime_api.h>

#include <unordered_map>

#include "utils/ep_utils.h"
#include "utils/cuda/cuda_call.h"

namespace trt_ep {

using CudaGraphAnnotation_t = int;

constexpr CudaGraphAnnotation_t kCudaGraphAnnotationSkip = -1;
constexpr CudaGraphAnnotation_t kCudaGraphAnnotationDefault = 0;

// Storage for captured CUDA graph executables, keyed by annotation ID.
class CudaGraphSet {
 public:
  CudaGraphSet() = default;
  ~CudaGraphSet();

  void Clear();
  bool Contains(CudaGraphAnnotation_t cuda_graph_annotation_id) const;
  void Put(CudaGraphAnnotation_t cuda_graph_annotation_id, cudaGraphExec_t graph_exec);
  cudaGraphExec_t Get(CudaGraphAnnotation_t cuda_graph_annotation_id) const;

 private:
  std::unordered_map<CudaGraphAnnotation_t, cudaGraphExec_t> cuda_graphs_;
};

// Orchestrates CUDA graph capture, instantiation, and replay.
// Aligned with onnxruntime::cuda_plugin::CudaGraphManager.
class CudaGraphManager {
 public:
  CudaGraphManager() = default;
  explicit CudaGraphManager(cudaStream_t stream);
  ~CudaGraphManager();

  void SetStream(cudaStream_t stream);
  void CaptureBegin(CudaGraphAnnotation_t cuda_graph_annotation_id);
  void CaptureEnd(CudaGraphAnnotation_t cuda_graph_annotation_id);
  OrtStatus* Replay(CudaGraphAnnotation_t cuda_graph_annotation_id, bool sync = true);

  void Reset();

  bool IsGraphCaptureAllowedOnRun(CudaGraphAnnotation_t cuda_graph_annotation_id) const;
  bool IsGraphCaptured(CudaGraphAnnotation_t cuda_graph_annotation_id) const;

  // Warm-up tracking: per-annotation run counters
  bool IsGraphCaptureAllowed(CudaGraphAnnotation_t cuda_graph_annotation_id, int min_runs) const;
  void IncrementRegularRunCount(CudaGraphAnnotation_t cuda_graph_annotation_id);

 private:
  CudaGraphSet cuda_graph_set_;
  cudaStream_t stream_ = nullptr;
  std::unordered_map<CudaGraphAnnotation_t, int> run_count_;
};

}  // namespace trt_ep
