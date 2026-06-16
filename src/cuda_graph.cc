// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_graph.h"

#include <cuda_runtime_api.h>
#include <driver_types.h>

namespace trt_ep {

CudaGraphSet::~CudaGraphSet() {
  Clear();
}

void CudaGraphSet::Clear() {
  for (auto& it : cuda_graphs_) {
    cudaGraphExecDestroy(it.second);
  }
  cuda_graphs_.clear();
}

bool CudaGraphSet::Contains(CudaGraphAnnotation_t cuda_graph_annotation_id) const {
  return cuda_graphs_.find(cuda_graph_annotation_id) != cuda_graphs_.end();
}

void CudaGraphSet::Put(CudaGraphAnnotation_t cuda_graph_annotation_id, cudaGraphExec_t graph_exec) {
  if (Contains(cuda_graph_annotation_id)) {
    THROW("CUDA graph annotation id ", cuda_graph_annotation_id, " already exists.");
  }
  cuda_graphs_.emplace(cuda_graph_annotation_id, graph_exec);
}

cudaGraphExec_t CudaGraphSet::Get(CudaGraphAnnotation_t cuda_graph_annotation_id) const {
  if (!Contains(cuda_graph_annotation_id)) {
    THROW("CUDA graph annotation id ", cuda_graph_annotation_id, " not found.");
  }
  return cuda_graphs_.at(cuda_graph_annotation_id);
}

CudaGraphManager::CudaGraphManager(cudaStream_t stream) : stream_(stream) {
}

void CudaGraphManager::SetStream(cudaStream_t stream) {
  stream_ = stream;
}

void CudaGraphManager::CaptureBegin(CudaGraphAnnotation_t cuda_graph_annotation_id) {
  if (!IsGraphCaptureAllowedOnRun(cuda_graph_annotation_id)) {
    THROW("CUDA graph capture is not allowed on this run.");
  }

  if (cuda_graph_set_.Contains(cuda_graph_annotation_id)) {
    THROW("Trying to capture a graph with annotation id ", cuda_graph_annotation_id,
          " that already used. Please use a different annotation id.");
  }

  CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
  // Use cudaStreamCaptureModeThreadLocal to support multiple threads with
  // multiple graphs and streams (aligned with CUDA plugin EP).
  CUDA_CALL_THROW(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
}

void CudaGraphManager::CaptureEnd(CudaGraphAnnotation_t cuda_graph_annotation_id) {
  cudaGraph_t graph = nullptr;
  CUDA_CALL_THROW(cudaStreamEndCapture(stream_, &graph));
  if (graph == nullptr) {
    THROW("CudaGraphManager::CaptureEnd: graph is NULL");
  }

  cudaGraphExec_t graph_exec = nullptr;
  cudaError_t instantiate_err = cudaGraphInstantiate(&graph_exec, graph, 0);
  // Always destroy the graph definition, even if instantiate failed.
  cudaError_t destroy_err = cudaGraphDestroy(graph);

  if (instantiate_err != cudaSuccess) {
    THROW("cudaGraphInstantiate failed: ", cudaGetErrorString(instantiate_err));
  }
  if (destroy_err != cudaSuccess) {
    THROW("cudaGraphDestroy failed: ", cudaGetErrorString(destroy_err));
  }

  cuda_graph_set_.Put(cuda_graph_annotation_id, graph_exec);
}

OrtStatus* CudaGraphManager::Replay(CudaGraphAnnotation_t cuda_graph_annotation_id, bool sync) {
  cudaGraphExec_t graph_exec = cuda_graph_set_.Get(cuda_graph_annotation_id);
  RETURN_IF_ERROR(CUDA_CALL(cudaGraphLaunch(graph_exec, stream_)));
  if (sync) {
    RETURN_IF_ERROR(CUDA_CALL(cudaStreamSynchronize(stream_)));
  }
  return nullptr;
}

bool CudaGraphManager::IsGraphCaptureAllowedOnRun(CudaGraphAnnotation_t cuda_graph_annotation_id) const {
  return cuda_graph_annotation_id != kCudaGraphAnnotationSkip;
}

bool CudaGraphManager::IsGraphCaptured(CudaGraphAnnotation_t cuda_graph_annotation_id) const {
  return cuda_graph_set_.Contains(cuda_graph_annotation_id);
}

bool CudaGraphManager::IsGraphCaptureAllowed(CudaGraphAnnotation_t cuda_graph_annotation_id, int min_runs) const {
  if (!IsGraphCaptureAllowedOnRun(cuda_graph_annotation_id)) {
    return false;
  }
  auto it = run_count_.find(cuda_graph_annotation_id);
  if (it == run_count_.end()) {
    return false;
  }
  return it->second >= min_runs;
}

void CudaGraphManager::IncrementRegularRunCount(CudaGraphAnnotation_t cuda_graph_annotation_id) {
  auto& count = run_count_[cuda_graph_annotation_id];
  count++;
}

void CudaGraphManager::Reset() {
  cuda_graph_set_.Clear();
  run_count_.clear();
}

CudaGraphManager::~CudaGraphManager() {
  Reset();
}

}  // namespace trt_ep
