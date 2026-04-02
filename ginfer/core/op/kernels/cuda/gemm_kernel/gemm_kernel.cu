#include "ginfer/common/device.h"
#include "ginfer/core/op/kernels/kernel_registry.h"
#include "ginfer/core/op/kernels/kernels.h"
#include "ginfer/core/tensor/tensor.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <glog/logging.h>
#include <cstdint>

namespace ginfer::core::op::kernel {

namespace {

template <typename T>
struct CublasTypeTraits;

template <>
struct CublasTypeTraits<__half> {
  static constexpr cudaDataType_t kCudaType = CUDA_R_16F;
};

template <>
struct CublasTypeTraits<__nv_bfloat16> {
  static constexpr cudaDataType_t kCudaType = CUDA_R_16BF;
};

inline cublasLtHandle_t getCublasLtHandle() {
  static thread_local cublasLtHandle_t handle = nullptr;
  if (handle == nullptr) {
    auto status = cublasLtCreate(&handle);
    CHECK(status == CUBLAS_STATUS_SUCCESS) << "Failed to create cuBLASLt handle.";
  }
  return handle;
}

}  // namespace

template <typename T, typename Context>
void gemmKernel(const Context& ctx,
                const tensor::Tensor& a,
                const tensor::Tensor& b,
                std::optional<std::reference_wrapper<const tensor::Tensor>> bias,
                tensor::Tensor& c) {
  CHECK(ctx.getDeviceType() == common::DeviceType::kDeviceCUDA)
      << "gemmKernel only supports CUDA device type.";

  const auto& cuda_ctx = dynamic_cast<const common::CUDADeviceContext&>(ctx);

  const auto& a_shape = a.shape();
  const auto& b_shape = b.shape();
  const auto& a_strides = a.strides();
  const auto& b_strides = b.strides();
  const auto& c_shape = c.shape();
  const auto& c_strides = c.strides();
  CHECK(a_shape.ndim() == 2 && b_shape.ndim() == 2 && c_shape.ndim() == 2)
      << "gemmKernel only supports 2D tensors.";
  CHECK(a_shape[1] == b_shape[0]) << "Inner dimensions of A and B must match.";
  CHECK(c_shape[0] == a_shape[0] && c_shape[1] == b_shape[1]) << "Output shape mismatch.";
  CHECK(a_strides[1] == 1) << "A must be row-major contiguous.";
  CHECK(b_strides[0] == 1) << "B must be in column-major order.";
  CHECK(c_strides[1] == 1) << "C must be row-major contiguous.";

  const tensor::Tensor* bias_tensor = nullptr;
  if (bias.has_value()) {
    bias_tensor = &bias->get();
    CHECK(bias_tensor->shape().ndim() == 1) << "Bias must be a 1D tensor.";
    CHECK(bias_tensor->shape()[0] == b_shape[1]) << "Bias size must match output columns.";
    CHECK(bias_tensor->strides().size() == 1 && bias_tensor->strides()[0] == 1)
        << "Bias must be contiguous.";
  }

  constexpr auto kCudaType = CublasTypeTraits<T>::kCudaType;
  const int M = static_cast<int>(a_shape[0]);
  const int K = static_cast<int>(a_shape[1]);
  const int N = static_cast<int>(b_shape[1]);
  const int rows = N;
  const int cols = M;
  const int inner_dim = K;
  const int ldb = K;
  const int lda = K;
  const int ldc = N;

  const T* a_data = reinterpret_cast<const T*>(a.data<T>());
  const T* b_data = reinterpret_cast<const T*>(b.data<T>());
  const T* bias_data =
      bias_tensor == nullptr ? nullptr : reinterpret_cast<const T*>(bias_tensor->data<T>());
  T* c_data = reinterpret_cast<T*>(c.data<T>());

  cublasLtHandle_t handle = getCublasLtHandle();
  cublasLtMatmulDesc_t operation_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;

  auto status = cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  CHECK(status == CUBLAS_STATUS_SUCCESS) << "Failed to create cuBLASLt matmul descriptor.";

  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  status = cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa,
                                          sizeof(transa));
  CHECK(status == CUBLAS_STATUS_SUCCESS) << "Failed to set cuBLASLt transa attribute.";
  status = cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb,
                                          sizeof(transb));
  CHECK(status == CUBLAS_STATUS_SUCCESS) << "Failed to set cuBLASLt transb attribute.";

  cublasLtEpilogue_t epilogue =
      bias_tensor == nullptr ? CUBLASLT_EPILOGUE_DEFAULT : CUBLASLT_EPILOGUE_BIAS;
  status = cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                          sizeof(epilogue));
  CHECK(status == CUBLAS_STATUS_SUCCESS) << "Failed to set cuBLASLt epilogue attribute.";
  if (bias_tensor != nullptr) {
    status = cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                            &bias_data, sizeof(bias_data));
    CHECK(status == CUBLAS_STATUS_SUCCESS) << "Failed to set cuBLASLt bias pointer.";
  }

  status = cublasLtMatrixLayoutCreate(&b_desc, kCudaType, inner_dim, rows, ldb);
  CHECK(status == CUBLAS_STATUS_SUCCESS) << "Failed to create B layout.";
  status = cublasLtMatrixLayoutCreate(&a_desc, kCudaType, inner_dim, cols, lda);
  CHECK(status == CUBLAS_STATUS_SUCCESS) << "Failed to create A layout.";
  status = cublasLtMatrixLayoutCreate(&c_desc, kCudaType, rows, cols, ldc);
  CHECK(status == CUBLAS_STATUS_SUCCESS) << "Failed to create C layout.";
  status = cublasLtMatrixLayoutCreate(&d_desc, kCudaType, rows, cols, ldc);
  CHECK(status == CUBLAS_STATUS_SUCCESS) << "Failed to create D layout.";

  uint64_t workspace_size = cuda_ctx.getWorkspaceSize();
  void* workspace = cuda_ctx.getWorkspace();
  CHECK(workspace != nullptr || workspace_size == 0) << "Invalid CUDA workspace.";

  cublasLtMatmulPreference_t preference = nullptr;
  status = cublasLtMatmulPreferenceCreate(&preference);
  CHECK(status == CUBLAS_STATUS_SUCCESS) << "Failed to create cuBLASLt preference.";
  status =
      cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                           &workspace_size, sizeof(workspace_size));
  CHECK(status == CUBLAS_STATUS_SUCCESS) << "Failed to set cuBLASLt workspace size.";

  cublasLtMatmulHeuristicResult_t heuristic_result;
  int returned_results = 0;
  status = cublasLtMatmulAlgoGetHeuristic(handle, operation_desc, b_desc, a_desc, c_desc, d_desc,
                                          preference, 1, &heuristic_result, &returned_results);
  CHECK(status == CUBLAS_STATUS_SUCCESS && returned_results > 0)
      << "Failed to get cuBLASLt heuristic.";

  float alpha = 1.0f;
  float beta = 0.0f;
  status = cublasLtMatmul(handle, operation_desc, &alpha, b_data, b_desc, a_data, a_desc, &beta,
                          c_data, c_desc, c_data, d_desc, &heuristic_result.algo, workspace,
                          workspace_size, cuda_ctx.getStream());
  CHECK(status == CUBLAS_STATUS_SUCCESS) << "cuBLASLt GEMM failed.";

  cublasLtMatmulPreferenceDestroy(preference);
  cublasLtMatrixLayoutDestroy(d_desc);
  cublasLtMatrixLayoutDestroy(c_desc);
  cublasLtMatrixLayoutDestroy(a_desc);
  cublasLtMatrixLayoutDestroy(b_desc);
  cublasLtMatmulDescDestroy(operation_desc);
}

REGISTER_KERNEL(gemm, CUDA, gemmKernel, Float16, BFloat16);

}  // namespace ginfer::core::op::kernel