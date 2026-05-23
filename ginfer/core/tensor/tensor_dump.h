#pragma once

#include <cstddef>
#include <iostream>
#include <string>

#include "ginfer/core/tensor/tensor.h"

namespace ginfer::core::tensor {

struct TensorDumpOptions {
  size_t max_elements = 64;
  bool include_data = true;
  bool copy_to_cpu = true;
};

std::string dumpTensor(const TensorRef& tensor, const TensorDumpOptions& options = {});

void printTensor(const TensorRef& tensor,
                 const TensorDumpOptions& options = {},
                 std::ostream& os = std::cerr);

}  // namespace ginfer::core::tensor
