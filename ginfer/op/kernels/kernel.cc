#include "ginfer/op/kernels/kernel.h"

namespace ginfer::op::kernel {}  // namespace ginfer::op::kernel

namespace std {

size_t hash<ginfer::op::kernel::KernelInfo>::operator()(
    const ginfer::op::kernel::KernelInfo& k) const {
  size_t seed = 0;
  hash_combine(seed, k.name);
  hash_combine(seed, static_cast<int>(k.dev_type));
  hash_combine(seed, static_cast<int>(k.input_dtype));
  hash_combine(seed, static_cast<int>(k.output_dtype));
  return seed;
}

}  // namespace std